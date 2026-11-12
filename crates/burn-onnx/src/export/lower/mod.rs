//! Lower resolved Burn export graphs into ONNX protobuf models.
//!
//! Lowering is intentionally independent from trace-based shape inference. The
//! module receives a [`crate::export::ResolvedExportGraph`], validates its value
//! bindings, initializes graph boundaries and embedded values, and then
//! dispatches each Burn operation to an operation-family lowerer.
//!
//! [`context::LoweringContext`] owns protobuf construction and deterministic
//! naming. The family modules only translate operations.

mod base;
mod context;
mod direct;
mod module;
mod numeric;

use std::collections::BTreeMap;

use burn::backend::ir::{OperationIr, ScalarIr, TensorId, TensorIr};
use burn::backend::{DType, TensorData, f16};
use context::LoweringContext;
use hashbrown::{HashMap, HashSet};
use onnx_ir::{GraphProto, TensorProto, TypeProto, ValueInfoProto};
use protobuf::MessageField;

use crate::export::{ExportError, OnnxModel, Opset, ResolvedExportGraph};

/// ONNX IR version emitted by this exporter.
const ONNX_IR_VERSION: i64 = 8;
/// Maximum protobuf payload supported by embedded ONNX tensor data.
const MAX_EMBEDDED_PROTOBUF_BYTES: u64 = i32::MAX as u64;

pub(crate) fn export_graph_with_bindings_and_opset(
    graph: &ResolvedExportGraph,
    values: &BTreeMap<TensorId, TensorData>,
    runtime_inputs: &[TensorId],
    initializer_names: &HashMap<TensorId, String>,
    opset: Opset,
) -> Result<OnnxModel, ExportError> {
    validate_bindings(graph, values, runtime_inputs, initializer_names)?;
    let runtime_set: HashSet<_> = runtime_inputs.iter().copied().collect();
    let embedded_bytes = values
        .iter()
        .filter(|(id, _)| !runtime_set.contains(*id))
        .map(|(_, value)| value.bytes().len() as u64)
        .sum::<u64>();
    if embedded_bytes > MAX_EMBEDDED_PROTOBUF_BYTES {
        return Err(ExportError::Serialization(format!(
            "embedded tensor data is {embedded_bytes} bytes, exceeding the protobuf limit of {MAX_EMBEDDED_PROTOBUF_BYTES} bytes"
        )));
    }

    let mut proto = GraphProto::new();
    proto.name = "burn_graph".into();
    for &id in &graph.graph.inputs {
        let tensor = find_tensor(graph, id).ok_or(ExportError::MissingValue(id))?;
        proto
            .input
            .push(value_info(tensor, &graph.dynamic_axes, initializer_names)?);
    }
    for &id in &graph.graph.outputs {
        let tensor = find_tensor(graph, id).ok_or(ExportError::MissingValue(id))?;
        let mut output = value_info(tensor, &graph.dynamic_axes, initializer_names)?;
        if graph.graph.inputs.contains(&id) {
            output.name = pass_through_output_name(id);
        }
        proto.output.push(output);
    }
    let mut initializers: Vec<_> = values
        .iter()
        .filter(|(id, _)| !runtime_set.contains(*id))
        .collect();
    initializers.sort_by(|(lhs, _), (rhs, _)| {
        tensor_name(**lhs, initializer_names).cmp(&tensor_name(**rhs, initializer_names))
    });
    for (&id, data) in initializers {
        let mut initializer = TensorProto::new();
        initializer.name = tensor_name(id, initializer_names);
        initializer.data_type = onnx_dtype_parts(id, data.dtype())?;
        initializer.dims = data.shape().iter().map(|dim| *dim as i64).collect();
        initializer.raw_data = bytes::Bytes::copy_from_slice(data.bytes().as_ref());
        proto.initializer.push(initializer);
    }

    let mut context = LoweringContext::new(graph, proto, initializer_names, opset);
    for (position, &id) in graph.graph.outputs.iter().enumerate() {
        if graph.graph.inputs.contains(&id) {
            let input = context.tensor_name(id);
            context.node(
                format!("output_{position}_identity"),
                "Identity",
                vec![input],
                vec![pass_through_output_name(id)],
            );
        }
    }
    for (index, operation) in graph.graph.operations.iter().enumerate() {
        lower_operation(&mut context, index, operation)?;
    }
    context.finish()
}

fn lower_operation(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<(), ExportError> {
    // Init records carry tensor metadata and initialized values, but do not represent an ONNX
    // computation. Keeping them in the graph also preserves metadata for pass-through boundaries.
    if matches!(operation, OperationIr::Init(_)) {
        return Ok(());
    }
    if numeric::lower(context, index, operation)?
        || base::lower(context, index, operation)?
        || module::lower(context, index, operation)?
        || direct::lower(context, index, operation)?
    {
        return Ok(());
    }
    Err(ExportError::UnsupportedOperation {
        operation: index,
        kind: format!("{operation:?}"),
    })
}

fn validate_bindings(
    graph: &ResolvedExportGraph,
    values: &BTreeMap<TensorId, TensorData>,
    runtime_inputs: &[TensorId],
    initializer_names: &HashMap<TensorId, String>,
) -> Result<(), ExportError> {
    let mut runtime = HashSet::new();
    for &id in runtime_inputs {
        if !runtime.insert(id) {
            return Err(ExportError::InvalidBoundary(format!(
                "duplicate runtime input tensor {id}"
            )));
        }
        if !graph.graph.inputs.contains(&id) {
            return Err(ExportError::InvalidBoundary(format!(
                "runtime input tensor {id} is not a declared graph input"
            )));
        }
    }
    if runtime.len() != graph.graph.inputs.len() {
        return Err(ExportError::InvalidBoundary(
            "every declared graph input must have one runtime input binding".into(),
        ));
    }
    for (&id, name) in initializer_names {
        if runtime.contains(&id) {
            return Err(ExportError::InvalidBoundary(format!(
                "runtime input tensor {id} cannot also have an initializer name"
            )));
        }
        if !values.contains_key(&id) {
            return Err(ExportError::MissingValue(id));
        }
        if name.is_empty() {
            return Err(ExportError::InvalidBoundary(format!(
                "initializer tensor {id} has an empty name"
            )));
        }
    }

    let mut names = HashMap::<String, TensorId>::new();
    let ids = graph
        .graph
        .operations
        .iter()
        .flat_map(OperationIr::nodes)
        .map(|tensor| tensor.id)
        .chain(values.keys().copied());
    for id in ids {
        let name = tensor_name(id, initializer_names);
        if let Some(previous) = names.insert(name.clone(), id)
            && previous != id
        {
            return Err(ExportError::InvalidBoundary(format!(
                "ONNX value name `{name}` is shared by tensors {previous} and {id}"
            )));
        }
    }

    for (&id, data) in values {
        let Some(tensor) = find_tensor(graph, id) else {
            continue;
        };
        if tensor.dtype != data.dtype() || tensor.shape != *data.shape() {
            return Err(ExportError::InvalidValue {
                tensor: id,
                reason: format!(
                    "graph metadata is {:?} {:?}, initialized value is {:?} {:?}",
                    tensor.dtype,
                    tensor.shape,
                    data.dtype(),
                    data.shape()
                ),
            });
        }
    }
    Ok(())
}

fn value_info(
    tensor: &TensorIr,
    dynamic_axes: &[crate::export::DynamicAxis],
    initializer_names: &HashMap<TensorId, String>,
) -> Result<ValueInfoProto, ExportError> {
    let mut info = ValueInfoProto::new();
    info.name = tensor_name(tensor.id, initializer_names);
    let mut ty = TypeProto::new();
    let tensor_type = ty.mut_tensor_type();
    tensor_type.elem_type = onnx_dtype_parts(tensor.id, tensor.dtype)?;
    let shape = tensor_type.shape.mut_or_insert_default();
    for (axis, &dim) in tensor.shape.iter().enumerate() {
        shape.dim.push(Default::default());
        let dimension = shape.dim.last_mut().unwrap();
        if let Some(dynamic) = dynamic_axes
            .iter()
            .find(|dynamic| dynamic.tensor == tensor.id && dynamic.axis == axis)
        {
            dimension.set_dim_param(dynamic.symbol.clone());
        } else {
            dimension.set_dim_value(dim as i64);
        }
    }
    info.type_ = MessageField::some(ty);
    Ok(info)
}

fn onnx_dtype_parts(tensor: TensorId, dtype: DType) -> Result<i32, ExportError> {
    // TensorProto.DataType numeric values from the ONNX specification.
    match dtype {
        DType::F32 => Ok(1),
        DType::U8 => Ok(2),
        DType::I8 => Ok(3),
        DType::U16 => Ok(4),
        DType::I16 => Ok(5),
        DType::I32 => Ok(6),
        DType::I64 => Ok(7),
        DType::Bool(_) => Ok(9),
        DType::F16 => Ok(10),
        DType::F64 => Ok(11),
        DType::U32 => Ok(12),
        DType::U64 => Ok(13),
        DType::BF16 => Ok(16),
        dtype => Err(ExportError::UnsupportedDType {
            tensor,
            dtype: format!("{dtype:?}"),
        }),
    }
}

fn find_tensor(graph: &ResolvedExportGraph, id: TensorId) -> Option<&TensorIr> {
    graph
        .graph
        .operations
        .iter()
        .flat_map(OperationIr::nodes)
        .find(|tensor| tensor.id == id)
}

fn name(id: TensorId) -> String {
    format!("tensor_{}", id.value())
}

fn pass_through_output_name(id: TensorId) -> String {
    format!("{}_output", name(id))
}

fn tensor_name(id: TensorId, initializer_names: &HashMap<TensorId, String>) -> String {
    initializer_names
        .get(&id)
        .filter(|name| !name.is_empty())
        .cloned()
        .unwrap_or_else(|| name(id))
}

fn scalar_tensor(
    dtype: DType,
    value: ScalarIr,
    tensor: TensorId,
) -> Result<TensorProto, ExportError> {
    let mut initializer = TensorProto::new();
    initializer.data_type = onnx_dtype_parts(tensor, dtype)?;
    initializer.dims = vec![1];
    let bytes = match dtype {
        DType::F32 => value.elem::<f32>().to_le_bytes().to_vec(),
        DType::F64 => value.elem::<f64>().to_le_bytes().to_vec(),
        DType::F16 => value.elem::<f16>().to_le_bytes().to_vec(),
        DType::I8 => value.elem::<i8>().to_le_bytes().to_vec(),
        DType::I16 => value.elem::<i16>().to_le_bytes().to_vec(),
        DType::I32 => value.elem::<i32>().to_le_bytes().to_vec(),
        DType::I64 => value.elem::<i64>().to_le_bytes().to_vec(),
        DType::U8 => value.elem::<u8>().to_le_bytes().to_vec(),
        DType::U16 => value.elem::<u16>().to_le_bytes().to_vec(),
        DType::U32 => value.elem::<u32>().to_le_bytes().to_vec(),
        DType::U64 => value.elem::<u64>().to_le_bytes().to_vec(),
        DType::Bool(_) => vec![value.elem::<bool>() as u8],
        dtype => {
            return Err(ExportError::UnsupportedDType {
                tensor,
                dtype: format!("{dtype:?} scalar initializer"),
            });
        }
    };
    initializer.raw_data = bytes::Bytes::from(bytes);
    Ok(initializer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Shape;
    use burn::backend::ir::{
        ActivationOperationIr, BaseOperationIr, BinaryOpIr, CastOpIr, ClampOpIr, DimOpIr,
        FloatOperationIr, GraphIr, IntOperationIr, NumericOperationIr, PermuteOpIr, ReduceDimOpIr,
        ReduceDimWithIndicesOpIr, ScalarIr, ShapeOpIr, SliceOpIr, SwapDimsOpIr, UnaryOpIr,
    };
    use onnx_ir::{ModelProto, NodeProto};
    use protobuf::Message;

    #[test]
    fn lowers_dynamic_slice_with_shape_derived_ends() {
        // positions_table[4, 3] sliced to [0..t): the captured bound is the
        // sample trace's t, so the resolved dimension is dynamic and `ends`
        // must be computed at runtime (Shape -> Gather), not baked.
        let input = TensorIr::uninit(TensorId::new(1), Shape::new([4, 3]), DType::F32);
        let output = TensorIr::uninit(TensorId::new(2), Shape::new([2, 3]), DType::F32);
        let graph = GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::Slice(
            SliceOpIr {
                tensor: input,
                ranges: vec![burn::backend::Slice {
                    start: 0,
                    end: Some(2),
                    step: 1,
                }],
                out: output,
            },
        ))]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![crate::export::resolved::ResolvedShape {
                operation: 0,
                tensor: TensorId::new(2),
                dimensions: vec![
                    crate::export::resolved::ShapeExpr::InputDim {
                        input: TensorId::new(1),
                        axis: 0,
                    },
                    crate::export::resolved::ShapeExpr::Static(3),
                ],
            }],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        let slice = model
            .graph
            .node
            .iter()
            .find(|node| node.op_type == "Slice")
            .unwrap();
        let ends = &slice.input[2];
        assert!(
            !model
                .graph
                .initializer
                .iter()
                .any(|initializer| initializer.name == *ends),
            "dynamic slice ends must be a computed tensor, not an initializer"
        );
        let ends_concat = model
            .graph
            .node
            .iter()
            .find(|node| node.op_type == "Concat" && node.output.contains(ends))
            .expect("ends is assembled by a Concat");
        assert!(
            ends_concat
                .input
                .iter()
                .any(|part| model
                    .graph
                    .node
                    .iter()
                    .any(|node| node.op_type == "Gather" && node.output.contains(part))),
            "ends derives from a Shape -> Gather chain"
        );
    }

    fn tensor(id: u64) -> TensorIr {
        TensorIr::uninit(TensorId::new(id), Shape::new([2, 3]), DType::F32)
    }

    fn perm_attribute(node: &NodeProto) -> Vec<i64> {
        node.attribute
            .iter()
            .find(|attribute| attribute.name == "perm")
            .map(|attribute| attribute.ints.clone())
            .unwrap_or_default()
    }

    #[test]
    fn serializes_all_constant_of_shape_opset_18_scalar_dtypes() {
        let cases = [
            (DType::F16, vec![0, 60]),
            (DType::F32, vec![0, 0, 128, 63]),
            (DType::F64, vec![0, 0, 0, 0, 0, 0, 240, 63]),
            (DType::I8, vec![1]),
            (DType::I16, vec![1, 0]),
            (DType::I32, vec![1, 0, 0, 0]),
            (DType::I64, vec![1, 0, 0, 0, 0, 0, 0, 0]),
            (DType::U8, vec![1]),
            (DType::U16, vec![1, 0]),
            (DType::U32, vec![1, 0, 0, 0]),
            (DType::U64, vec![1, 0, 0, 0, 0, 0, 0, 0]),
            (DType::Bool(burn::backend::BoolStore::Native), vec![1]),
        ];

        for (dtype, expected) in cases {
            let tensor = TensorId::new(1);
            let scalar = scalar_tensor(dtype, ScalarIr::new(1, &dtype), tensor).unwrap();

            assert_eq!(scalar.data_type, onnx_dtype_parts(tensor, dtype).unwrap());
            assert_eq!(scalar.raw_data.as_ref(), expected);
        }
    }

    #[test]
    fn writes_parseable_opset_18_model() {
        let graph = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Add(BinaryOpIr {
                lhs: tensor(1),
                rhs: tensor(2),
                out: tensor(3),
            }),
        )]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();
        assert_eq!(model.ir_version, ONNX_IR_VERSION);
        assert_eq!(model.opset_import[0].version, Opset::default().version());
        assert_eq!(model.graph.node[0].op_type, "Add");
        assert_eq!(model.graph.input.len(), 2);
        assert_eq!(model.graph.output.len(), 1);
    }

    #[test]
    fn casts_non_i64_topk_indices_without_swapping_outputs() {
        let input = tensor(1);
        let output = TensorIr::uninit(TensorId::new(2), Shape::new([2, 1]), DType::F32);
        let indices = TensorIr::uninit(TensorId::new(3), Shape::new([2, 1]), DType::I32);
        let graph = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::MaxDimWithIndices(ReduceDimWithIndicesOpIr {
                tensor: input,
                dim: 1,
                out: output,
                out_indices: indices,
            }),
        )]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        assert_eq!(model.graph.node[0].op_type, "TopK");
        assert_eq!(model.graph.node[0].output[0], "tensor_2");
        assert_eq!(model.graph.node[0].output[1], "node_0_indices64");
        assert_eq!(model.graph.node[1].op_type, "Cast");
        assert_eq!(model.graph.node[1].input, ["node_0_indices64"]);
        assert_eq!(model.graph.node[1].output, ["tensor_3"]);
        assert_eq!(
            model.graph.node[1]
                .attribute
                .iter()
                .find(|attribute| attribute.name == "to")
                .unwrap()
                .i,
            6
        );
    }

    #[test]
    fn lowers_max_and_min_dim_as_reduce_nodes() {
        for (operation, op_type) in [
            (
                NumericOperationIr::MaxDim(ReduceDimOpIr {
                    input: tensor(1),
                    out: TensorIr::uninit(TensorId::new(2), Shape::new([2, 1]), DType::F32),
                    axis: 1,
                    accumulator_len: 3,
                }),
                "ReduceMax",
            ),
            (
                NumericOperationIr::MinDim(ReduceDimOpIr {
                    input: tensor(1),
                    out: TensorIr::uninit(TensorId::new(2), Shape::new([2, 1]), DType::F32),
                    axis: 1,
                    accumulator_len: 3,
                }),
                "ReduceMin",
            ),
        ] {
            let graph = GraphIr::new(vec![OperationIr::NumericFloat(DType::F32, operation)]);
            let graph = ResolvedExportGraph {
                graph,
                shapes: vec![],
                dynamic_axes: vec![],
            };
            let runtime_inputs = graph.graph.inputs.clone();
            let model = export_graph_with_bindings_and_opset(
                &graph,
                &BTreeMap::new(),
                &runtime_inputs,
                &HashMap::new(),
                Opset::default(),
            )
            .unwrap();
            let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

            assert_eq!(model.graph.node.len(), 1);
            assert_eq!(model.graph.node[0].op_type, op_type);
            assert_eq!(model.graph.node[0].input, ["tensor_1", "node_0_axes"]);
            assert_eq!(model.graph.node[0].output, ["tensor_2"]);
            assert_eq!(
                model.graph.node[0]
                    .attribute
                    .iter()
                    .find(|attribute| attribute.name == "keepdims")
                    .unwrap()
                    .i,
                1
            );
        }
    }

    #[test]
    fn lowers_mean_dim_as_reduce_mean_node() {
        let input = tensor(1);
        let output = TensorIr::uninit(TensorId::new(2), Shape::new([2, 1]), DType::F32);
        let graph = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::MeanDim(ReduceDimOpIr {
                input,
                out: output,
                axis: 1,
                accumulator_len: 3,
            }),
        )]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        assert_eq!(model.graph.node.len(), 1);
        assert_eq!(model.graph.node[0].op_type, "ReduceMean");
        assert_eq!(model.graph.node[0].input, ["tensor_1", "node_0_axes"]);
        assert_eq!(model.graph.node[0].output, ["tensor_2"]);
        let axes = model
            .graph
            .initializer
            .iter()
            .find(|initializer| initializer.name == "node_0_axes")
            .unwrap();
        assert_eq!(axes.dims, [1]);
        assert_eq!(axes.raw_data.as_ref(), 1_i64.to_le_bytes());
        assert_eq!(
            model.graph.node[0]
                .attribute
                .iter()
                .find(|attribute| attribute.name == "keepdims")
                .unwrap()
                .i,
            1
        );
    }

    #[test]
    fn lowers_softmax_with_axis_attribute() {
        let input = tensor(1);
        let output = TensorIr::uninit(TensorId::new(2), Shape::new([2, 3]), DType::F32);
        let graph = GraphIr::new(vec![OperationIr::Activation(
            ActivationOperationIr::Softmax(DimOpIr {
                input,
                out: output,
                axis: 1,
            }),
        )]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        assert_eq!(model.graph.node.len(), 1);
        assert_eq!(model.graph.node[0].op_type, "Softmax");
        assert_eq!(model.graph.node[0].input, ["tensor_1"]);
        assert_eq!(model.graph.node[0].output, ["tensor_2"]);
        assert_eq!(
            model.graph.node[0]
                .attribute
                .iter()
                .find(|attribute| attribute.name == "axis")
                .unwrap()
                .i,
            1
        );
    }

    #[test]
    fn lowers_log_softmax_and_softmin_with_axis_attribute() {
        for (operation, op_type) in [
            (
                ActivationOperationIr::LogSoftmax(DimOpIr {
                    input: tensor(1),
                    out: TensorIr::uninit(TensorId::new(2), Shape::new([2, 3]), DType::F32),
                    axis: 0,
                }),
                "LogSoftmax",
            ),
            (
                ActivationOperationIr::Softmin(DimOpIr {
                    input: tensor(1),
                    out: TensorIr::uninit(TensorId::new(2), Shape::new([2, 3]), DType::F32),
                    axis: 1,
                }),
                "Softmin",
            ),
        ] {
            let graph = GraphIr::new(vec![OperationIr::Activation(operation)]);
            let graph = ResolvedExportGraph {
                graph,
                shapes: vec![],
                dynamic_axes: vec![],
            };
            let runtime_inputs = graph.graph.inputs.clone();
            let model = export_graph_with_bindings_and_opset(
                &graph,
                &BTreeMap::new(),
                &runtime_inputs,
                &HashMap::new(),
                Opset::default(),
            )
            .unwrap();
            let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

            assert_eq!(model.graph.node.len(), 1);
            assert_eq!(model.graph.node[0].op_type, op_type);
            assert!(
                model.graph.node[0]
                    .attribute
                    .iter()
                    .any(|attribute| attribute.name == "axis")
            );
        }
    }

    #[test]
    fn lowers_recip_as_reciprocal_node() {
        let graph = GraphIr::new(vec![OperationIr::Float(
            DType::F32,
            FloatOperationIr::Recip(UnaryOpIr {
                input: tensor(1),
                out: tensor(2),
            }),
        )]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        assert_eq!(model.graph.node.len(), 1);
        assert_eq!(model.graph.node[0].op_type, "Reciprocal");
        assert_eq!(model.graph.node[0].input, ["tensor_1"]);
        assert_eq!(model.graph.node[0].output, ["tensor_2"]);
    }

    #[test]
    fn lowers_expand_with_resolved_shape_input() {
        let input = tensor(1);
        let output = TensorIr::uninit(TensorId::new(2), Shape::new([2, 3]), DType::F32);
        let graph = GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::Expand(
            ShapeOpIr { input, out: output },
        ))]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![crate::export::resolved::ResolvedShape {
                operation: 0,
                tensor: TensorId::new(2),
                dimensions: vec![
                    crate::export::resolved::ShapeExpr::Static(2),
                    crate::export::resolved::ShapeExpr::Static(3),
                ],
            }],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        assert_eq!(model.graph.node.len(), 1);
        assert_eq!(model.graph.node[0].op_type, "Expand");
        assert_eq!(model.graph.node[0].input, ["tensor_1", "node_0_shape"]);
        assert_eq!(model.graph.node[0].output, ["tensor_2"]);
        let shape = model
            .graph
            .initializer
            .iter()
            .find(|initializer| initializer.name == "node_0_shape")
            .unwrap();
        assert_eq!(shape.dims, [2]);
        assert_eq!(
            shape.raw_data.as_ref(),
            [2_i64, 3_i64]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
        );
    }

    #[test]
    fn lowers_int_into_float_as_cast_node() {
        let input = TensorIr::uninit(TensorId::new(1), Shape::new([2, 3]), DType::I64);
        let output = TensorIr::uninit(TensorId::new(2), Shape::new([2, 3]), DType::F32);
        let graph = GraphIr::new(vec![OperationIr::Int(IntOperationIr::IntoFloat(
            CastOpIr { input, out: output },
        ))]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        assert_eq!(model.graph.node.len(), 1);
        assert_eq!(model.graph.node[0].op_type, "Cast");
        assert_eq!(model.graph.node[0].input, ["tensor_1"]);
        assert_eq!(model.graph.node[0].output, ["tensor_2"]);
        assert_eq!(
            model.graph.node[0]
                .attribute
                .iter()
                .find(|attribute| attribute.name == "to")
                .unwrap()
                .i,
            1
        );
    }

    #[test]
    fn lowers_clamp_as_clip_with_scalar_bounds() {
        let input = tensor(1);
        let output = TensorIr::uninit(TensorId::new(2), Shape::new([2, 3]), DType::F32);
        let graph = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Clamp(ClampOpIr {
                tensor: input,
                min: ScalarIr::new(-1.0, &DType::F32),
                max: ScalarIr::new(1.0, &DType::F32),
                out: output,
            }),
        )]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        assert_eq!(model.graph.node.len(), 1);
        assert_eq!(model.graph.node[0].op_type, "Clip");
        assert_eq!(
            model.graph.node[0].input,
            ["tensor_1", "node_0_min", "node_0_max"]
        );
        assert_eq!(model.graph.node[0].output, ["tensor_2"]);
        for (name, expected) in [("node_0_min", -1.0f32), ("node_0_max", 1.0f32)] {
            let bound = model
                .graph
                .initializer
                .iter()
                .find(|initializer| initializer.name == name)
                .unwrap();
            assert_eq!(bound.raw_data.as_ref(), expected.to_le_bytes());
        }
    }

    #[test]
    fn lowers_gelu_as_erf_decomposition() {
        let graph = GraphIr::new(vec![OperationIr::Activation(ActivationOperationIr::Gelu(
            UnaryOpIr {
                input: tensor(1),
                out: tensor(2),
            },
        ))]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        let ops = model
            .graph
            .node
            .iter()
            .map(|node| node.op_type.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ops, ["Div", "Erf", "Add", "Mul", "Mul"]);
        assert_eq!(model.graph.node[0].input, ["tensor_1", "node_0_sqrt2"]);
        assert_eq!(model.graph.node[1].input, ["node_0_scaled"]);
        assert_eq!(model.graph.node[2].input, ["node_0_erf", "node_0_one"]);
        assert_eq!(model.graph.node[3].input, ["tensor_1", "node_0_gate"]);
        assert_eq!(model.graph.node[4].input, ["node_0_gated", "node_0_half"]);
        assert_eq!(model.graph.node[4].output, ["tensor_2"]);
    }

    #[test]
    fn lowers_swap_dims_as_transpose_with_swapped_perm() {
        let input = tensor(1);
        let output = TensorIr::uninit(TensorId::new(2), Shape::new([3, 2]), DType::F32);
        let graph = GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::SwapDims(
            SwapDimsOpIr {
                input,
                out: output,
                dim1: 0,
                dim2: 1,
            },
        ))]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        assert_eq!(model.graph.node.len(), 1);
        assert_eq!(model.graph.node[0].op_type, "Transpose");
        assert_eq!(model.graph.node[0].input, ["tensor_1"]);
        assert_eq!(model.graph.node[0].output, ["tensor_2"]);
        assert_eq!(perm_attribute(&model.graph.node[0]), [1, 0]);
    }

    #[test]
    fn lowers_permute_as_transpose_with_verbatim_axes() {
        let input = tensor(1);
        let output = TensorIr::uninit(TensorId::new(2), Shape::new([2, 3]), DType::F32);
        let graph = GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::Permute(
            PermuteOpIr {
                input,
                out: output,
                axes: vec![2, 0, 1],
            },
        ))]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        assert_eq!(model.graph.node.len(), 1);
        assert_eq!(model.graph.node[0].op_type, "Transpose");
        assert_eq!(model.graph.node[0].input, ["tensor_1"]);
        assert_eq!(model.graph.node[0].output, ["tensor_2"]);
        assert_eq!(perm_attribute(&model.graph.node[0]), [2, 0, 1]);
    }

    #[test]
    fn casts_non_i64_arg_reduction_output() {
        let input = tensor(1);
        let output = TensorIr::uninit(TensorId::new(2), Shape::new([2, 1]), DType::I32);
        let graph = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::ArgMax(ReduceDimOpIr {
                input,
                out: output,
                axis: 1,
                accumulator_len: 3,
            }),
        )]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();

        assert_eq!(model.graph.node[0].op_type, "ArgMax");
        assert_eq!(model.graph.node[0].output, ["node_0_indices64"]);
        assert_eq!(model.graph.node[1].op_type, "Cast");
        assert_eq!(model.graph.node[1].input, ["node_0_indices64"]);
        assert_eq!(model.graph.node[1].output, ["tensor_2"]);
    }
}
