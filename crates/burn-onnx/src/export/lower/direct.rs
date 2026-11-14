//! Lowering for operations that map directly to one ONNX node.
//!
//! Input and output ordering is taken directly from [`burn::backend::ir::OperationIr`].

use burn::backend::ir::{
    ActivationOperationIr, BoolOperationIr, FloatOperationIr, IntOperationIr, OperationIr,
};

use crate::export::ExportError;

use super::{context::LoweringContext, onnx_dtype_parts};

/// Lower operations whose ONNX form has the same tensor inputs and outputs.
pub(super) fn lower(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<bool, ExportError> {
    if let Some(cast) = match operation {
        OperationIr::Int(IntOperationIr::IntoFloat(cast)) => Some(cast),
        OperationIr::Bool(BoolOperationIr::IntoFloat(cast)) => Some(cast),
        _ => None,
    } {
        let input = context.tensor_name(cast.input.id);
        let output = context.tensor_name(cast.out.id);
        context.node(format!("node_{index}"), "Cast", vec![input], vec![output]);
        context.int_attribute("to", onnx_dtype_parts(cast.out.id, cast.out.dtype)? as i64);
        return Ok(true);
    }
    let op_type = match operation {
        OperationIr::Float(_, operation) => match operation {
            FloatOperationIr::Exp(_) => "Exp",
            FloatOperationIr::Log(_) => "Log",
            FloatOperationIr::Sqrt(_) => "Sqrt",
            FloatOperationIr::Tanh(_) => "Tanh",
            FloatOperationIr::Matmul(_) => "MatMul",
            _ => return Ok(false),
        },
        OperationIr::Int(IntOperationIr::Matmul(_)) => "MatMul",
        OperationIr::Activation(operation) => match operation {
            ActivationOperationIr::Relu(_) => "Relu",
            ActivationOperationIr::Sigmoid(_) => "Sigmoid",
            _ => return Ok(false),
        },
        _ => return Ok(false),
    };
    let inputs = operation
        .inputs()
        .map(|tensor| context.tensor_name(tensor.id))
        .collect();
    let outputs = operation
        .outputs()
        .map(|tensor| context.tensor_name(tensor.id))
        .collect();
    context.node(format!("node_{index}"), op_type, inputs, outputs);
    Ok(true)
}
