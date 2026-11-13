//! Lowering for operations that map directly to one ONNX node.
//!
//! Input and output ordering is taken directly from [`burn::backend::ir::OperationIr`].

use burn::backend::ir::{
    ActivationOperationIr, BoolOperationIr, DimOpIr, FloatOperationIr, IntOperationIr, OperationIr,
    ScalarIr, UnaryOpIr,
};

use crate::export::ExportError;

use super::{context::LoweringContext, onnx_dtype_parts};

/// Lower operations whose ONNX form has the same tensor inputs and outputs.
pub(super) fn lower(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<bool, ExportError> {
    if let OperationIr::Activation(ActivationOperationIr::Gelu(operation)) = operation {
        lower_gelu(context, index, operation)?;
        return Ok(true);
    }
    if let Some((operation, op_type)) = match operation {
        OperationIr::Activation(ActivationOperationIr::Softmax(operation)) => {
            Some((operation, "Softmax"))
        }
        OperationIr::Activation(ActivationOperationIr::LogSoftmax(operation)) => {
            Some((operation, "LogSoftmax"))
        }
        _ => None,
    } {
        lower_normalized_exponential(context, index, operation, op_type)?;
        return Ok(true);
    }
    if let OperationIr::Activation(ActivationOperationIr::Softmin(operation)) = operation {
        lower_softmin(context, index, operation)?;
        return Ok(true);
    }
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
            FloatOperationIr::Recip(_) => "Reciprocal",
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

/// Lower `Softmin` as `Neg` followed by `Softmax` (`softmin(x) = softmax(-x)`);
/// ONNX has no `Softmin` operator — it never existed in any opset.
fn lower_softmin(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &DimOpIr,
) -> Result<(), ExportError> {
    let input = context.tensor_name(operation.input.id);
    let negated = format!("node_{index}_neg_out");
    let output = context.tensor_name(operation.out.id);
    context.node(
        format!("node_{index}_neg"),
        "Neg",
        vec![input],
        vec![negated.clone()],
    );
    context.node(
        format!("node_{index}"),
        "Softmax",
        vec![negated],
        vec![output],
    );
    context.int_attribute("axis", operation.axis as i64);
    Ok(())
}

/// Lower a per-axis normalized exponential (`Softmax`, `LogSoftmax`)
/// whose opset 13+ form carries the axis as an attribute.
fn lower_normalized_exponential(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &DimOpIr,
    op_type: &'static str,
) -> Result<(), ExportError> {
    let input = context.tensor_name(operation.input.id);
    let output = context.tensor_name(operation.out.id);
    context.node(format!("node_{index}"), op_type, vec![input], vec![output]);
    context.int_attribute("axis", operation.axis as i64);
    Ok(())
}

/// Lower `Gelu` via its exact erf form `0.5 * x * (1 + erf(x / sqrt(2)))`.
///
/// ONNX `Gelu` is opset-20 and the exporter pins opset 18, so the native
/// node cannot be emitted yet; `Erf` is available from opset 13 onward.
fn lower_gelu(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &UnaryOpIr,
) -> Result<(), ExportError> {
    let input = context.tensor_name(operation.input.id);
    let output = context.tensor_name(operation.out.id);
    let dtype = operation.input.dtype;

    let sqrt2_name = format!("node_{index}_sqrt2");
    context.scalar_initializer(
        sqrt2_name.clone(),
        dtype,
        ScalarIr::Float(core::f64::consts::SQRT_2),
        operation.input.id,
    )?;
    let scaled_name = format!("node_{index}_scaled");
    context.node(
        scaled_name.clone(),
        "Div",
        vec![input.clone(), sqrt2_name],
        vec![scaled_name.clone()],
    );

    let erf_name = format!("node_{index}_erf");
    context.node(
        erf_name.clone(),
        "Erf",
        vec![scaled_name],
        vec![erf_name.clone()],
    );

    let one_name = format!("node_{index}_one");
    context.scalar_initializer(
        one_name.clone(),
        dtype,
        ScalarIr::Float(1.0),
        operation.input.id,
    )?;
    let gate_name = format!("node_{index}_gate");
    context.node(
        gate_name.clone(),
        "Add",
        vec![erf_name, one_name],
        vec![gate_name.clone()],
    );

    let gated_name = format!("node_{index}_gated");
    context.node(
        gated_name.clone(),
        "Mul",
        vec![input, gate_name],
        vec![gated_name.clone()],
    );

    let half_name = format!("node_{index}_half");
    context.scalar_initializer(
        half_name.clone(),
        dtype,
        ScalarIr::Float(0.5),
        operation.input.id,
    )?;
    context.node(
        format!("node_{index}"),
        "Mul",
        vec![gated_name, half_name],
        vec![output],
    );
    Ok(())
}
