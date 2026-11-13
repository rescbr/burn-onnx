//! Lowering for operations that map directly to one ONNX node.
//!
//! Input and output ordering is taken directly from [`burn::backend::ir::OperationIr`].

use burn::backend::ir::{
    ActivationOperationIr, DimOpIr, FloatOperationIr, IntOperationIr, OperationIr,
};

use crate::export::ExportError;

use super::context::LoweringContext;

/// Lower operations whose ONNX form has the same tensor inputs and outputs.
pub(super) fn lower(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<bool, ExportError> {
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
    context.node(format!("node_{index}_neg"), "Neg", vec![input], vec![negated.clone()]);
    context.node(format!("node_{index}"), "Softmax", vec![negated], vec![output]);
    context.int_attribute("axis", operation.axis as i64);
    Ok(())
}

/// Lower a per-axis normalized exponential (`Softmax`, `LogSoftmax`) whose
/// opset 13+ form carries the axis as an attribute.
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
