use arrow::array::ArrayRef;
use arrow::datatypes::Field;
use arrow::error::ArrowError;

/// A transform that can be applied to a single column of a `RecordBatch`.
///
/// Implementations define how to convert an input column array into an output
/// array, and how the corresponding schema field should change.
pub trait ColumnTransform {
    /// Apply the transform to a column array, producing a new array.
    fn apply(&self, col: &ArrayRef) -> Result<ArrayRef, ArrowError>;
    /// Return the output field that results from transforming `input_field`.
    fn output_field(&self, input_field: &Field) -> Field;
}