use arrow::array::{
    Array, ArrayRef, GenericListArray, GenericListBuilder, LargeStringBuilder, OffsetSizeTrait,
};
use arrow::datatypes::{DataType, Field};
use arrow::error::ArrowError;
use arrow_json::writer::{EncoderOptions, make_encoder};
use std::sync::Arc;

use crate::ColumnTransform;

/// A [`ColumnTransform`] that serializes columns to their JSON representation.
///
/// Scalar columns are converted to a `LargeUtf8` array where each element is the
/// JSON encoding of the original value. List columns become a
/// `List<LargeUtf8>` (or `LargeList<LargeUtf8>`) preserving the list
/// structure, with each leaf value JSON-encoded. Nulls are preserved.
pub struct ToJson;

impl ColumnTransform for ToJson {
    fn apply(&self, col: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        match col.data_type() {
            DataType::List(_) => serialize_list::<i32>(col),
            DataType::LargeList(_) => serialize_list::<i64>(col),
            _ => serialize_scalar(col),
        }
    }

    fn output_field(&self, input_field: &Field) -> Field {
        let output_type = match input_field.data_type() {
            DataType::List(_) => {
                DataType::List(Arc::new(Field::new("item", DataType::LargeUtf8, true)))
            }
            DataType::LargeList(_) => {
                DataType::LargeList(Arc::new(Field::new("item", DataType::LargeUtf8, true)))
            }
            _ => DataType::LargeUtf8,
        };
        Field::new(input_field.name(), output_type, input_field.is_nullable())
    }
}

fn serialize_list<O: OffsetSizeTrait>(col: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    let list_array = col
        .as_any()
        .downcast_ref::<GenericListArray<O>>()
        .ok_or_else(|| {
            ArrowError::InvalidArgumentError("failed to downcast to GenericListArray".to_owned())
        })?;

    let field = match list_array.data_type() {
        DataType::List(f) | DataType::LargeList(f) => f,
        _ => unreachable!(),
    };

    let inner = list_array.values();
    let options = EncoderOptions::default();
    let mut encoder = make_encoder(field, inner.as_ref(), &options)?;

    let offsets = list_array.offsets();
    let mut list_builder = GenericListBuilder::<O, _>::new(LargeStringBuilder::new());
    let mut buf: Vec<u8> = Vec::new();

    for i in 0..list_array.len() {
        if list_array.is_null(i) {
            list_builder.append_null();
            continue;
        }

        let start = offsets[i].as_usize();
        let end = offsets[i + 1].as_usize();

        for j in start..end {
            buf.clear();
            if encoder.is_null(j) {
                list_builder.values().append_null();
            } else {
                encoder.encode(j, &mut buf);
                list_builder.values().append_value(
                    std::str::from_utf8(&buf)
                        .map_err(|e| ArrowError::ExternalError(Box::new(e)))?,
                );
            }
        }
        list_builder.append(true);
    }

    Ok(Arc::new(list_builder.finish()))
}

fn serialize_scalar(col: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    let field = Arc::new(Field::new("item", col.data_type().clone(), true));
    let options = EncoderOptions::default();
    let mut encoder = make_encoder(&field, col.as_ref(), &options)?;

    let mut builder = LargeStringBuilder::new();
    let mut buf: Vec<u8> = Vec::new();

    for i in 0..col.len() {
        buf.clear();
        if encoder.is_null(i) {
            builder.append_null();
        } else {
            encoder.encode(i, &mut buf);
            builder.append_value(
                std::str::from_utf8(&buf).map_err(|e| ArrowError::ExternalError(Box::new(e)))?,
            );
        }
    }

    Ok(Arc::new(builder.finish()))
}