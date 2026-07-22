mod column_transform;
mod to_json;

pub use column_transform::ColumnTransform;
pub use to_json::ToJson;

use arrow::array::ArrayRef;
use arrow::datatypes::{Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

/// Apply a sequence of column transforms to a `RecordBatch`.
///
/// Transforms are applied sequentially, in order. Each transform replaces the
/// named column in-place while preserving the schema, data, and metadata of
/// all other columns.
///
/// # Errors
///
/// Returns an error if a named column does not exist in the batch, or if any
/// transform fails.
pub fn transform_record_batch(
    batch: &RecordBatch,
    transforms: &[(&str, &dyn ColumnTransform)],
) -> Result<RecordBatch, ArrowError> {
    transforms.iter().try_fold(batch.clone(), |b, (col_name, processor)| {
        apply_transform(&b, col_name, *processor)
    })
}

fn apply_transform(
    batch: &RecordBatch,
    col_name: &str,
    processor: &dyn ColumnTransform,
) -> Result<RecordBatch, ArrowError> {
    let col_idx = batch.schema().index_of(col_name).map_err(|_| {
        ArrowError::InvalidArgumentError(format!("column '{}' not found in batch", col_name))
    })?;

    let new_col = processor.apply(batch.column(col_idx))?;

    let new_fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if i == col_idx {
                processor.output_field(f)
            } else {
                f.as_ref().clone()
            }
        })
        .collect();

    let new_schema = Arc::new(Schema::new_with_metadata(
        new_fields,
        batch.schema().metadata().clone(),
    ));

    let new_columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .enumerate()
        .map(|(i, col)| {
            if i == col_idx {
                new_col.clone()
            } else {
                col.clone()
            }
        })
        .collect();

    RecordBatch::try_new(new_schema, new_columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int32Array, LargeStringArray, ListArray, StringArray};
    use arrow::datatypes::{DataType, Int32Type};

    #[test]
    fn to_json_int32() {
        let col = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let result = ToJson.apply(&col).unwrap();
        let strings = result.as_any().downcast_ref::<LargeStringArray>().unwrap();
        assert_eq!(strings.value(0), "1");
        assert_eq!(strings.value(1), "2");
        assert_eq!(strings.value(2), "3");
    }

    #[test]
    fn to_json_int32_with_nulls() {
        let col = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;
        let result = ToJson.apply(&col).unwrap();
        let strings = result.as_any().downcast_ref::<LargeStringArray>().unwrap();
        assert_eq!(strings.value(0), "1");
        assert!(strings.is_null(1));
        assert_eq!(strings.value(2), "3");
    }

    #[test]
    fn to_json_utf8() {
        let col = Arc::new(StringArray::from(vec!["hello", "world"])) as ArrayRef;
        let result = ToJson.apply(&col).unwrap();
        let strings = result.as_any().downcast_ref::<LargeStringArray>().unwrap();
        assert_eq!(strings.value(0), "\"hello\"");
        assert_eq!(strings.value(1), "\"world\"");
    }

    #[test]
    fn to_json_list() {
        let col = Arc::new(
            ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                Some(vec![Some(1), Some(2), Some(3)]),
                Some(vec![Some(4), None, Some(6)]),
                None,
            ]),
        ) as ArrayRef;
        let result = ToJson.apply(&col).unwrap();
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        let inner = list.values();
        let strings = inner.as_any().downcast_ref::<LargeStringArray>().unwrap();
        assert_eq!(strings.value(0), "1");
        assert_eq!(strings.value(1), "2");
        assert_eq!(strings.value(2), "3");
        assert_eq!(strings.value(3), "4");
        assert!(strings.is_null(4));
        assert_eq!(strings.value(5), "6");
        assert!(list.is_null(2));
    }

    #[test]
    fn output_field_scalar() {
        let input = Field::new("x", DataType::Int32, true);
        let output = ToJson.output_field(&input);
        assert_eq!(output.name(), "x");
        assert_eq!(output.data_type(), &DataType::LargeUtf8);
        assert!(output.is_nullable());
    }

    #[test]
    fn output_field_list() {
        let input = Field::new("xs", DataType::List(Arc::new(Field::new("item", DataType::Int32, true))), false);
        let output = ToJson.output_field(&input);
        assert_eq!(output.name(), "xs");
        assert_eq!(
            output.data_type(),
            &DataType::List(Arc::new(Field::new("item", DataType::LargeUtf8, true)))
        );
        assert!(!output.is_nullable());
    }

    #[test]
    fn transform_single_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef,
                Arc::new(StringArray::from(vec!["x", "y", "z"])) as ArrayRef,
            ],
        )
        .unwrap();

        let result = transform_record_batch(&batch, &[("a", &ToJson)]).unwrap();

        let col_a = result.column(0);
        let strings = col_a.as_any().downcast_ref::<LargeStringArray>().unwrap();
        assert_eq!(strings.value(0), "1");
        assert!(strings.is_null(1));
        assert_eq!(strings.value(2), "3");
        assert_eq!(result.schema().field(0).data_type(), &DataType::LargeUtf8);
        assert_eq!(result.schema().field(1).data_type(), &DataType::Utf8);
    }

    #[test]
    fn transform_multiple_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                Arc::new(Int32Array::from(vec![3, 4])) as ArrayRef,
            ],
        )
        .unwrap();

        let result = transform_record_batch(&batch, &[("a", &ToJson), ("b", &ToJson)]).unwrap();

        let a = result.column(0).as_any().downcast_ref::<LargeStringArray>().unwrap();
        let b = result.column(1).as_any().downcast_ref::<LargeStringArray>().unwrap();
        assert_eq!(a.value(0), "1");
        assert_eq!(a.value(1), "2");
        assert_eq!(b.value(0), "3");
        assert_eq!(b.value(1), "4");
    }

    #[test]
    fn transform_missing_column() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef]).unwrap();

        let result = transform_record_batch(&batch, &[("nonexistent", &ToJson)]);
        assert!(result.is_err());
    }

    #[test]
    fn transform_preserves_metadata() {
        let schema = Arc::new(Schema::new_with_metadata(
            vec![Field::new("a", DataType::Int32, true)],
            [("key".to_owned(), "val".to_owned())].into(),
        ));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef]).unwrap();

        let result = transform_record_batch(&batch, &[("a", &ToJson)]).unwrap();
        assert_eq!(result.schema().metadata().get("key").unwrap(), "val");
    }
}