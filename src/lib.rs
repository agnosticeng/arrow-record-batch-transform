mod column_transform;
mod to_json;

pub use column_transform::ColumnTransform;
pub use to_json::ToJson;

use arrow::array::ArrayRef;
use arrow::datatypes::{Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

/// Apply a sequence of column transforms to a `RecordBatch`.
///
/// All transforms are applied in a single pass over the columns. Transformed
/// columns are replaced in-place; all other columns and metadata are preserved.
///
/// # Errors
///
/// Returns an error if a named column does not exist in the batch, or if any
/// transform fails.
pub fn transform_record_batch(
    batch: &RecordBatch,
    transforms: &[(&str, &dyn ColumnTransform)],
) -> Result<RecordBatch, ArrowError> {
    for (name, _) in transforms {
        batch.schema().index_of(name).map_err(|_| {
            ArrowError::InvalidArgumentError(format!("column '{}' not found in batch", name))
        })?;
    }

    let map: HashMap<String, &dyn ColumnTransform> =
        transforms.iter().map(|&(n, t)| (n.to_string(), t)).collect();

    let mut new_fields: Vec<Field> = Vec::with_capacity(batch.num_columns());
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());

    for (i, field) in batch.schema().fields().iter().enumerate() {
        let col = batch.column(i);
        if let Some(t) = map.get(field.name()) {
            new_fields.push(t.output_field(field));
            new_columns.push(t.apply(col)?);
        } else {
            new_fields.push(field.as_ref().clone());
            new_columns.push(col.clone());
        }
    }

    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            new_fields,
            batch.schema().metadata().clone(),
        )),
        new_columns,
    )
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