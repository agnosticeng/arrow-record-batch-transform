# arrow-record-batch-transform

Composable column transforms for Apache Arrow RecordBatches.

## Overview

This crate provides a `ColumnTransform` trait and a `transform_record_batch` function for applying column-level transformations to Arrow `RecordBatch`es. Each transform replaces a column in-place while preserving the schema and metadata of other columns.

## API

```rust
use arrow_record_batch_transform::{transform_record_batch, ToJson};

// `ToJson` serializes columns to JSON strings
let result = transform_record_batch(&batch, &[("my_column", &ToJson)])?;
```

### `ColumnTransform` trait

```rust
pub trait ColumnTransform {
    fn apply(&self, col: &ArrayRef) -> Result<ArrayRef, ArrowError>;
    fn output_field(&self, input_field: &Field) -> Field;
}
```

### `ToJson`

Serializes any Arrow column to JSON strings using the `arrow-json` encoder. Handles:
- Scalar columns → `LargeUtf8` array
- `List` columns → `List<LargeUtf8>` preserving the list structure
- `LargeList` columns → `LargeList<LargeUtf8>` preserving the list structure
- Null values are preserved

### `transform_record_batch`

```rust
pub fn transform_record_batch(
    batch: &RecordBatch,
    transforms: &[(&str, &dyn ColumnTransform)],
) -> Result<RecordBatch, ArrowError>
```

Applies transforms sequentially. Each transform replaces the named column with the transform's output while copying remaining columns and schema metadata.

## Example

```rust
use arrow::array::Int32Array;
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_record_batch_transform::{transform_record_batch, ToJson};
use std::sync::Arc;

let schema = Arc::new(Schema::new(vec![
    Field::new("a", arrow::datatypes::DataType::Int32, true),
    Field::new("b", arrow::datatypes::DataType::Utf8, true),
]));
let batch = RecordBatch::try_new(
    schema,
    vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        Arc::new(arrow::array::StringArray::from(vec!["x", "y", "z"])),
    ],
)?;

let result = transform_record_batch(&batch, &[("a", &ToJson)])?;
// Column "a" is now LargeUtf8 with values "1", "2", "3"
```

## License

MIT OR Apache-2.0