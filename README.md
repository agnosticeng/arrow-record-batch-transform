# arrow-record-batch-transform

Composable column transforms for Apache Arrow `RecordBatch`es.

## Overview

This crate provides two complementary building blocks for working with
[Apache Arrow] `RecordBatch`es:

- **[`ColumnTransform`] / [`transform_record_batch`]** — apply typed,
  per-column transformations (e.g. JSON-serialize every value with
  [`ToJson`]), replacing columns in place while preserving the schema and
  metadata of the rest of the batch.
- **[`Denormalize`]** — the inverse of Arrow's [`RecordBatch::normalize`]:
  rebuild a flat batch (e.g. loaded from a columnar source) back into a nested
  struct schema, with automatic casting of leaf columns.

[Apache Arrow]: https://arrow.apache.org/
[`RecordBatch::normalize`]: https://docs.rs/arrow/latest/arrow/array/struct.RecordBatch.html#method.normalize

## Table of contents

- [Column transforms](#column-transforms)
  - [`ColumnTransform` trait](#columntransform-trait)
  - [`ToJson`](#tojson)
  - [`transform_record_batch`](#transform_record_batch)
- [Denormalize](#denormalize)
  - [Example](#example)
  - [`DenormalizeOptions`](#denormalizeoptions)
  - [Supported shapes](#supported-shapes)
- [License](#license)

## Column transforms

### `ColumnTransform` trait

```rust
pub trait ColumnTransform {
    fn apply(&self, col: &ArrayRef) -> Result<ArrayRef, ArrowError>;
    fn output_field(&self, input_field: &Field) -> Field;
}
```

Implement this trait to convert an input column array into an output array and
to declare how the corresponding schema field changes.

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

Applies transforms sequentially. Each transform replaces the named column with
the transform's output while copying the remaining columns and the schema
metadata.

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

## Denormalize

The inverse of `RecordBatch::normalize`: rebuild a flat batch into a nested
struct schema. Each output field is described by a [`Field`]; its scalar leaves
are looked up in the batch by their dotted path joined with a separator
(default `_`, so `device.os.name` -> `device_os_name`). Leaf columns that are
not an exact match for the target type are converted using the Arrow cast
kernel.

### Example

```rust
use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Fields, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow_record_batch_transform::{Denormalize, DenormalizeOptions};
use std::sync::Arc;

// A flat batch: device_ts, device_os, device_boot
let schema = Arc::new(Schema::new(vec![
    Field::new("device_ts", DataType::Utf8, true),    // string timestamps
    Field::new("device_os", DataType::Utf8, true),
    Field::new("device_boot", DataType::Int32, true),
]));
let batch = RecordBatch::try_new(
    schema,
    vec![
        Arc::new(arrow::array::StringArray::from(vec!["2024-01-01T00:00:00"])),
        Arc::new(arrow::array::StringArray::from(vec!["debian"])),
        Arc::new(Int32Array::from(vec![1])),
    ],
)?;

// The target nested schema.
let fields = vec![Arc::new(Field::new(
    "device",
    DataType::Struct(Fields::from(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("os", DataType::Utf8, true),
        Field::new("boot", DataType::Int32, true),
    ])),
    true,
))];

// device_ts (Utf8) is auto-cast to Timestamp(ms); other leaves match directly.
let options = DenormalizeOptions::default();
let nested: RecordBatch = batch.denormalize_with(&fields, &options)?;
```

The same operation is available as a free function when you would rather not
bring the trait into scope:

```rust
use arrow_record_batch_transform::denormalize_record_batch;

let nested = denormalize_record_batch(&batch, &fields, &DenormalizeOptions::default())?;
```

Both entry points preserve the source batch's schema metadata.

### `DenormalizeOptions`

| Field | Default | Description |
| --- | --- | --- |
| `separator` | `"_"` | Separator joining nested field paths into flat column names |
| `missing_leaf` | `MissingLeaf::Error` | Behaviour when a leaf column is absent; `MissingLeaf::Null` fills an all-null array |
| `struct_nulls_from_children` | `false` | Mark a struct row null iff every child is null |

### Supported shapes

- Nested `Struct` at any depth
- `List` / `LargeList` / `FixedSizeList`, including `List<Struct>` from aligned
  container columns (misaligned columns produce an error)
- Dictionary / scalar leaves, with automatic casting via the Arrow cast kernel
  (e.g. `Utf8` -> `Timestamp`, string to numeric)
- Empty structs, preserving the batch row count

## License

MIT OR Apache-2.0
