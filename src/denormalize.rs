use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, GenericListArray, OffsetSizeTrait, StructArray,
    new_null_array,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::compute::kernels::cast::{self, can_cast_types};
use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;

/// Policy for handling a leaf column that is missing from the batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingLeaf {
    /// Return an [`ArrowError`] describing the missing column.
    Error,
    /// Materialise an all-null array of the expected type and length.
    Null,
}

/// Options controlling how a [`RecordBatch`] is rebuilt into a nested schema.
///
/// The defaults mirror the usual output of [`RecordBatch::normalize`]: columns
/// are looked up by dotted paths joined with `_`, and a missing or
/// non-castable leaf produces an error.
#[derive(Debug, Clone)]
pub struct DenormalizeOptions {
    /// Separator used to reconstruct the flat name of a leaf column from its
    /// nested field path (e.g. `device.os.name` -> `device_os_name`).
    ///
    /// Defaults to `"_"`. It should match the separator used by the flattening
    /// step that produced the flat batch.
    pub separator: String,
    /// Behaviour when a leaf column cannot be found in the batch.
    ///
    /// Defaults to [`MissingLeaf::Error`].
    pub missing_leaf: MissingLeaf,
    /// When `true`, a struct value is marked null if and only if all of its
    /// child values are null (a null "repetition" layout).
    ///
    /// When `false` (the default), the built struct columns carry no null
    /// mask, so every struct row is valid regardless of its children.
    pub struct_nulls_from_children: bool,
}

impl Default for DenormalizeOptions {
    fn default() -> Self {
        Self {
            separator: "_".to_owned(),
            missing_leaf: MissingLeaf::Error,
            struct_nulls_from_children: false,
        }
    }
}

/// Rebuild a flat [`RecordBatch`] into a nested (struct) schema.
///
/// This is the inverse of [`RecordBatch::normalize`]: each field in `fields`
/// describes an output column, and every leaf column is located in the batch
/// under its dotted path joined by the configured separator (e.g.
/// `device.os.name` -> `device_os_name`).
///
/// Let `fields` be the target nested schema (typically produced by a
/// de/serialization framework from a `serde` type, or authored by hand).
/// Each field in `fields` describes one output column.
pub trait Denormalize {
    /// Rebuild this batch using the default [`DenormalizeOptions`].
    fn denormalize(&self, fields: &[FieldRef]) -> Result<RecordBatch, ArrowError> {
        self.denormalize_with(fields, &DenormalizeOptions::default())
    }

    /// Rebuild this batch using the given [`DenormalizeOptions`].
    fn denormalize_with(
        &self,
        fields: &[FieldRef],
        options: &DenormalizeOptions,
    ) -> Result<RecordBatch, ArrowError>;
}

impl Denormalize for RecordBatch {
    fn denormalize_with(
        &self,
        fields: &[FieldRef],
        options: &DenormalizeOptions,
    ) -> Result<RecordBatch, ArrowError> {
        denormalize_impl(self, fields, options)
    }
}

/// Rebuild a flat [`RecordBatch`] into a nested schema.
///
/// Equivalent to [`Denormalize::denormalize_with`] as a free function, keeping
/// it symmetric with [`crate::transform_record_batch`].
pub fn denormalize_record_batch(
    batch: &RecordBatch,
    fields: &[FieldRef],
    options: &DenormalizeOptions,
) -> Result<RecordBatch, ArrowError> {
    denormalize_impl(batch, fields, options)
}

fn denormalize_impl(
    batch: &RecordBatch,
    fields: &[FieldRef],
    options: &DenormalizeOptions,
) -> Result<RecordBatch, ArrowError> {
    let ctx = BatchContext::new(batch)?;
    let num_rows = batch.num_rows();

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(fields.len());
    for field in fields {
        let column = build_array(&ctx, field, field.name(), 0, num_rows, options)?;
        columns.push(column);
    }

    let out_fields: Fields = fields.iter().map(|field| field.as_ref().clone()).collect();
    let schema = Arc::new(Schema::new_with_metadata(
        out_fields,
        batch.schema().metadata().clone(),
    ));

    RecordBatch::try_new(schema, columns)
}

/// Precomputed column lookup for a batch.
struct BatchContext<'a> {
    columns: &'a [ArrayRef],
    names: HashMap<String, usize>,
}

impl<'a> BatchContext<'a> {
    fn new(batch: &'a RecordBatch) -> Result<Self, ArrowError> {
        let names = batch
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name().clone(), i))
            .collect();
        Ok(Self {
            columns: batch.columns(),
            names,
        })
    }

    fn column(&self, name: &str) -> Option<&'a ArrayRef> {
        self.names.get(name).map(|&i| &self.columns[i])
    }
}

/// Build the array for `field`, whose flat (un-nested) columns live under
/// `prefix`.
///
/// `depth` is the number of container levels (lists) that have already been
/// descended, and `count` is the number of values at the current level, used
/// when materialising missing leaves.
fn build_array(
    ctx: &BatchContext<'_>,
    field: &Field,
    prefix: &str,
    depth: usize,
    count: usize,
    options: &DenormalizeOptions,
) -> Result<ArrayRef, ArrowError> {
    match field.data_type() {
        DataType::Struct(children) => {
            build_struct(ctx, field, prefix, depth, count, children, options)
        }
        DataType::List(_) => build_list::<i32>(ctx, field, prefix, depth, count, options),
        DataType::LargeList(_) => build_list::<i64>(ctx, field, prefix, depth, count, options),
        DataType::FixedSizeList(_, _) => {
            build_fixed_size_list(ctx, field, prefix, depth, count, options)
        }
        _ => build_leaf(ctx, field, prefix, depth, count, options),
    }
}

fn build_struct(
    ctx: &BatchContext<'_>,
    _field: &Field,
    prefix: &str,
    depth: usize,
    count: usize,
    children: &arrow::datatypes::Fields,
    options: &DenormalizeOptions,
) -> Result<ArrayRef, ArrowError> {
    let arrs: Vec<ArrayRef> = children
        .iter()
        .map(|child| {
            let child_prefix = join(prefix, child.name(), &options.separator);
            build_array(ctx, child, &child_prefix, depth, count, options)
        })
        .collect::<Result<_, _>>()?;

    if arrs.is_empty() {
        // Degenerate struct: no children to derive length from, so carry the
        // current level's length in the null buffer.
        let nulls = NullBuffer::new_valid(count);
        return Ok(Arc::new(StructArray::new_empty_fields(count, Some(nulls))));
    }

    let nulls = if options.struct_nulls_from_children {
        Some(struct_nulls(count, &arrs))
    } else {
        None
    };

    StructArray::try_new(children.clone(), arrs, nulls).map(|a| Arc::new(a) as ArrayRef)
}

/// Null buffer where a struct row is null iff every child is null.
fn struct_nulls(count: usize, arrs: &[ArrayRef]) -> NullBuffer {
    let mut any_valid = vec![false; count];
    for arr in arrs {
        for (i, slot) in any_valid.iter_mut().enumerate() {
            if !*slot && !arr.is_null(i) {
                *slot = true;
            }
        }
    }
    NullBuffer::new(BooleanBuffer::from(any_valid))
}

fn build_list<O: OffsetSizeTrait>(
    ctx: &BatchContext<'_>,
    field: &Field,
    prefix: &str,
    depth: usize,
    _count: usize,
    options: &DenormalizeOptions,
) -> Result<ArrayRef, ArrowError> {
    let inner = match field.data_type() {
        DataType::List(f) | DataType::LargeList(f) => f.clone(),
        _ => unreachable!(),
    };

    let leaves = leaf_names(&inner, prefix, &options.separator);
    let first_leaf = leaves.first().ok_or_else(|| {
        ArrowError::SchemaError(format!("empty container at '{prefix}' has no leaf columns"))
    })?;

    let first_array = level_array(ctx, first_leaf, depth)?.ok_or_else(|| {
        ArrowError::SchemaError(format!("column '{first_leaf}' not found in batch"))
    })?;
    let first_list = first_array
        .as_any()
        .downcast_ref::<GenericListArray<O>>()
        .ok_or_else(|| {
            ArrowError::SchemaError(format!(
                "expected a list column at '{first_leaf}' (depth {depth})"
            ))
        })?;
    let offsets = first_list.offsets().clone();
    let nulls = first_list.nulls().cloned();

    for name in &leaves[1..] {
        let column = level_array(ctx, name, depth)?.ok_or_else(|| {
            ArrowError::SchemaError(format!("column '{name}' not found in batch"))
        })?;
        let list = column
            .as_any()
            .downcast_ref::<GenericListArray<O>>()
            .ok_or_else(|| {
                ArrowError::SchemaError(format!(
                    "expected a list column at '{name}' (depth {depth})"
                ))
            })?;
        if list.offsets().inner() != offsets.inner() {
            return Err(ArrowError::SchemaError(format!(
                "unaligned list columns under '{prefix}': offsets of '{name}' differ from '{first_leaf}'"
            )));
        }
    }

    let child_count = offsets
        .inner()
        .last()
        .map(|o| o.as_usize())
        .unwrap_or_default();
    let values = build_array(ctx, &inner, prefix, depth + 1, child_count, options)?;

    GenericListArray::<O>::try_new(inner, offsets, values, nulls).map(|a| Arc::new(a) as ArrayRef)
}

fn build_fixed_size_list(
    ctx: &BatchContext<'_>,
    field: &Field,
    prefix: &str,
    depth: usize,
    count: usize,
    options: &DenormalizeOptions,
) -> Result<ArrayRef, ArrowError> {
    let (inner, size) = match field.data_type() {
        DataType::FixedSizeList(f, n) => (f.clone(), *n),
        _ => unreachable!(),
    };

    let leaves = leaf_names(&inner, prefix, &options.separator);

    let (child_count, nulls) = if let Some(name) = leaves.first() {
        let column = level_array(ctx, name, depth)?.ok_or_else(|| {
            ArrowError::SchemaError(format!("column '{name}' not found in batch"))
        })?;
        let array = column
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                ArrowError::SchemaError(format!(
                    "expected a fixed-size-list column at '{name}' (depth {depth})"
                ))
            })?;
        let count = array.len().saturating_mul(size as usize);
        (count, array.nulls().cloned())
    } else {
        (count.saturating_mul(size as usize), None)
    };

    let values = build_array(ctx, &inner, prefix, depth + 1, child_count, options)?;

    FixedSizeListArray::try_new(inner, size, values, nulls).map(|a| Arc::new(a) as ArrayRef)
}

fn build_leaf(
    ctx: &BatchContext<'_>,
    field: &Field,
    prefix: &str,
    depth: usize,
    count: usize,
    options: &DenormalizeOptions,
) -> Result<ArrayRef, ArrowError> {
    let column = match ctx.column(prefix) {
        Some(column) => column.clone(),
        None => {
            return match options.missing_leaf {
                MissingLeaf::Error => Err(ArrowError::SchemaError(format!(
                    "column '{prefix}' not found in batch"
                ))),
                MissingLeaf::Null => Ok(new_null_array(field.data_type(), count)),
            };
        }
    };

    let mut array = column;
    for _ in 0..depth {
        array = unwind_one_level(&array)?;
    }

    let target = field.data_type();
    if array.data_type() == target {
        return Ok(array);
    }
    if can_cast_types(array.data_type(), target) {
        return cast::cast(&array, target);
    }

    Err(ArrowError::SchemaError(format!(
        "cannot cast column '{prefix}' from {:?} to {:?}",
        array.data_type(),
        target
    )))
}

/// Descend `depth` container levels into the column named `name`, returning the
/// array at that level, or `None` if the column does not exist.
fn level_array(
    ctx: &BatchContext<'_>,
    name: &str,
    depth: usize,
) -> Result<Option<ArrayRef>, ArrowError> {
    let mut array = match ctx.column(name) {
        Some(column) => column.clone(),
        None => return Ok(None),
    };
    for _ in 0..depth {
        array = unwind_one_level(&array)?;
    }
    Ok(Some(array))
}

/// Unwrap one container level, yielding the element array.
fn unwind_one_level(array: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    match array.data_type() {
        DataType::List(_) => array
            .as_any()
            .downcast_ref::<GenericListArray<i32>>()
            .map(|a| a.values().clone())
            .ok_or_else(|| ArrowError::SchemaError("failed to downcast List column".to_owned())),
        DataType::LargeList(_) => array
            .as_any()
            .downcast_ref::<GenericListArray<i64>>()
            .map(|a| a.values().clone())
            .ok_or_else(|| {
                ArrowError::SchemaError("failed to downcast LargeList column".to_owned())
            }),
        DataType::FixedSizeList(_, _) => array
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .map(|a| a.values().clone())
            .ok_or_else(|| {
                ArrowError::SchemaError("failed to downcast FixedSizeList column".to_owned())
            }),
        other => Err(ArrowError::SchemaError(format!(
            "expected a list column, found {other}",
        ))),
    }
}

/// Collect the fully-qualified (flat) names of every scalar leaf under `field`.
///
/// Containers do not contribute a name segment, matching how flattening keeps
/// the leaf path unchanged across list levels.
fn collect_leaf_names(field: &Field, prefix: &str, separator: &str, out: &mut Vec<String>) {
    match field.data_type() {
        DataType::Struct(children) => {
            for child in children.iter() {
                let child_prefix = join(prefix, child.name(), separator);
                collect_leaf_names(child, &child_prefix, separator, out);
            }
        }
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => {
            collect_leaf_names(f, prefix, separator, out);
        }
        _ => out.push(prefix.to_owned()),
    }
}

/// Entry point for [`collect_leaf_names`] that allocates the output vector.
fn leaf_names(field: &Field, prefix: &str, separator: &str) -> Vec<String> {
    let mut names = Vec::new();
    collect_leaf_names(field, prefix, separator, &mut names);
    names
}

fn join(parent: &str, child: &str, separator: &str) -> String {
    if parent.is_empty() {
        child.to_owned()
    } else {
        format!("{parent}{separator}{child}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Array, FixedSizeListArray, Int32Array, LargeListArray, ListArray, StringArray, StructArray,
        TimestampMillisecondArray,
    };
    use arrow::datatypes::{DataType, Int32Type, TimeUnit};

    fn denormalize(
        batch: &RecordBatch,
        fields: &[FieldRef],
        options: &DenormalizeOptions,
    ) -> Result<RecordBatch, ArrowError> {
        denormalize_impl(batch, fields, options)
    }

    fn opts(separator: &str) -> DenormalizeOptions {
        DenormalizeOptions {
            separator: separator.to_owned(),
            ..Default::default()
        }
    }

    fn str_list(rows: Vec<Option<Vec<&str>>>) -> ArrayRef {
        use arrow::array::builder::{ListBuilder, StringBuilder};
        let mut builder = ListBuilder::new(StringBuilder::new());
        for row in rows {
            match row {
                Some(values) => {
                    for value in values {
                        builder.values().append_value(value);
                    }
                    builder.append(true);
                }
                None => builder.append(false),
            }
        }
        Arc::new(builder.finish())
    }

    #[test]
    fn round_trip_with_normalize() {
        let os_field = Arc::new(Field::new(
            "os",
            DataType::Struct(Fields::from(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("kernel", DataType::Utf8, true),
            ])),
            true,
        ));
        let id_field = Arc::new(Field::new("id", DataType::Int32, true));
        let device_field = Arc::new(Field::new(
            "device",
            DataType::Struct(Fields::from(vec![os_field.clone(), id_field.clone()])),
            true,
        ));
        let tags_field = Arc::new(Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ));

        let os = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("name", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec!["debian", "fedora"])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("kernel", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec!["6.1", "6.5"])) as ArrayRef,
            ),
        ])) as ArrayRef;
        let device = Arc::new(StructArray::from(vec![
            (os_field.clone(), os),
            (
                id_field.clone(),
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            ),
        ])) as ArrayRef;
        let tags = str_list(vec![Some(vec!["a", "b"]), Some(vec!["c"])]);

        let schema = Arc::new(Schema::new(vec![device_field.clone(), tags_field.clone()]));
        let batch = RecordBatch::try_new(schema, vec![device, tags]).expect("valid batch");

        let flattened = batch.normalize(".", None).expect("normalize");
        let restored =
            denormalize(&flattened, &[device_field, tags_field], &opts(".")).expect("denormalize");

        assert_eq!(restored, batch);
    }

    #[test]
    fn nested_three_levels() {
        let inner = FieldRef::from(Field::new(
            "inner",
            DataType::Struct(Fields::from(vec![Field::new(
                "leaf",
                DataType::Int32,
                true,
            )])),
            true,
        ));
        let outer = FieldRef::from(Field::new(
            "outer",
            DataType::Struct(Fields::from(vec![Field::new("mid", DataType::Int32, true)])),
            true,
        ));
        let _ = outer;

        // Flat schema: outer_mid, outer_inner_leaf with `_` separator.
        let schema = Arc::new(Schema::new(vec![
            Field::new("outer_mid", DataType::Int32, true),
            Field::new("outer_inner_leaf", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                Arc::new(Int32Array::from(vec![3, 4])) as ArrayRef,
            ],
        )
        .expect("valid batch");

        let target = vec![FieldRef::from(Field::new(
            "outer",
            DataType::Struct(Fields::from(vec![
                Field::new("mid", DataType::Int32, true),
                inner.as_ref().clone(),
            ])),
            true,
        ))];

        let result = denormalize(&batch, &target, &opts("_")).expect("denormalize");
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let mid = col.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let inner = col
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let leaf = inner
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        assert_eq!(mid.values(), &[1, 2]);
        assert_eq!(leaf.values(), &[3, 4]);
    }

    #[test]
    fn casts_utf8_to_timestamp() {
        let ts = Arc::new(Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ));

        let schema = Arc::new(Schema::new(vec![Field::new("ts", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["2024-01-01T00:00:00"])) as ArrayRef],
        )
        .expect("valid batch");

        let result = denormalize(&batch, &[ts], &Default::default()).expect("denormalize");
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap();
        assert_eq!(col.value(0), 1704067200000);
    }

    #[test]
    fn non_nullable_int_field() {
        let field = FieldRef::from(Field::new("x", DataType::Int32, false));
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef],
        )
        .expect("valid batch");

        let result = denormalize(&batch, &[field], &Default::default()).expect("denormalize");
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(col.values(), &[1, 2, 3]);
    }

    #[test]
    fn missing_leaf_error() {
        let field = FieldRef::from(Field::new(
            "xs",
            DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int32, true)])),
            true,
        ));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "unrelated",
            DataType::Int32,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef],
        )
        .expect("valid batch");

        let err = denormalize(&batch, &[field], &Default::default()).unwrap_err();
        assert!(err.to_string().contains("xs_a"), "{err}");
    }

    #[test]
    fn missing_leaf_null() {
        let field = FieldRef::from(Field::new("foo", DataType::Int32, true));
        let schema = Arc::new(Schema::new(vec![Field::new("other", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef],
        )
        .expect("valid batch");

        let options = DenormalizeOptions {
            missing_leaf: MissingLeaf::Null,
            ..Default::default()
        };
        let result = denormalize(&batch, &[field], &options).expect("denormalize");
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(col.len(), 2);
        assert!(col.is_null(0) && col.is_null(1));
    }

    #[test]
    fn misaligned_list_of_struct_errors() {
        let field = FieldRef::from(Field::new(
            "n",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(Fields::from(vec![
                    Field::new("a", DataType::Int32, true),
                    Field::new("b", DataType::Int32, true),
                ])),
                true,
            ))),
            true,
        ));

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "n_a",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
            Field::new(
                "n_b",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(1), Some(2), Some(3)]),
                ])) as ArrayRef,
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(4), Some(5)]),
                ])) as ArrayRef,
            ],
        )
        .expect("valid batch");

        let err = denormalize(&batch, &[field], &Default::default()).unwrap_err();
        assert!(err.to_string().contains("unaligned"), "{err}");
    }

    #[test]
    fn list_of_struct() {
        let field = FieldRef::from(Field::new(
            "n",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(Fields::from(vec![
                    Field::new("a", DataType::Int32, true),
                    Field::new("b", DataType::Utf8, true),
                ])),
                true,
            ))),
            true,
        ));

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "n_a",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
            Field::new(
                "n_b",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(1), Some(2)]),
                    Some(vec![Some(3)]),
                ])) as ArrayRef,
                str_list(vec![Some(vec!["a", "b"]), Some(vec!["c"])]),
            ],
        )
        .expect("valid batch");

        let result = denormalize(&batch, &[field], &Default::default()).expect("denormalize");
        let list = result
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(list.len(), 2);
        let inner = list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        let inner_a = inner
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let inner_b = inner
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(inner_a.values(), &[1, 2, 3]);
        assert_eq!(
            vec![inner_b.value(0), inner_b.value(1), inner_b.value(2)],
            vec!["a", "b", "c"]
        );

        // Row 0 has two structs, row 1 has one.
        let offsets = list.offsets();
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 2);
        assert_eq!(offsets[2], 3);
    }

    #[test]
    fn large_list_passthrough() {
        let field = FieldRef::from(Field::new(
            "lg",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        ));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "lg",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]));
        let arr = Arc::new(LargeListArray::from_iter_primitive::<Int32Type, _, _>(
            vec![Some(vec![Some(1), Some(2)]), None],
        )) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![arr.clone()]).expect("valid batch");

        let result = denormalize(&batch, &[field], &Default::default()).expect("denormalize");
        assert!(result.column(0).eq(&arr));
    }

    #[test]
    fn fixed_size_list_passthrough() {
        let item = Arc::new(Field::new("item", DataType::Int32, true));
        let field = FieldRef::from(Field::new(
            "pts",
            DataType::FixedSizeList(item.clone(), 2),
            true,
        ));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "pts",
            DataType::FixedSizeList(item, 2),
            true,
        )]));
        let arr = Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            2,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            None,
        )) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![arr.clone()]).expect("valid batch");

        let result = denormalize(&batch, &[field], &Default::default()).expect("denormalize");
        assert!(result.column(0).eq(&arr));
    }

    #[test]
    fn empty_struct_preserves_row_count() {
        let field = FieldRef::from(Field::new("e", DataType::Struct(Fields::empty()), true));
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef],
        )
        .expect("valid batch");

        let result = denormalize(&batch, &[field], &Default::default()).expect("denormalize");
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(col.len(), 3);
        assert!(!col.is_null(0) && !col.is_null(2));
    }

    #[test]
    fn struct_nulls_from_children() {
        let field = FieldRef::from(Field::new(
            "s",
            DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Int32, true),
            ])),
            true,
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("s_a", DataType::Int32, true),
            Field::new("s_b", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef,
                Arc::new(Int32Array::from(vec![None, None, Some(3)])) as ArrayRef,
            ],
        )
        .expect("valid batch");

        // Default: struct is never null.
        let plain = denormalize(&batch, std::slice::from_ref(&field), &Default::default()).unwrap();
        let plain_s = plain
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert!(plain_s.null_count() == 0);

        // With the flag: null iff every child is null.
        let options = DenormalizeOptions {
            struct_nulls_from_children: true,
            ..Default::default()
        };
        let nested = denormalize(&batch, &[field], &options).unwrap();
        let s = nested
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert!(!s.is_null(0));
        assert!(s.is_null(1));
        assert!(!s.is_null(2));
    }

    #[test]
    fn preserves_schema_metadata() {
        let field = FieldRef::from(Field::new("x", DataType::Int32, true));
        let schema = Arc::new(Schema::new_with_metadata(
            vec![Field::new("x", DataType::Int32, true)],
            [("k".to_owned(), "v".to_owned())].into(),
        ));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef],
        )
        .expect("valid batch");

        let result = denormalize(&batch, &[field], &Default::default()).unwrap();
        assert_eq!(result.schema().metadata().get("k").unwrap(), "v");
    }

    #[test]
    fn public_api_trait_and_free_fn() {
        let field = FieldRef::from(Field::new("x", DataType::Int32, true));
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
        )
        .expect("valid batch");

        // Trait: default options.
        let via_trait = batch.denormalize(std::slice::from_ref(&field)).unwrap();
        assert_eq!(
            via_trait
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[1, 2]
        );

        // Trait: explicit options.
        let via_trait_with = batch
            .denormalize_with(std::slice::from_ref(&field), &Default::default())
            .unwrap();
        assert_eq!(via_trait_with.num_columns(), 1);

        // Free function.
        let via_fn = denormalize_record_batch(&batch, &[field], &Default::default()).unwrap();
        assert_eq!(
            via_fn
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[1, 2]
        );
    }

    #[test]
    fn list_scalar_element_casts() {
        let field = FieldRef::from(Field::new(
            "times",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ))),
            true,
        ));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "times",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![str_list(vec![Some(vec!["2024-01-01T00:00:00"])])],
        )
        .expect("valid batch");

        let result = denormalize(&batch, &[field], &Default::default()).expect("denormalize");
        let list = result
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let values = list
            .values()
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap();
        assert_eq!(values.value(0), 1704067200000);
    }

    #[test]
    fn list_of_list_passthrough() {
        use arrow::array::builder::{Int32Builder, ListBuilder};
        let field = FieldRef::from(Field::new(
            "outer",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ))),
            true,
        ));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "outer",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ))),
            true,
        )]));

        let mut outer = ListBuilder::new(ListBuilder::new(Int32Builder::new()));
        outer.values().values().append_value(1);
        outer.values().append(true);
        outer.values().values().append_value(2);
        outer.values().values().append_value(3);
        outer.values().append(true);
        outer.append(true);
        let arr = Arc::new(outer.finish()) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![arr.clone()]).expect("valid batch");

        let result = denormalize(&batch, &[field], &Default::default()).expect("denormalize");
        assert!(result.column(0).eq(&arr));
    }

    #[test]
    fn large_list_of_struct() {
        use arrow::array::builder::{LargeListBuilder, StringBuilder};
        let field = FieldRef::from(Field::new(
            "n",
            DataType::LargeList(Arc::new(Field::new(
                "item",
                DataType::Struct(Fields::from(vec![
                    Field::new("a", DataType::Int32, true),
                    Field::new("b", DataType::Utf8, true),
                ])),
                true,
            ))),
            true,
        ));
        let a_schema = Field::new(
            "n_a",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        );
        let b_schema = Field::new(
            "n_b",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        );
        let schema = Arc::new(Schema::new(vec![a_schema, b_schema]));

        let a = Arc::new(LargeListArray::from_iter_primitive::<Int32Type, _, _>(
            vec![Some(vec![Some(1), Some(2)]), Some(vec![Some(3)])],
        )) as ArrayRef;

        let mut b = LargeListBuilder::new(StringBuilder::new());
        b.values().append_value("x");
        b.values().append_value("y");
        b.append(true);
        b.values().append_value("z");
        b.append(true);
        let b = Arc::new(b.finish()) as ArrayRef;

        let batch = RecordBatch::try_new(schema, vec![a, b]).expect("valid batch");

        let result = denormalize(&batch, &[field], &Default::default()).expect("denormalize");
        let list = result
            .column(0)
            .as_any()
            .downcast_ref::<LargeListArray>()
            .unwrap();
        let inner = list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let inner_a = inner
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(inner_a.values(), &[1, 2, 3]);
    }

    #[test]
    fn fixed_size_list_of_struct() {
        let item_a = Arc::new(Field::new("item", DataType::Int32, true));
        let item_b = Arc::new(Field::new("item", DataType::Utf8, true));
        let field = FieldRef::from(Field::new(
            "n",
            DataType::FixedSizeList(
                Arc::new(Field::new(
                    "item",
                    DataType::Struct(Fields::from(vec![
                        Field::new("a", DataType::Int32, true),
                        Field::new("b", DataType::Utf8, true),
                    ])),
                    true,
                )),
                2,
            ),
            true,
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("n_a", DataType::FixedSizeList(item_a.clone(), 2), true),
            Field::new("n_b", DataType::FixedSizeList(item_b.clone(), 2), true),
        ]));

        let a = Arc::new(FixedSizeListArray::new(
            item_a,
            2,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            None,
        )) as ArrayRef;
        let b = Arc::new(FixedSizeListArray::new(
            item_b,
            2,
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            None,
        )) as ArrayRef;

        let batch = RecordBatch::try_new(schema, vec![a, b]).expect("valid batch");
        let result = denormalize(&batch, &[field], &Default::default()).expect("denormalize");

        let list = result
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let inner = list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let inner_a = inner
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(inner_a.values(), &[1, 2, 3, 4]);
    }

    #[test]
    fn dictionary_leaf_casts_to_values() {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int8Type;
        let field = FieldRef::from(Field::new("d", DataType::Utf8, true));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "d",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            true,
        )]));
        let arr = Arc::new(DictionaryArray::<Int8Type>::from_iter(vec![
            Some("x"),
            Some("y"),
            Some("x"),
        ])) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![arr]).expect("valid batch");

        let result = denormalize(&batch, &[field], &Default::default()).expect("denormalize");
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            vec![col.value(0), col.value(1), col.value(2)],
            vec!["x", "y", "x"]
        );
    }

    #[test]
    fn missing_leaf_null_inside_struct() {
        let field = FieldRef::from(Field::new(
            "s",
            DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Int32, true),
            ])),
            true,
        ));
        // Only `s_a` is present; `s_b` is missing.
        let schema = Arc::new(Schema::new(vec![Field::new("s_a", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef],
        )
        .expect("valid batch");

        let options = DenormalizeOptions {
            missing_leaf: MissingLeaf::Null,
            ..Default::default()
        };
        let result =
            denormalize(&batch, std::slice::from_ref(&field), &options).expect("denormalize");
        let s = result
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let b = s.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(b.len(), 3);
        assert!(b.is_null(0) && b.is_null(1) && b.is_null(2));
    }
}
