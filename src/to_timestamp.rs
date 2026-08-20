use arrow::array::ArrayRef;
use arrow::compute::kernels::cast;
use arrow::datatypes::{DataType, Field, TimeUnit};
use arrow::error::ArrowError;
use std::sync::Arc;

use crate::ColumnTransform;

/// A [`ColumnTransform`] that casts columns to a `Timestamp` type.
///
/// The target time unit and optional timezone are configurable via the
/// constructor.
pub struct ToTimestamp {
    unit: TimeUnit,
    tz: Option<Arc<str>>,
}

impl ToTimestamp {
    pub fn new(unit: TimeUnit, tz: Option<Arc<str>>) -> Self {
        Self { unit, tz }
    }
}

impl ColumnTransform for ToTimestamp {
    fn apply(&self, col: &ArrayRef) -> Result<ArrayRef, ArrowError> {
        cast::cast(col, &DataType::Timestamp(self.unit, self.tz.clone()))
    }

    fn output_field(&self, input_field: &Field) -> Field {
        Field::new(
            input_field.name(),
            DataType::Timestamp(self.unit, self.tz.clone()),
            input_field.is_nullable(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, TimestampMillisecondArray, TimestampSecondArray};

    #[test]
    fn apply_millis_no_tz() {
        let col = Arc::new(Int64Array::from(vec![1_000, 2_000, 3_000])) as ArrayRef;
        let t = ToTimestamp::new(TimeUnit::Millisecond, None);
        let result = t.apply(&col).unwrap();
        let ts = result
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap();
        assert_eq!(ts.value(0), 1_000);
        assert_eq!(ts.value(1), 2_000);
        assert_eq!(ts.value(2), 3_000);
    }

    #[test]
    fn apply_seconds_with_tz() {
        let col = Arc::new(Int64Array::from(vec![100, 200])) as ArrayRef;
        let t = ToTimestamp::new(TimeUnit::Second, Some("UTC".into()));
        let result = t.apply(&col).unwrap();
        let ts = result
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .unwrap();
        assert_eq!(ts.value(0), 100);
        assert_eq!(ts.value(1), 200);
        assert_eq!(
            ts.data_type(),
            &DataType::Timestamp(TimeUnit::Second, Some("UTC".into()))
        );
    }

    #[test]
    fn apply_preserves_nulls() {
        let col = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;
        let t = ToTimestamp::new(TimeUnit::Millisecond, None);
        let result = t.apply(&col).unwrap();
        let ts = result
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap();
        assert_eq!(ts.value(0), 1);
        assert!(ts.is_null(1));
        assert_eq!(ts.value(2), 3);
    }

    #[test]
    fn output_field_configured() {
        let input = Field::new("ts", DataType::Int64, true);
        let t = ToTimestamp::new(TimeUnit::Microsecond, Some("US/Eastern".into()));
        let output = t.output_field(&input);
        assert_eq!(output.name(), "ts");
        assert_eq!(
            output.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some("US/Eastern".into()))
        );
        assert!(output.is_nullable());
    }

    #[test]
    fn output_field_preserves_nullable() {
        let input = Field::new("ts", DataType::Int64, false);
        let t = ToTimestamp::new(TimeUnit::Second, None);
        let output = t.output_field(&input);
        assert!(!output.is_nullable());
        assert_eq!(
            output.data_type(),
            &DataType::Timestamp(TimeUnit::Second, None)
        );
    }
}
