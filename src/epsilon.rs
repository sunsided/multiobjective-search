//! Rational approximation factors `1 + ε` for the approximate searches.

use search_core::Cost;

/// A rational approximation tolerance `ε = num / den >= 0`, used as the
/// factor `1 + ε`.
///
/// [`Cost`] has no division or fractional scaling, only the saturating
/// integer product [`Cost::saturating_times`]. A rational `ε` lets the
/// searches test `x <= (1 + ε) · y` exactly as `den · x <= (den + num) · y`.
///
/// # Overflow
///
/// The products are saturating. When `(den + num) · y` saturates, the test
/// falls back to `x <= y`, which is the exact (`ε = 0`) answer. The searches
/// then prune or merge less but keep their guarantees. Keep `den + num`
/// small relative to the headroom of the cost type to get the full benefit.
///
/// # Examples
///
/// ```
/// use multiobjective_search::Epsilon;
///
/// let ten_percent = Epsilon::new(1, 10);
/// assert!(ten_percent.within(110_u32, 100));
/// assert!(!ten_percent.within(111_u32, 100));
/// assert!(Epsilon::ZERO.within(100_u32, 100));
/// assert!(!Epsilon::ZERO.within(101_u32, 100));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Epsilon {
    /// Numerator of `ε`.
    num: u32,
    /// Denominator of `ε`. Never zero; `num + den` fits in `u32`.
    den: u32,
}

impl Default for Epsilon {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Epsilon {
    /// No tolerance: `ε = 0`, which turns the approximate searches into
    /// exact ones.
    pub const ZERO: Self = Self { num: 0, den: 1 };

    /// Creates `ε = num / den`.
    ///
    /// # Panics
    ///
    /// Panics if `den` is zero or `num + den` overflows `u32`.
    #[must_use]
    pub const fn new(num: u32, den: u32) -> Self {
        assert!(den > 0, "Epsilon denominator must be positive");
        assert!(
            num.checked_add(den).is_some(),
            "Epsilon num + den must fit in u32"
        );

        Self { num, den }
    }

    /// The numerator of `ε`.
    #[must_use]
    pub const fn num(self) -> u32 {
        self.num
    }

    /// The denominator of `ε`. Never zero.
    #[must_use]
    pub const fn den(self) -> u32 {
        self.den
    }

    /// Returns `true` if `x <= (1 + ε) · y`.
    ///
    /// Exact unless `(den + num) · y` saturates the cost type; then it
    /// answers `x <= y`, which implies the exact answer.
    #[must_use]
    pub fn within<C: Cost>(self, x: C, y: C) -> bool {
        let scaled = y.saturating_times(self.den + self.num);
        if is_saturated(scaled) {
            return x <= y;
        }

        x.saturating_times(self.den) <= scaled
    }
}

/// Returns `true` if `value` is the largest value of its cost type, as a
/// saturating product reports an overflow.
fn is_saturated<C: Cost>(value: C) -> bool {
    value != C::ZERO && value.saturating_add(value) == value
}

#[cfg(test)]
mod tests {
    //! Unit and property tests for [`Epsilon`].

    use super::Epsilon;
    use proptest::prelude::*;

    /// A saturating product falls back to the exact comparison.
    #[test]
    fn saturation_falls_back_to_exact() {
        let half = Epsilon::new(1, 2);
        assert!(!half.within(u8::MAX, 200));
        assert!(half.within(15_u8, 10));
        assert!(!half.within(16_u8, 10));
        // 3 * 100 saturates u8, so the answer is the exact `150 <= 100`.
        assert!(!half.within(150_u8, 100));
        assert!(half.within(100_u8, 100));
        assert!(half.within(u8::MAX, u8::MAX));
    }

    /// A zero denominator is rejected.
    #[test]
    #[should_panic(expected = "denominator")]
    fn zero_denominator_panics() {
        let _ = Epsilon::new(1, 0);
    }

    proptest! {
        /// Without saturation, `within` matches exact rational arithmetic.
        #[test]
        fn matches_rational(x in 0..1_000_u32, y in 0..1_000_u32, num in 0..50_u32, den in 1..50_u32) {
            let exact = u64::from(den) * u64::from(x) <= (u64::from(den) + u64::from(num)) * u64::from(y);
            prop_assert_eq!(Epsilon::new(num, den).within(x, y), exact);
        }

        /// With saturation, a `true` answer is still correct.
        #[test]
        fn never_wrongly_true(x: u16, y: u16, num in 0..1_000_u32, den in 1..1_000_u32) {
            let exact = u64::from(den) * u64::from(x) <= (u64::from(den) + u64::from(num)) * u64::from(y);
            if Epsilon::new(num, den).within(x, y) {
                prop_assert!(exact);
            }
        }
    }
}
