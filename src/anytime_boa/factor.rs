//! Exact approximation factors: the paper's domination factor `DF` and the
//! approximation factor of an interval, as rationals of costs.

use core::cmp::Ordering;

use search_core::{Cost, WideMul};

/// An approximation factor `ε >= 0`, kept exactly as a ratio of costs.
///
/// `ε` says how well a set of solutions covers the Pareto-optimal front:
/// every Pareto-optimal cost `s` has a solution `r` with
/// `r_i <= (1 + ε) · s_i` in both objectives. The searches derive `ε` from
/// cost vectors (`(r_i - s_i) / s_i`), so it is a ratio of two costs and
/// compared by exact cross-multiplication ([`WideMul`]). It is infinite
/// when a solution must cover a zero cost with a positive one.
///
/// Equality and order are numeric: `1/2` equals `2/4`.
#[derive(Debug, Clone, Copy)]
pub enum ApproxFactor<C> {
    /// `ε = 0`: the solutions include the whole Pareto-optimal front.
    Zero,
    /// `ε = num / den`, both positive.
    Ratio {
        /// Numerator, positive.
        num: C,
        /// Denominator, positive.
        den: C,
    },
    /// `ε = ∞`: some Pareto-optimal cost with a zero component is not yet
    /// matched by a solution.
    Infinite,
}

impl<C: Cost + WideMul> ApproxFactor<C> {
    /// Returns `true` for `ε = 0`.
    #[must_use]
    pub fn is_zero(self) -> bool {
        matches!(self, Self::Zero)
    }

    /// Returns `true` if `x <= (1 + ε) · y`, exactly.
    #[must_use]
    pub fn covers(self, x: C, y: C) -> bool {
        if x <= y {
            return true;
        }

        match self {
            Self::Zero => false,
            Self::Infinite => y > C::ZERO,
            // x > y: (x - y) / y <= num / den.
            Self::Ratio { num, den } => x.abs_diff(y).wide_mul(den) <= num.wide_mul(y),
        }
    }

    /// The factor by which the cost `by` exceeds `of` in one objective,
    /// `max((by - of) / of, 0)`.
    pub(crate) fn excess(by: C, of: C) -> Self {
        if by <= of {
            Self::Zero
        } else if of == C::ZERO {
            Self::Infinite
        } else {
            Self::Ratio {
                num: by.abs_diff(of),
                den: of,
            }
        }
    }

    /// The domination factor `DF(by, of)` of the paper: the smallest `ε`
    /// with `by <= (1 + ε) · of` in both objectives.
    pub(crate) fn domination(by: [C; 2], of: [C; 2]) -> Self {
        Self::excess(by[0], of[0]).max(Self::excess(by[1], of[1]))
    }

    /// `ε / d`. A denominator that saturates the cost type gives zero, so
    /// that a schedule never stalls at the smallest representable ratio.
    pub(crate) fn divided(self, d: u32) -> Self {
        match self {
            Self::Ratio { num, den } => {
                let den = den.saturating_times(d);
                if den.saturating_add(den) == den {
                    Self::Zero
                } else {
                    Self::Ratio { num, den }
                }
            }
            other => other,
        }
    }
}

impl<C: Cost + WideMul> PartialEq for ApproxFactor<C> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<C: Cost + WideMul> Eq for ApproxFactor<C> {}

impl<C: Cost + WideMul> PartialOrd for ApproxFactor<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Cost + WideMul> Ord for ApproxFactor<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (*self, *other) {
            (Self::Zero, Self::Zero) | (Self::Infinite, Self::Infinite) => Ordering::Equal,
            (Self::Zero, _) | (_, Self::Infinite) => Ordering::Less,
            (_, Self::Zero) | (Self::Infinite, _) => Ordering::Greater,
            (Self::Ratio { num: a, den: b }, Self::Ratio { num: c, den: d }) => {
                a.wide_mul(d).cmp(&c.wide_mul(b))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`ApproxFactor`].

    use super::ApproxFactor;

    /// Ratios compare numerically, and zero and infinity bound them.
    #[test]
    fn order_is_numeric() {
        let half = ApproxFactor::Ratio { num: 1_u32, den: 2 };
        let two_quarters = ApproxFactor::Ratio { num: 2, den: 4 };
        let third = ApproxFactor::Ratio { num: 1, den: 3 };
        assert_eq!(half, two_quarters);
        assert!(third < half);
        assert!(ApproxFactor::Zero < third);
        assert!(half < ApproxFactor::Infinite);
    }

    /// `DF` matches the paper's Example 1: `DF((2, 3), (3, 2)) = 0.5`.
    #[test]
    fn domination_factor() {
        let df = ApproxFactor::domination([2_u32, 3], [3, 2]);
        assert_eq!(df, ApproxFactor::Ratio { num: 1, den: 2 });
        assert!(ApproxFactor::domination([1_u32, 4], [1, 4]).is_zero());
        assert_eq!(
            ApproxFactor::domination([1_u32, 1], [1, 0]),
            ApproxFactor::Infinite
        );
    }

    /// `covers` is `x <= (1 + ε) y`.
    #[test]
    fn covers_is_exact() {
        let half = ApproxFactor::Ratio { num: 1_u32, den: 2 };
        assert!(half.covers(15, 10));
        assert!(!half.covers(16, 10));
        assert!(!ApproxFactor::Zero.covers(11_u32, 10));
        assert!(ApproxFactor::Infinite.covers(100_u32, 1));
        assert!(!ApproxFactor::Infinite.covers(1_u32, 0));
    }
}
