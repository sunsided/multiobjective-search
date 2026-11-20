//! Cost vectors and Pareto dominance.
//!
//! A cost vector is a plain array `[C; N]` of a scalar [`Cost`] from
//! `search-core`. Arrays already provide `Copy` and lexicographic `Ord`, and
//! `[C::ZERO; N]` is the zero vector, so no newtype is needed.

use search_core::Cost;

/// Pareto dominance between cost vectors.
pub trait Dominates {
    /// Returns `true` if `self` is no worse than `other` in every objective
    /// and strictly better in at least one.
    fn dominates(&self, other: &Self) -> bool;
}

impl<C: Cost, const N: usize> Dominates for [C; N] {
    fn dominates(&self, other: &Self) -> bool {
        weakly_dominates(self, other) && self != other
    }
}

/// Returns `true` if `a` is no worse than `b` in every objective.
///
/// Unlike [`Dominates::dominates`], equal vectors weakly dominate each other.
/// Pruning uses this form: a label equal to a known one adds no new cost
/// vector to the result.
#[must_use]
pub fn weakly_dominates<C: Cost, const N: usize>(a: &[C; N], b: &[C; N]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y)
}

/// Adds two cost vectors component-wise.
#[must_use]
pub fn add<C: Cost, const N: usize>(a: [C; N], b: [C; N]) -> [C; N] {
    core::array::from_fn(|i| a[i] + b[i])
}

#[cfg(test)]
mod tests {
    //! Unit tests for dominance and vector addition.

    use super::{Dominates, add, weakly_dominates};

    /// Strict dominance needs one strictly better component.
    #[test]
    fn dominance_is_strict() {
        assert!([1_u32, 2].dominates(&[1, 3]));
        assert!(![1_u32, 2].dominates(&[1, 2]));
        assert!(![1_u32, 4].dominates(&[2, 3]));
    }

    /// Weak dominance accepts equal vectors.
    #[test]
    fn weak_dominance_accepts_equal() {
        assert!(weakly_dominates(&[1_u32, 2], &[1, 2]));
        assert!(!weakly_dominates(&[1_u32, 4], &[2, 3]));
    }

    /// Addition is component-wise.
    #[test]
    fn add_is_componentwise() {
        assert_eq!(add([1_u8, 2, 3], [4, 5, 6]), [5, 7, 9]);
    }
}
