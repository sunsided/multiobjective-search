//! A frontier that ignores the first objective, for searches that pop
//! labels in lexicographic order.
//!
//! When labels leave the open list in lexicographic order, every label
//! already processed is no greater in the first objective than the current
//! one. Dominance against those labels then only needs the remaining
//! objectives ("truncated" dominance, Pulido et al. 2015; Ren et al. 2022).
//! That drops one dimension: three objectives become a two-dimensional
//! frontier with `O(log n)` queries.

use crate::cost::Cost;
use crate::pareto::Front;

/// Pareto frontier over objectives `1..N` of `N`-objective vectors.
///
/// Only sound against vectors that are lexicographically no greater than
/// every later probe, which is what lexicographic best-first searches
/// guarantee for expanded labels and for solutions.
#[derive(Debug, Clone)]
pub(crate) enum TruncatedFront<C, const N: usize> {
    /// `N == 3`: tails stored as two-dimensional points `[v[1], v[2]]`.
    Two(Front<C, 2>),
    /// Any other `N`: full vectors with the first component set to zero.
    /// For `N = 2` this is the `O(log n)` two-dimensional path as well.
    Zeroed(Front<C, N>),
}

impl<C: Cost, const N: usize> TruncatedFront<C, N> {
    /// Creates an empty frontier, picking the representation for `N`.
    pub(crate) const fn new() -> Self {
        if N == 3 {
            Self::Two(Front::new())
        } else {
            Self::Zeroed(Front::new())
        }
    }

    /// Returns `true` if some stored vector is no worse than `point` in
    /// every objective but the first.
    pub(crate) fn weakly_dominates(&self, point: &[C; N]) -> bool {
        match self {
            Self::Two(front) => front.weakly_dominates(&[point[1], point[2]]),
            Self::Zeroed(front) => front.weakly_dominates(&zero_first(*point)),
        }
    }

    /// Inserts `point` unless it is weakly dominated in objectives `1..N`.
    /// Returns `true` if it was inserted.
    pub(crate) fn insert(&mut self, point: [C; N]) -> bool {
        match self {
            Self::Two(front) => front.insert([point[1], point[2]]),
            Self::Zeroed(front) => front.insert(zero_first(point)),
        }
    }
}

/// Returns `point` with its first component replaced by zero.
fn zero_first<C: Cost, const N: usize>(mut point: [C; N]) -> [C; N] {
    if let Some(first) = point.first_mut() {
        *first = C::default();
    }

    point
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`TruncatedFront`].

    use super::TruncatedFront;

    /// The first objective is ignored in every dimension.
    #[test]
    fn ignores_first_objective() {
        let mut two = TruncatedFront::<u8, 2>::new();
        assert!(two.insert([9, 3]));
        assert!(two.weakly_dominates(&[0, 3]));
        assert!(!two.weakly_dominates(&[0, 2]));

        let mut three = TruncatedFront::<u8, 3>::new();
        assert!(three.insert([9, 3, 3]));
        assert!(three.weakly_dominates(&[0, 3, 4]));
        assert!(!three.weakly_dominates(&[0, 4, 2]));

        let mut four = TruncatedFront::<u8, 4>::new();
        assert!(four.insert([9, 3, 3, 3]));
        assert!(four.weakly_dominates(&[0, 3, 4, 3]));
        assert!(!four.weakly_dominates(&[9, 3, 3, 2]));
    }
}
