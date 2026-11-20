//! Pareto frontiers of cost vectors.

mod truncated;

use alloc::vec::Vec;

use crate::cost::{Cost, weakly_dominates};

pub(crate) use truncated::TruncatedFront;

/// A Pareto frontier of `N`-objective cost vectors.
///
/// Holds mutually non-dominated points, at most one per distinct vector,
/// sorted lexicographically ascending. For two objectives that means the
/// first objective strictly ascending and the second strictly descending.
///
/// Only points that are lexicographically no greater than a probe can
/// weakly dominate it, so queries binary-search that prefix first:
///
/// | Operation | `N <= 2` | `N > 2` |
/// |---|---|---|
/// | [`Front::weakly_dominates`] | `O(log n)` | `O(log n + k)`, `k` the prefix length |
/// | [`Front::insert`] | `O(log n)` plus a tail shift | `O(n)` |
///
/// # Examples
///
/// ```
/// use multiobjective_search::Front;
///
/// let mut front = Front::new();
/// assert!(front.insert([1_u32, 2, 3]));
/// assert!(front.insert([3, 2, 1]));
/// assert!(!front.insert([3, 3, 3])); // dominated by both
/// assert!(front.insert([1, 2, 2])); // replaces [1, 2, 3]
/// assert_eq!(front.as_slice(), &[[1, 2, 2], [3, 2, 1]]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Front<C, const N: usize> {
    /// Non-dominated points, strictly ascending in lexicographic order.
    points: Vec<[C; N]>,
}

/// A Pareto frontier of two-objective cost vectors: [`Front`] with `N = 2`.
///
/// Dominance queries are a binary search, `O(log n)`; an insertion also
/// shifts the tail, `O(n)` worst case.
///
/// # Examples
///
/// ```
/// use multiobjective_search::BiFront;
///
/// let mut front = BiFront::new();
/// assert!(front.insert([3_u32, 1]));
/// assert!(front.insert([1, 3]));
/// assert!(!front.insert([4, 4])); // dominated by [3, 1]
/// assert!(front.insert([2, 2]));
/// assert!(front.insert([1, 1])); // dominates everything else
/// assert_eq!(front.as_slice(), &[[1, 1]]);
/// ```
pub type BiFront<C> = Front<C, 2>;

impl<C, const N: usize> Default for Front<C, N> {
    fn default() -> Self {
        Self { points: Vec::new() }
    }
}

impl<C: Cost, const N: usize> Front<C, N> {
    /// Creates an empty frontier.
    #[must_use]
    pub const fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Returns `true` if some point on the frontier is no worse than `point`
    /// in every objective, equality included.
    #[must_use]
    pub fn weakly_dominates(&self, point: &[C; N]) -> bool {
        let candidates = &self.points[..self.points.partition_point(|q| q <= point)];

        if N <= 2 {
            // Of all points lexicographically <= point, the last one has the
            // smallest second cost.
            candidates
                .last()
                .is_some_and(|q| weakly_dominates(q, point))
        } else {
            candidates.iter().any(|q| weakly_dominates(q, point))
        }
    }

    /// Inserts `point` unless the frontier already weakly dominates it.
    ///
    /// Removes every point that `point` dominates. Returns `true` if the
    /// point was inserted, `false` if it was dominated by or equal to an
    /// existing point.
    pub fn insert(&mut self, point: [C; N]) -> bool {
        if self.weakly_dominates(&point) {
            return false;
        }

        // Every point that `point` dominates is lexicographically greater.
        let start = self.points.partition_point(|q| q < &point);

        if N <= 2 {
            // shortcut: Vec splice shifts the tail in O(n); use a balanced tree if profiles show it.
            let dominated = self.points[start..].partition_point(|q| weakly_dominates(&point, q));
            self.points.splice(start..start + dominated, [point]);
        } else {
            self.points.insert(start, point);
            let mut index = 0;
            self.points.retain(|q| {
                let keep = index <= start || !weakly_dominates(&point, q);
                index += 1;
                keep
            });
        }

        true
    }

    /// Number of points on the frontier.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the frontier holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The points, strictly ascending in lexicographic order.
    #[must_use]
    pub fn as_slice(&self) -> &[[C; N]] {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`Front`] and [`BiFront`].

    use super::{BiFront, Front};

    /// Equal points are rejected and the frontier keeps one copy.
    #[test]
    fn rejects_equal_point() {
        let mut front = BiFront::new();
        assert!(front.insert([2_u8, 2]));
        assert!(!front.insert([2, 2]));
        assert_eq!(front.len(), 1);
    }

    /// A point with an equal first cost and a lower second cost replaces
    /// the old one.
    #[test]
    fn replaces_point_with_equal_first_cost() {
        let mut front = BiFront::new();
        assert!(front.insert([1_u8, 5]));
        assert!(front.insert([3, 3]));
        assert!(front.insert([3, 2]));
        assert_eq!(front.as_slice(), &[[1, 5], [3, 2]]);
    }

    /// Inserting in the middle removes only the run the point dominates.
    #[test]
    fn removes_only_dominated_run() {
        let mut front = BiFront::new();
        for p in [[0_u8, 9], [2, 7], [4, 5], [6, 3], [8, 1]] {
            assert!(front.insert(p));
        }

        assert!(front.insert([3, 3]));
        assert_eq!(front.as_slice(), &[[0, 9], [2, 7], [3, 3], [8, 1]]);
    }

    /// In three dimensions, dominated points need not be contiguous.
    #[test]
    fn removes_scattered_points_in_three_dimensions() {
        let mut front = Front::new();
        for p in [[1_u8, 1, 9], [2, 5, 5], [3, 9, 1], [4, 6, 4]] {
            assert!(front.insert(p));
        }

        assert!(front.insert([2, 5, 4]));
        assert_eq!(front.as_slice(), &[[1, 1, 9], [2, 5, 4], [3, 9, 1]]);
    }
}
