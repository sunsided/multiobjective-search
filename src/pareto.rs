//! Pareto frontiers of cost vectors.

use alloc::vec::Vec;

use crate::cost::Cost;

/// A Pareto frontier of two-objective cost vectors.
///
/// Holds mutually non-dominated points, at most one per distinct vector.
/// Points are sorted by the first objective ascending, which on a frontier
/// means the second objective strictly descending. Dominance queries are a
/// binary search, `O(log n)`; an insertion also shifts the tail, `O(n)`
/// worst case.
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BiFront<C> {
    /// Non-dominated points, first objective strictly ascending, second
    /// strictly descending.
    points: Vec<[C; 2]>,
}

impl<C: Cost> BiFront<C> {
    /// Creates an empty frontier.
    #[must_use]
    pub const fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Returns `true` if some point on the frontier is no worse than `point`
    /// in both objectives, equality included.
    #[must_use]
    pub fn weakly_dominates(&self, point: &[C; 2]) -> bool {
        let at_most_first = self.points.partition_point(|q| q[0] <= point[0]);

        // Of all points with a first cost <= point[0], the last one has the
        // smallest second cost.
        at_most_first > 0 && self.points[at_most_first - 1][1] <= point[1]
    }

    /// Inserts `point` unless the frontier already weakly dominates it.
    ///
    /// Removes every point that `point` dominates. Returns `true` if the
    /// point was inserted, `false` if it was dominated by or equal to an
    /// existing point.
    pub fn insert(&mut self, point: [C; 2]) -> bool {
        if self.weakly_dominates(&point) {
            return false;
        }

        let start = self.points.partition_point(|q| q[0] < point[0]);
        let dominated = self.points[start..].partition_point(|q| q[1] >= point[1]);
        self.points.splice(start..start + dominated, [point]);

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

    /// The points, first objective strictly ascending.
    #[must_use]
    pub fn as_slice(&self) -> &[[C; 2]] {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`BiFront`].

    use super::BiFront;

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
}
