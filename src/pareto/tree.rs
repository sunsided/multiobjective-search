//! A Pareto frontier in a balanced search tree, for four or more objectives.
//!
//! [`Front`](crate::Front) keeps its points in a sorted `Vec`. For three or
//! more objectives its dominance query scans the whole lexicographic prefix
//! below the probe, and an insertion shifts the tail. [`TreeFront`] keeps the
//! same points in a treap (a randomized balanced binary search tree, keyed
//! lexicographically) whose nodes also store the component-wise minimum and
//! maximum of their subtree. A query descends only into subtrees whose
//! minimum is no worse than the probe in every objective, so whole regions
//! of the frontier are skipped at once. This is the balanced-tree frontier
//! of EMOA* (Ren et al., `SoCS` 2022), with bounding boxes for the pruning.
//!
//! ```mermaid
//! graph TD
//!     q[weakly_dominates p] --> box{subtree min <= p?}
//!     box -- no --> skip[skip subtree]
//!     box -- yes --> here{node <= p?}
//!     here -- yes --> hit[true]
//!     here -- no --> left[search left]
//!     left --> right{node < p lexicographically?}
//!     right -- yes --> rs[search right]
//!     right -- no --> skip
//! ```
//!
//! Structure operations take expected `O(log n)`. A dominance query is
//! output-sensitive: it visits the nodes whose bounding box admits the probe,
//! which is far fewer than `n` on typical fronts but `O(n)` in the worst
//! case. No dynamic structure with cheap updates answers `d >= 3`
//! dominance queries in polylogarithmic worst-case time; range trees would,
//! at `O(log^(d-1) n)` per query and much costlier updates.

use alloc::vec::Vec;

use search_core::Cost;

use crate::cost::weakly_dominates;

/// A Pareto frontier of `N`-objective cost vectors in a balanced tree.
///
/// Holds the same points as [`Front`](crate::Front) and answers the same
/// queries, but with sublinear typical cost for many objectives; see the
/// module documentation. Prefer [`Front`](crate::Front) for one to three
/// objectives, where its sorted `Vec` is faster.
///
/// # Examples
///
/// ```
/// use multiobjective_search::TreeFront;
///
/// let mut front = TreeFront::new();
/// assert!(front.insert([1_u32, 2, 3, 4]));
/// assert!(front.insert([4, 3, 2, 1]));
/// assert!(!front.insert([4, 4, 4, 4])); // dominated by both
/// assert!(front.insert([1, 2, 3, 3])); // replaces [1, 2, 3, 4]
/// assert_eq!(front.to_vec(), [[1, 2, 3, 3], [4, 3, 2, 1]]);
/// assert!(front.weakly_dominates(&[5, 5, 5, 5]));
/// ```
#[derive(Debug, Clone)]
pub struct TreeFront<C, const N: usize> {
    /// Node arena; freed slots are listed in [`TreeFront::free`].
    nodes: Vec<TreeNode<C, N>>,
    /// Arena indices of removed nodes, reused by later insertions.
    free: Vec<usize>,
    /// Root of the treap; `None` when empty.
    root: Option<usize>,
    /// Number of points stored.
    len: usize,
    /// State of the xorshift generator that draws node priorities. Fixed
    /// seed, so runs are deterministic.
    rng: u64,
}

/// One point of a [`TreeFront`] with its subtree summary.
#[derive(Debug, Clone)]
struct TreeNode<C, const N: usize> {
    /// The stored cost vector; the lexicographic search key.
    point: [C; N],
    /// Component-wise minimum over this node's subtree, itself included.
    min: [C; N],
    /// Component-wise maximum over this node's subtree, itself included.
    max: [C; N],
    /// Heap priority: a parent's is at least its children's.
    priority: u64,
    /// Subtree of lexicographically smaller points.
    left: Option<usize>,
    /// Subtree of lexicographically greater points.
    right: Option<usize>,
}

impl<C, const N: usize> Default for TreeFront<C, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C, const N: usize> TreeFront<C, N> {
    /// Creates an empty frontier.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            root: None,
            len: 0,
            rng: 0x9e37_79b9_7f4a_7c15,
        }
    }

    /// Number of points on the frontier.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the frontier holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<C: Cost, const N: usize> TreeFront<C, N> {
    /// Returns `true` if some point on the frontier is no worse than `point`
    /// in every objective, equality included.
    #[must_use]
    pub fn weakly_dominates(&self, point: &[C; N]) -> bool {
        self.any_below(self.root, point)
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

        self.root = self.remove_dominated(self.root, &point);

        let node = self.allocate(point);
        let (less, greater) = self.split(self.root, |q| q < &point);
        let less = self.merge(less, Some(node));
        self.root = self.merge(less, greater);
        self.len += 1;

        true
    }

    /// The points, strictly ascending in lexicographic order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<[C; N]> {
        let mut out = Vec::with_capacity(self.len);
        self.collect_in_order(self.root, &mut out);

        out
    }

    /// Appends the points of the subtree at `node` to `out` in order.
    fn collect_in_order(&self, node: Option<usize>, out: &mut Vec<[C; N]>) {
        if let Some(index) = node {
            let n = &self.nodes[index];
            self.collect_in_order(n.left, out);
            out.push(n.point);
            self.collect_in_order(n.right, out);
        }
    }

    /// Returns `true` if the subtree at `node` holds a point that weakly
    /// dominates `point`.
    fn any_below(&self, node: Option<usize>, point: &[C; N]) -> bool {
        let Some(index) = node else {
            return false;
        };

        let n = &self.nodes[index];
        if !weakly_dominates(&n.min, point) {
            return false;
        }

        if weakly_dominates(&n.point, point) || self.any_below(n.left, point) {
            return true;
        }

        // Points greater than `point` lexicographically cannot be <= it.
        n.point < *point && self.any_below(n.right, point)
    }

    /// Removes from the subtree at `node` every point that `point` weakly
    /// dominates and returns the new subtree root.
    fn remove_dominated(&mut self, node: Option<usize>, point: &[C; N]) -> Option<usize> {
        let index = node?;
        if !weakly_dominates(point, &self.nodes[index].max) {
            return node;
        }

        // Points smaller than `point` lexicographically cannot be >= it.
        if self.nodes[index].point > *point {
            let left = self.remove_dominated(self.nodes[index].left, point);
            self.nodes[index].left = left;
        }

        let right = self.remove_dominated(self.nodes[index].right, point);
        self.nodes[index].right = right;

        if weakly_dominates(point, &self.nodes[index].point) {
            self.free.push(index);
            self.len -= 1;
            return self.merge(self.nodes[index].left, self.nodes[index].right);
        }

        self.update(index);
        node
    }

    /// Stores a new leaf node for `point` and returns its arena index.
    fn allocate(&mut self, point: [C; N]) -> usize {
        let node = TreeNode {
            point,
            min: point,
            max: point,
            priority: self.next_priority(),
            left: None,
            right: None,
        };

        if let Some(index) = self.free.pop() {
            self.nodes[index] = node;
            index
        } else {
            self.nodes.push(node);
            self.nodes.len() - 1
        }
    }

    /// Draws the next node priority (xorshift64).
    fn next_priority(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// Splits the subtree at `node` into the points for which `goes_left`
    /// holds and the rest. `goes_left` must be monotone: true on a prefix of
    /// the lexicographic order.
    fn split(
        &mut self,
        node: Option<usize>,
        goes_left: impl Fn(&[C; N]) -> bool + Copy,
    ) -> (Option<usize>, Option<usize>) {
        let Some(index) = node else {
            return (None, None);
        };

        if goes_left(&self.nodes[index].point) {
            let (middle, right) = self.split(self.nodes[index].right, goes_left);
            self.nodes[index].right = middle;
            self.update(index);
            (Some(index), right)
        } else {
            let (left, middle) = self.split(self.nodes[index].left, goes_left);
            self.nodes[index].left = middle;
            self.update(index);
            (left, Some(index))
        }
    }

    /// Joins two subtrees where every point of `left` is lexicographically
    /// smaller than every point of `right`.
    fn merge(&mut self, left: Option<usize>, right: Option<usize>) -> Option<usize> {
        let (l, r) = match (left, right) {
            (None, tree) | (tree, None) => return tree,
            (Some(l), Some(r)) => (l, r),
        };

        if self.nodes[l].priority >= self.nodes[r].priority {
            let merged = self.merge(self.nodes[l].right, Some(r));
            self.nodes[l].right = merged;
            self.update(l);
            Some(l)
        } else {
            let merged = self.merge(Some(l), self.nodes[r].left);
            self.nodes[r].left = merged;
            self.update(r);
            Some(r)
        }
    }

    /// Recomputes the subtree bounds of `index` from its children.
    fn update(&mut self, index: usize) {
        let n = &self.nodes[index];
        let (mut min, mut max) = (n.point, n.point);

        for child in [n.left, n.right].into_iter().flatten() {
            let c = &self.nodes[child];
            for i in 0..N {
                min[i] = min[i].min(c.min[i]);
                max[i] = max[i].max(c.max[i]);
            }
        }

        let n = &mut self.nodes[index];
        n.min = min;
        n.max = max;
    }
}

impl<C: Cost, const N: usize> PartialEq for TreeFront<C, N> {
    /// Two frontiers are equal when they hold the same points.
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.to_vec() == other.to_vec()
    }
}

impl<C: Cost, const N: usize> Eq for TreeFront<C, N> {}

#[cfg(test)]
mod tests {
    //! Unit tests for [`TreeFront`]; the property tests in `tests/it/front.rs`
    //! cross-check it against [`Front`](crate::Front).

    use super::TreeFront;

    /// Equal points are rejected and the frontier keeps one copy.
    #[test]
    fn rejects_equal_point() {
        let mut front = TreeFront::new();
        assert!(front.insert([2_u8, 2, 2, 2]));
        assert!(!front.insert([2, 2, 2, 2]));
        assert_eq!(front.len(), 1);
    }

    /// A point that dominates scattered points removes exactly those.
    #[test]
    fn removes_scattered_dominated_points() {
        let mut front = TreeFront::new();
        for p in [[1_u8, 1, 9, 9], [2, 5, 5, 5], [3, 9, 1, 1], [4, 6, 4, 6]] {
            assert!(front.insert(p));
        }

        assert!(front.insert([2, 5, 4, 5]));
        assert_eq!(front.to_vec(), [[1, 1, 9, 9], [2, 5, 4, 5], [3, 9, 1, 1]]);
        assert_eq!(front.len(), 3);
    }

    /// Freed arena slots are reused.
    #[test]
    fn reuses_freed_slots() {
        let mut front = TreeFront::new();
        for i in 0..8_u8 {
            assert!(front.insert([i, 8 - i, 1, 1]));
        }

        assert!(front.insert([0, 0, 0, 0]));
        assert_eq!(front.to_vec(), [[0, 0, 0, 0]]);
        assert_eq!(front.nodes.len(), 8);
    }
}
