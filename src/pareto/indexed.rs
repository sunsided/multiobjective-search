//! An immutable, indexed set of cost vectors for fast "is this weakly
//! dominated?" queries: the base of SIP-MOSP's frontier snapshots.

use alloc::vec::Vec;

use search_core::Cost;

/// How an [`IndexedFront`] summarizes its vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrontIndex {
    /// No summary: queries scan the lexicographic prefix.
    None,
    /// Component-wise minima of fixed-size blocks of this many vectors
    /// (SIP-MOSP-BM). A block whose minimum exceeds the query in some
    /// objective is skipped whole.
    #[default]
    Blocks,
    /// Component-wise minima in a segment tree over the vectors
    /// (SIP-MOSP-ST), which skips ranges at every granularity.
    Tree,
}

/// Default vectors per block of [`FrontIndex::Blocks`].
pub const DEFAULT_BLOCK: usize = 16;

/// An immutable set of `N`-objective cost vectors, sorted
/// lexicographically, with a range-minimum index (Xu et al., "Asynchronous
/// Parallel Search for Exact Multi-Objective Shortest Paths with Versioned
/// Frontier Snapshots and Indexed Dominance Pruning", arXiv 2609.11944,
/// Algorithms 4 and 5).
///
/// A query `q` first binary-searches the prefix of vectors that are
/// lexicographically no greater than `q`, since only those can weakly
/// dominate it. The index then discards every block or tree range whose
/// component-wise minimum exceeds `q` in some objective; the vectors that
/// survive are compared exactly, so the answer is the plain dominance
/// test. The index pays off when the prefix is long and the vectors spread
/// across objectives; its worst case stays linear.
///
/// The vectors need not be mutually non-dominated.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{FrontIndex, IndexedFront};
///
/// // The base of the paper's Figure 2.
/// let base = vec![
///     [4_u32, 9, 9], [5, 8, 7], [6, 4, 9], [6, 7, 8],
///     [7, 3, 7], [7, 5, 5], [8, 1, 1], [9, 0, 0],
/// ];
/// for index in [FrontIndex::None, FrontIndex::Blocks, FrontIndex::Tree] {
///     let front = IndexedFront::new(base.clone(), index);
///     assert!(front.weakly_dominates(&[7, 6, 6])); // by [7, 5, 5]
///     assert!(!front.weakly_dominates(&[6, 6, 6]));
/// }
/// ```
#[derive(Debug, Clone)]
pub struct IndexedFront<C, const N: usize> {
    /// The vectors, sorted lexicographically ascending.
    points: Vec<[C; N]>,
    /// The summary chosen at construction.
    index: Summary<C, N>,
    /// Vectors per block of [`FrontIndex::Blocks`]; at least 1.
    block: usize,
}

/// The range minima of an [`IndexedFront`].
#[derive(Debug, Clone)]
enum Summary<C, const N: usize> {
    /// No index.
    None,
    /// `mins[k]` is the component-wise minimum of block `k`.
    Blocks(Vec<[C; N]>),
    /// A segment tree in heap layout over `leaves` leaf slots (a power of
    /// two); node 1 is the root, node `i` has children `2i` and `2i + 1`,
    /// and leaf `j` is node `leaves + j`. Unused leaves are absent.
    Tree {
        /// Number of leaf slots, a power of two at least `points.len()`.
        leaves: usize,
        /// Range minima per node; `None` for an empty range.
        mins: Vec<Option<[C; N]>>,
    },
}

impl<C: Cost, const N: usize> IndexedFront<C, N> {
    /// Sorts `points` and builds the chosen index over them, with blocks of
    /// 16 vectors for [`FrontIndex::Blocks`].
    #[must_use]
    pub fn new(points: Vec<[C; N]>, index: FrontIndex) -> Self {
        Self::with_block_size(points, index, DEFAULT_BLOCK)
    }

    /// [`IndexedFront::new`] with `block` vectors per block of
    /// [`FrontIndex::Blocks`] (ignored by the other indexes).
    ///
    /// # Panics
    ///
    /// Panics if `block` is zero.
    #[must_use]
    pub fn with_block_size(mut points: Vec<[C; N]>, index: FrontIndex, block: usize) -> Self {
        assert!(block > 0, "the block size must be positive");
        points.sort_unstable();
        let index = match index {
            FrontIndex::None => Summary::None,
            FrontIndex::Blocks => Summary::Blocks(points.chunks(block).map(minimum).collect()),
            FrontIndex::Tree => tree(&points),
        };

        Self {
            points,
            index,
            block,
        }
    }

    /// The vectors, sorted lexicographically ascending.
    #[must_use]
    pub fn as_slice(&self) -> &[[C; N]] {
        &self.points
    }

    /// The number of vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns `true` if some vector is no worse than `q` in every
    /// objective.
    #[must_use]
    pub fn weakly_dominates(&self, q: &[C; N]) -> bool {
        let prefix = self.points.partition_point(|p| p <= q);
        let candidates = &self.points[..prefix];

        match &self.index {
            Summary::None => candidates.iter().any(|p| below(p, q)),
            Summary::Blocks(mins) => candidates
                .chunks(self.block)
                .zip(mins)
                .any(|(block, min)| below_tail(min, q) && block.iter().any(|p| below(p, q))),
            Summary::Tree { leaves, mins } => self.tree_dominates(*leaves, mins, prefix, q),
        }
    }

    /// The segment-tree query of Algorithm 5: descend from the root into
    /// ranges inside the prefix whose minima do not exceed `q`.
    fn tree_dominates(
        &self,
        leaves: usize,
        mins: &[Option<[C; N]>],
        prefix: usize,
        q: &[C; N],
    ) -> bool {
        // (node, first leaf, number of leaves) of the ranges still to visit.
        let mut stack = alloc::vec![(1, 0, leaves)];

        while let Some((node, first, width)) = stack.pop() {
            if first >= prefix {
                continue;
            }

            let Some(min) = &mins[node] else { continue };
            if !below_tail(min, q) {
                continue;
            }

            if width == 1 {
                if below(&self.points[first], q) {
                    return true;
                }

                continue;
            }

            let half = width / 2;
            stack.push((2 * node + 1, first + half, half));
            stack.push((2 * node, first, half));
        }

        false
    }
}

/// Builds the segment tree of [`Summary::Tree`].
fn tree<C: Cost, const N: usize>(points: &[[C; N]]) -> Summary<C, N> {
    let leaves = points.len().max(1).next_power_of_two();
    let mut mins = alloc::vec![None; 2 * leaves];
    for (j, p) in points.iter().enumerate() {
        mins[leaves + j] = Some(*p);
    }

    for node in (1..leaves).rev() {
        mins[node] = match (mins[2 * node], mins[2 * node + 1]) {
            (Some(a), Some(b)) => Some(core::array::from_fn(|i| a[i].min(b[i]))),
            (a, b) => a.or(b),
        };
    }

    Summary::Tree { leaves, mins }
}

/// The component-wise minimum of a non-empty block.
fn minimum<C: Cost, const N: usize>(block: &[[C; N]]) -> [C; N] {
    block
        .iter()
        .copied()
        .reduce(|a, b| core::array::from_fn(|i| a[i].min(b[i])))
        .expect("chunks are never empty")
}

/// Returns `true` if `p <= q` in every objective.
fn below<C: Cost, const N: usize>(p: &[C; N], q: &[C; N]) -> bool {
    p.iter().zip(q).all(|(a, b)| a <= b)
}

/// Returns `true` if `p <= q` in objectives `1..N`; the prefix already
/// guarantees the first.
fn below_tail<C: Cost, const N: usize>(p: &[C; N], q: &[C; N]) -> bool {
    p[1..].iter().zip(&q[1..]).all(|(a, b)| a <= b)
}

#[cfg(test)]
mod tests {
    //! Property tests: every index answers like the plain scan.

    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::{FrontIndex, IndexedFront};

    proptest! {
        /// All three indexes agree with exhaustive dominance.
        #[test]
        fn indexes_match_scan(
            points in prop::collection::vec(prop::array::uniform3(0..20_u32), 0..80),
            queries in prop::collection::vec(prop::array::uniform3(0..20_u32), 1..20),
        ) {
            let fronts: Vec<_> = [FrontIndex::None, FrontIndex::Blocks, FrontIndex::Tree]
                .into_iter()
                .map(|index| IndexedFront::new(points.clone(), index))
                .collect();

            for q in &queries {
                let expected = points.iter().any(|p| (0..3).all(|i| p[i] <= q[i]));
                for front in &fronts {
                    prop_assert_eq!(front.weakly_dominates(q), expected);
                }
            }
        }
    }
}
