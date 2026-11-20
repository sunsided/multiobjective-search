//! The search-space interface: successor generation and the zero heuristic
//! for cost vectors.

use alloc::vec::Vec;

use search_core::{Cost, Heuristic};

/// An implicit graph with vector-valued edge weights.
///
/// Successors are generated on demand, so the graph never needs to be
/// materialized.
pub trait Graph {
    /// A search state. `Ord` keys the per-state bookkeeping (a `BTreeMap`,
    /// since the crate is `no_std`).
    type State: Clone + Ord;

    /// The cost vector of one edge, for example `[u32; 2]`.
    type Weight;

    /// Appends every outgoing edge of `state` to `out` as
    /// `(successor, weight)`.
    ///
    /// `out` is empty on entry; the caller clears and reuses it between
    /// calls. Parallel edges and self-loops are allowed.
    fn successors(&self, state: &Self::State, out: &mut Vec<(Self::State, Self::Weight)>);
}

/// A [`Graph`] that can also enumerate incoming edges, for bidirectional
/// search.
///
/// The predecessors must mirror the successors exactly: `(p, w)` is
/// reported by `predecessors(s)` once for every `(s, w)` that
/// `successors(p)` reports, parallel edges included.
pub trait ReverseGraph: Graph {
    /// Appends every incoming edge of `state` to `out` as
    /// `(predecessor, weight)`, where `weight` is the cost of the edge from
    /// `predecessor` to `state`.
    ///
    /// `out` is empty on entry; the caller clears and reuses it between
    /// calls.
    fn predecessors(&self, state: &Self::State, out: &mut Vec<(Self::State, Self::Weight)>);
}

/// The zero heuristic for cost vectors: no estimate, the search is
/// uninformed.
///
/// Consistent for any graph with non-negative costs. `search_core::Zero`
/// covers scalar costs only, so cost vectors `[C; N]` get this local one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZeroHeuristic;

impl<S, C: Cost, const N: usize> Heuristic<S, [C; N]> for ZeroHeuristic {
    fn estimate(&mut self, _state: &S) -> Option<[C; N]> {
        Some([C::ZERO; N])
    }
}
