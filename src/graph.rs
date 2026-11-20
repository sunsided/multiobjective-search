//! The search-space interface: successor generation and heuristics.

use alloc::vec::Vec;

use crate::cost::Cost;

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

/// A per-objective lower bound on the remaining cost to a goal.
///
/// Algorithms state their own requirements; BOA* needs every objective to
/// be consistent (`h(s) <= c(s, t) + h(t)` for every edge) and zero on
/// every goal state.
///
/// Implemented for closures `Fn(&S) -> Option<W>`.
pub trait Heuristic<S, W> {
    /// Estimates the remaining cost from `state` to the goal.
    ///
    /// Returns `None` when no goal is reachable from `state`; the search
    /// then prunes it.
    fn estimate(&self, state: &S) -> Option<W>;
}

impl<S, W, F: Fn(&S) -> Option<W>> Heuristic<S, W> for F {
    fn estimate(&self, state: &S) -> Option<W> {
        self(state)
    }
}

/// The zero heuristic: no estimate, the search is uninformed.
///
/// Consistent for any graph with non-negative costs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZeroHeuristic;

impl<S, C: Cost, const N: usize> Heuristic<S, [C; N]> for ZeroHeuristic {
    fn estimate(&self, _state: &S) -> Option<[C; N]> {
        Some([C::default(); N])
    }
}
