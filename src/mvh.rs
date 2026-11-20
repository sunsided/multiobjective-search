//! Multi-valued heuristics: a set of cost-to-go estimates per state.

use alloc::vec::Vec;

use search_core::Heuristic;

/// A multi-valued heuristic (MVH): for every state, a set of cost vectors
/// that bound the remaining cost to a goal from below.
///
/// Wolff, Felner, Salzman, "Bridging Multi-Valued Heuristics and
/// Dimensionality Reduction in Multi-Objective Search", arXiv 2606.20644.
///
/// - **Admissible:** for every path from `state` to a goal, some estimate
///   weakly dominates its cost. In particular every goal has the zero
///   vector among its estimates.
/// - **Consistent:** every goal's estimates are `{0}`, and for every edge
///   `(s, t)` with cost `c` and every estimate `h` of `s`, some estimate `h'`
///   of `t` has `h <= c + h'` component-wise.
///
/// A single-valued [`Heuristic`] is the special case of one estimate per
/// state; wrap it in [`SingleValued`]. Implemented for closures through
/// [`mv_heuristic_fn`] and for `&mut M`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{MvHeuristic, mv_heuristic_fn};
///
/// // Two trade-offs from state 0, the goal 1 is free.
/// let mut h = mv_heuristic_fn(|s: &u8, out: &mut Vec<[u32; 2]>| match s {
///     0 => out.extend([[1, 5], [4, 2]]),
///     _ => out.push([0, 0]),
/// });
///
/// let mut out = Vec::new();
/// h.estimates(&0, &mut out);
/// assert_eq!(out, [[1, 5], [4, 2]]);
/// ```
pub trait MvHeuristic<S, V> {
    /// Appends the estimates of `state` to `out`.
    ///
    /// `out` is empty on entry. Appending nothing marks `state` as a dead
    /// end from which no goal is reachable. The estimates need not be
    /// mutually non-dominated or sorted; the searches handle both.
    fn estimates(&mut self, state: &S, out: &mut Vec<V>);
}

impl<S, V, M: MvHeuristic<S, V> + ?Sized> MvHeuristic<S, V> for &mut M {
    fn estimates(&mut self, state: &S, out: &mut Vec<V>) {
        (**self).estimates(state, out);
    }
}

/// A single-valued [`Heuristic`] as a [`MvHeuristic`] with one estimate
/// per state, none for a dead end.
///
/// Admissible or consistent exactly when the wrapped heuristic is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SingleValued<H>(pub H);

impl<S, V, H: Heuristic<S, V>> MvHeuristic<S, V> for SingleValued<H> {
    fn estimates(&mut self, state: &S, out: &mut Vec<V>) {
        out.extend(self.0.estimate(state));
    }
}

/// A [`MvHeuristic`] backed by a closure; see [`mv_heuristic_fn`].
#[derive(Debug, Clone, Copy)]
pub struct MvHeuristicFn<F>(F);

impl<S, V, F: FnMut(&S, &mut Vec<V>)> MvHeuristic<S, V> for MvHeuristicFn<F> {
    fn estimates(&mut self, state: &S, out: &mut Vec<V>) {
        (self.0)(state, out);
    }
}

/// Wraps a closure `|state, out| ...` that appends the estimates of
/// `state` to `out` as a [`MvHeuristic`].
pub const fn mv_heuristic_fn<S, V, F: FnMut(&S, &mut Vec<V>)>(f: F) -> MvHeuristicFn<F> {
    MvHeuristicFn(f)
}
