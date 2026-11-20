//! NAMOA*dr with multi-valued heuristics: the eager NAMOA*dr-mvh and the
//! lazy L-NAMOA*dr-mvh.
//!
//! Wolff, Felner, Salzman: "Bridging Multi-Valued Heuristics and
//! Dimensionality Reduction in Multi-Objective Search", arXiv 2606.20644.
//!
//! Dimensionality reduction drops the first objective from dominance
//! checks against closed labels. That needs the labels of a state to be
//! expanded in non-decreasing order of `g_1` (the paper's Property 1),
//! which a consistent single-valued heuristic guarantees. With several
//! estimates per state, two labels of one state can be ranked by
//! different estimates, and the order breaks even for a consistent MVH
//! (Lemma 1; see `tests/it/mvh.rs` for the paper's counterexample).
//!
//! - [`namoa_star_dr_mvh`] binds every label to one estimate, gives each
//!   `(state, estimate)` pair its own closed set, and follows only edges
//!   where the estimates stay consistent (Algorithm 1, red annotations).
//!   Needs a consistent MVH.
//! - [`namoa_star_dr_mvh_lazy`] keeps one label per path, binds it lazily
//!   to the lexicographically smallest estimate no solution rules out, and
//!   falls back to a full dominance check when a state's labels arrive out
//!   of `g_1` order (Algorithm 2). Needs only an admissible MVH.

mod lazy;

use alloc::vec::Vec;
use core::cell::RefCell;
use core::mem;

use search_core::{Cost, Heuristic};

use crate::cost::{add, weakly_dominates};
use crate::frontier_search;
use crate::graph::Graph;
use crate::mvh::MvHeuristic;
use crate::output::{Output, Solution};
use crate::pareto::TruncatedFront;

pub use lazy::namoa_star_dr_mvh_lazy;

/// NAMOA*dr-mvh: NAMOA*dr with a consistent multi-valued heuristic.
///
/// Every label is bound to one estimate of its state. A successor is
/// generated once for every estimate `h'` of the next state with
/// `h <= c + h'` (path-consistent selection, the paper's Definition 1), and
/// dominance against closed labels is checked per `(state, estimate)` pair
/// on objectives `1..N` (Theorem 1). This is exactly NAMOA*dr on the graph
/// of `(state, estimate)` pairs, whose single-valued heuristic, the bound
/// estimate, is consistent by construction; the implementation runs
/// [`namoa_star_dr`](crate::namoa_star_dr)'s engine on that graph.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is a consistent [`MvHeuristic`] (see there; the exact
///   Pareto sets of the remaining cost are). With an inconsistent one, a
///   path whose estimates admit no path-consistent chain is never
///   generated, so solutions can be lost and dominated ones returned.
///
/// Solutions are the Pareto-optimal cost vectors, one path each, sorted
/// lexicographically. The counters count `(label, estimate)` pairs, which
/// is the paper's state-space blow-up (its limitation L1): prefer
/// [`namoa_star_dr_mvh_lazy`] in practice. The heuristic is queried on
/// every generation; cache inside it if estimates are expensive.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, mv_heuristic_fn, namoa_star_dr_mvh};
///
/// /// Two routes from 0 to the goal 2 through 1.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         match state {
///             0 => out.push((1, [1, 1])),
///             1 => out.extend([(2, [1, 4]), (2, [3, 1])]),
///             _ => {}
///         }
///     }
/// }
///
/// // The exact Pareto sets of the remaining cost: consistent.
/// let h = mv_heuristic_fn(|s: &u8, out: &mut Vec<[u32; 2]>| match s {
///     0 => out.extend([[2, 5], [4, 2]]),
///     1 => out.extend([[1, 4], [3, 1]]),
///     _ => out.push([0, 0]),
/// });
///
/// let out = namoa_star_dr_mvh(&Routes, h, 0, |s| *s == 2);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[2, 5], [4, 2]]);
/// ```
pub fn namoa_star_dr_mvh<G, M, C, P, const N: usize>(
    graph: &G,
    mut heuristic: M,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]>,
    M: MvHeuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut start_estimates = Vec::new();
    heuristic.estimates(&start, &mut start_estimates);
    let Some(floor) = component_min(&start_estimates) else {
        return Output {
            solutions: Vec::new(),
            stats: crate::output::Stats::default(),
        };
    };

    let product = Bound {
        graph,
        heuristic: RefCell::new(heuristic),
        start: start.clone(),
        start_estimates,
        edges: RefCell::new(Vec::new()),
        estimates: RefCell::new(Vec::new()),
    };
    let goal = |(state, h): &(G::State, Option<[C; N]>)| h.is_some() && is_goal(state);
    let out = frontier_search::run::<_, _, C, _, TruncatedFront<C, N>, N>(
        &product,
        BoundEstimate(floor),
        (start, None),
        goal,
        true,
    );

    unbind(out)
}

/// The graph of `(state, bound estimate)` pairs that NAMOA*dr-mvh searches.
///
/// A virtual root `(start, None)` has a zero-cost edge to `(start, Some(h))`
/// for every start estimate `h`, so that the engine's single start label
/// fans out into one label per estimate.
struct Bound<'a, G: Graph, M, C, const N: usize> {
    /// The underlying graph.
    graph: &'a G,
    /// The multi-valued heuristic, behind a `RefCell` because successor
    /// generation only gets `&self`.
    heuristic: RefCell<M>,
    /// The start state, for the virtual root's edges.
    start: G::State,
    /// The start state's estimates; never empty.
    start_estimates: Vec<[C; N]>,
    /// Reused successor buffer, empty between calls.
    edges: RefCell<Vec<(G::State, [C; N])>>,
    /// Reused estimate buffer, empty between calls.
    estimates: RefCell<Vec<[C; N]>>,
}

impl<G, M, C, const N: usize> Graph for Bound<'_, G, M, C, N>
where
    G: Graph<Weight = [C; N]>,
    M: MvHeuristic<G::State, [C; N]>,
    C: Cost,
{
    type State = (G::State, Option<[C; N]>);
    type Weight = [C; N];

    fn successors(&self, state: &Self::State, out: &mut Vec<(Self::State, [C; N])>) {
        let (from, bound) = state;
        let Some(h) = bound else {
            let root = |&h| ((self.start.clone(), Some(h)), [C::ZERO; N]);
            out.extend(self.start_estimates.iter().map(root));
            return;
        };

        let mut edges = mem::take(&mut *self.edges.borrow_mut());
        let mut estimates = mem::take(&mut *self.estimates.borrow_mut());
        self.graph.successors(from, &mut edges);

        for (to, weight) in edges.drain(..) {
            self.heuristic.borrow_mut().estimates(&to, &mut estimates);
            for next in estimates.drain(..) {
                // Path-consistent selection: h <= c + h'.
                if weakly_dominates(h, &add(weight, next)) {
                    out.push(((to.clone(), Some(next)), weight));
                }
            }
        }

        *self.edges.borrow_mut() = edges;
        *self.estimates.borrow_mut() = estimates;
    }
}

/// The single-valued heuristic of the pair graph: the bound estimate, or
/// for the virtual root the component-wise minimum of the start estimates,
/// which keeps the root's edges consistent.
struct BoundEstimate<C, const N: usize>([C; N]);

impl<S, C: Cost, const N: usize> Heuristic<(S, Option<[C; N]>), [C; N]> for BoundEstimate<C, N> {
    fn estimate(&mut self, state: &(S, Option<[C; N]>)) -> Option<[C; N]> {
        Some(state.1.unwrap_or(self.0))
    }
}

/// The component-wise minimum of `vectors`, or `None` if it is empty.
fn component_min<C: Cost, const N: usize>(vectors: &[[C; N]]) -> Option<[C; N]> {
    vectors
        .iter()
        .copied()
        .reduce(|a, b| core::array::from_fn(|i| a[i].min(b[i])))
}

/// Maps a pair-graph result back to states: drops the virtual root from
/// every path and from the counters.
fn unbind<S, C, const N: usize>(out: Output<(S, Option<[C; N]>), [C; N]>) -> Output<S, [C; N]> {
    let solutions = out
        .solutions
        .into_iter()
        .map(|solution| Solution {
            cost: solution.cost,
            path: solution.path.into_iter().skip(1).map(|(s, _)| s).collect(),
        })
        .collect();

    let mut stats = out.stats;
    stats.expanded = stats.expanded.saturating_sub(1);
    stats.generated = stats.generated.saturating_sub(1);

    Output { solutions, stats }
}
