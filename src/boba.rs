//! BOBA*: bi-objective bidirectional A*.
//!
//! Ahmadi, Tack, Harabor, Kilby: "Bi-Objective Search with Bi-Directional
//! A*", ESA 2021 (`LIPIcs` 204).
//!
//! BOBA* runs two BOA* searches with opposite directions and opposite
//! objective orders. The forward search, from the start in `(f1, f2)`
//! order, finds solutions from the cheapest-first end of the front; the
//! backward search, from the goal over reversed edges in `(f2, f1)` order,
//! finds them from the cheapest-second end. Neither joins partial paths:
//! each search runs all the way to the opposite end, with exact
//! single-objective distances as heuristics (or the caller's own, through
//! [`boba_star_with`]). Each search's last solution
//! bounds the other's primary objective, so they split the front between
//! them and stop where they meet.
//!
//! ```mermaid
//! graph TD
//!     dists[Cost-bounded searches per objective:<br/>to the goal and from the start] --> fwd[Forward BOA*, order f1 f2]
//!     dists --> bwd[Backward BOA*, order f2 f1]
//!     fwd -- "last c2 bounds backward f2" --> bwd
//!     bwd -- "last c1 bounds forward f1" --> fwd
//!     fwd --> merge[Forward solutions, then backward ones reversed]
//!     bwd --> merge
//! ```
//!
//! With its own heuristics, BOBA* also uses the paper's two enhancements
//! (Algorithm 2): the *early solution update* joins a popped label with the
//! known optimal completion of its state, records the joined path as a
//! solution if its second cost is the smallest so far, and does not expand
//! the label at all when that completion is optimal in both objectives;
//! *heuristic tuning* raises the other direction's secondary heuristic at a
//! state to the primary `g` of its first expansion here. Tuning also runs
//! with caller-supplied heuristics ([`boba_star_with`]).
//!
//! Differences from the paper, all of which only cost speed:
//!
//! - The searches alternate one label at a time instead of running in
//!   parallel threads (the crate is `no_std`), unless the `parallel`
//!   feature is on and `boba_star_parallel` or `boba_star_with_parallel`
//!   is used.
//! - The four cost-bounded preliminary searches (see `bounds`) run one
//!   after another instead of two at a time, again unless in parallel.
//! - The search stops as soon as either direction is done, instead of
//!   letting both run out; at that point every solution has been found
//!   (see [`boba_star`]).

mod bounds;
#[cfg(feature = "parallel")]
mod parallel;

use alloc::vec::Vec;

use search_core::{Cost, Heuristic, heuristic_fn};

use crate::boa::{Completions, NoCompletions, Search, Tolerance};
use crate::graph::{Graph, ReverseGraph};
use crate::output::{Output, Solution, Stats};

#[cfg(feature = "parallel")]
pub use parallel::{boba_star_parallel, boba_star_with_parallel};

/// Finds one path for every Pareto-optimal cost vector from `start` to
/// `goal` by searching from both ends.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - [`ReverseGraph::predecessors`] mirrors [`Graph::successors`] exactly,
///   parallel edges included;
/// - finitely many states lie within the bounds `ub1`, `ub2` (the first
///   cost of the `(c2, c1)`-optimal path and the second cost of the
///   `(c1, c2)`-optimal one) of the start and of the goal: the preliminary
///   searches expand exactly those. An unreachable goal is detected by a
///   search from the start, which must then terminate too.
///
/// No heuristic is needed: BOBA* computes exact per-objective distances
/// itself, with four cost-bounded searches that skip states on no
/// Pareto-optimal path; see [`boba_star_with`] to supply your own.
/// Solutions are sorted as in [`boa_star`](crate::boa_star): first
/// objective strictly ascending, second strictly descending. The counters
/// in [`Output::stats`] add up both directions and leave out the
/// preliminary searches.
///
/// # Why stopping early is safe
///
/// The forward search stops once its best open label has `f1 >= ub1`,
/// where `ub1` is the first cost of the backward search's last solution
/// `b`. By then it has found every Pareto-optimal solution with a first
/// cost below `ub1`. Every other Pareto-optimal solution has a first cost
/// above `b`'s, so a second cost below it, and the backward search, which
/// finds solutions in ascending second cost, already has it. The same
/// holds with the roles swapped.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ReverseGraph, boba_star};
///
/// /// Two routes from 0 to 2: fast but costly, or slow but cheap.
/// struct Routes;
///
/// /// The edges as `(from, to, cost)`.
/// const EDGES: [(u8, u8, [u32; 2]); 3] = [(0, 1, [1, 5]), (1, 2, [1, 5]), (0, 2, [4, 1])];
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         out.extend(EDGES.iter().filter(|e| e.0 == *state).map(|e| (e.1, e.2)));
///     }
/// }
///
/// impl ReverseGraph for Routes {
///     fn predecessors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         out.extend(EDGES.iter().filter(|e| e.1 == *state).map(|e| (e.0, e.2)));
///     }
/// }
///
/// let out = boba_star(&Routes, 0, 2);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[2, 10], [4, 1]]);
/// assert_eq!(out.solutions[0].path, [0, 1, 2]);
/// ```
pub fn boba_star<G, C>(graph: &G, start: G::State, goal: G::State) -> Output<G::State, [C; 2]>
where
    G: ReverseGraph<Weight = [C; 2]>,
    C: Cost,
{
    let successors = |s: &G::State, out: &mut Vec<_>| graph.successors(s, out);
    let predecessors = |s: &G::State, out: &mut Vec<_>| graph.predecessors(s, out);
    let Some(bounds) = bounds::Bounds::new(&start, &goal, successors, predecessors) else {
        return Output {
            solutions: Vec::new(),
            stats: Stats::default(),
        };
    };

    let forward_h = heuristic_fn(|s: &G::State| bounds.forward(s));
    let backward_h = heuristic_fn(|s: &G::State| bounds.backward(s));
    let completions = (bounds::Forward(&bounds), bounds::Backward(&bounds));

    bidirectional(graph, start, goal, forward_h, backward_h, completions)
}

/// [`boba_star`] with caller-supplied heuristics instead of the internal
/// preliminary searches.
///
/// - `forward_h(s)` bounds the cost from `s` to `goal` per objective;
/// - `backward_h(s)` bounds the cost from `start` to `s` per objective, in
///   the graph's objective order (the backward search swaps them itself).
///
/// Requirements, in addition to those of [`boba_star`] (finiteness aside):
/// both heuristics are consistent in both objectives, along the edges
/// they are used on (`forward_h(p) <= c(p, s) + forward_h(s)` and
/// `backward_h(s) <= backward_h(p) + c(p, s)` for every edge `p -> s`), and
/// zero at `goal` and at `start` respectively. A heuristic may return
/// `None` only for states on no Pareto-optimal path. Both directions'
/// stopping rule only needs these lower bounds, so the output equals
/// [`boa_star`](crate::boa_star)'s in cost vectors for any such pair,
/// [`ZeroHeuristic`](crate::ZeroHeuristic) included.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ReverseGraph, ZeroHeuristic, boba_star, boba_star_with};
///
/// /// Two routes from 0 to 2: fast but costly, or slow but cheap.
/// struct Routes;
///
/// /// The edges as `(from, to, cost)`.
/// const EDGES: [(u8, u8, [u32; 2]); 3] = [(0, 1, [1, 5]), (1, 2, [1, 5]), (0, 2, [4, 1])];
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         out.extend(EDGES.iter().filter(|e| e.0 == *state).map(|e| (e.1, e.2)));
///     }
/// }
///
/// impl ReverseGraph for Routes {
///     fn predecessors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         out.extend(EDGES.iter().filter(|e| e.1 == *state).map(|e| (e.0, e.2)));
///     }
/// }
///
/// let blind = boba_star_with(&Routes, 0, 2, ZeroHeuristic, ZeroHeuristic);
/// assert_eq!(blind.solutions, boba_star(&Routes, 0, 2).solutions);
/// ```
pub fn boba_star_with<G, C, F, B>(
    graph: &G,
    start: G::State,
    goal: G::State,
    forward_h: F,
    backward_h: B,
) -> Output<G::State, [C; 2]>
where
    G: ReverseGraph<Weight = [C; 2]>,
    C: Cost,
    F: Heuristic<G::State, [C; 2]>,
    B: Heuristic<G::State, [C; 2]>,
{
    let completions = (NoCompletions, NoCompletions);

    bidirectional(graph, start, goal, forward_h, backward_h, completions)
}

/// Runs the two BOA* directions in lockstep, exchanging their bounds and,
/// for heuristic tuning, their first expansions. `completions` are the
/// known completions of both directions for the early solution update:
/// forward to the goal, backward to the start in swapped order.
fn bidirectional<G, C, F, B, KF, KB>(
    graph: &G,
    start: G::State,
    goal: G::State,
    forward_h: F,
    mut backward_h: B,
    completions: (KF, KB),
) -> Output<G::State, [C; 2]>
where
    G: ReverseGraph<Weight = [C; 2]>,
    C: Cost,
    F: Heuristic<G::State, [C; 2]>,
    B: Heuristic<G::State, [C; 2]>,
    KF: Completions<G::State, C>,
    KB: Completions<G::State, C>,
{
    let backward_h = heuristic_fn(move |s: &G::State| backward_h.estimate(s).map(swap));
    let reversed = Reversed(graph);

    let (forward_k, backward_k) = completions;
    let mut forward = Search::with_completions(
        graph,
        forward_h,
        equals(goal.clone()),
        Tolerance::EXACT,
        forward_k,
    );
    let mut backward = Search::with_completions(
        &reversed,
        backward_h,
        equals(start.clone()),
        Tolerance::EXACT,
        backward_k,
    );

    forward.push_start(start);
    backward.push_start(goal);

    loop {
        if !forward.step() {
            break;
        }

        tune(&mut forward, &mut backward);
        backward.set_primary_bound(forward.secondary_bound());
        if !backward.step() {
            break;
        }

        tune(&mut backward, &mut forward);
        forward.set_primary_bound(backward.secondary_bound());
    }

    merge(forward.into_output(), backward.into_output())
}

/// Heuristic tuning (Ahmadi et al., line 10 of Algorithm 2): the first
/// expansion of a state in one direction, at primary cost `g`, raises the
/// other direction's secondary heuristic there to `g`.
///
/// Every cheaper path to the state was pruned in `from`, so its
/// completions are weakly dominated by solutions `from` already has; the
/// other direction need not find them.
fn tune<G1, G2, H1, H2, P1, P2, K1, K2, C>(
    from: &mut Search<'_, G1, H1, C, P1, K1>,
    to: &mut Search<'_, G2, H2, C, P2, K2>,
) where
    G1: Graph<Weight = [C; 2]>,
    G2: Graph<State = G1::State, Weight = [C; 2]>,
    H1: Heuristic<G1::State, [C; 2]>,
    H2: Heuristic<G1::State, [C; 2]>,
    P1: Fn(&G1::State) -> bool,
    P2: Fn(&G1::State) -> bool,
    K1: Completions<G1::State, C>,
    K2: Completions<G1::State, C>,
    C: Cost,
{
    if let Some((state, g)) = from.take_first_expansion() {
        to.raise_secondary(state, g);
    }
}

/// The reversed graph with swapped objectives, searched by the backward
/// direction: its successors are the original predecessors.
struct Reversed<'a, G>(&'a G);

impl<G, C> Graph for Reversed<'_, G>
where
    G: ReverseGraph<Weight = [C; 2]>,
    C: Cost,
{
    type State = G::State;
    type Weight = [C; 2];

    fn successors(&self, state: &G::State, out: &mut Vec<(G::State, [C; 2])>) {
        self.0.predecessors(state, out);

        for (_, weight) in out.iter_mut() {
            *weight = swap(*weight);
        }
    }
}

/// A predicate that matches exactly `target`.
fn equals<S: Ord>(target: S) -> impl Fn(&S) -> bool {
    move |s| *s == target
}

/// Swaps the two objectives of a cost vector.
fn swap<C>([first, second]: [C; 2]) -> [C; 2] {
    [second, first]
}

/// Joins the two directions' outputs into one, in forward terms.
///
/// The forward solutions cover the low end of the first objective, the
/// backward ones the high end, in descending first-objective order; their
/// paths run from the goal to the start with swapped costs.
///
/// The two ends can overlap by one cost vector: a direction's last
/// solution may come from an early solution update, made before the other
/// direction's bound arrived, and be dominated by (or equal to) the other
/// direction's first. The join keeps only the non-dominated ones.
fn merge<S, C: Cost>(forward: Output<S, [C; 2]>, backward: Output<S, [C; 2]>) -> Output<S, [C; 2]> {
    let mut all = forward.solutions;

    all.extend(backward.solutions.into_iter().rev().map(|mut solution| {
        solution.path.reverse();
        Solution {
            cost: swap(solution.cost),
            path: solution.path,
        }
    }));

    // Stable: of two equal cost vectors, the forward solution stays.
    all.sort_by_key(|solution| solution.cost);
    let mut solutions: Vec<Solution<S, [C; 2]>> = Vec::with_capacity(all.len());
    for solution in all {
        if solutions
            .last()
            .is_none_or(|last| solution.cost[1] < last.cost[1])
        {
            solutions.push(solution);
        }
    }

    Output {
        solutions,
        stats: Stats {
            expanded: forward.stats.expanded + backward.stats.expanded,
            generated: forward.stats.generated + backward.stats.generated,
        },
    }
}
