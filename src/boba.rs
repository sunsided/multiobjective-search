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
//! single-objective distances as heuristics. Each search's last solution
//! bounds the other's primary objective, so they split the front between
//! them and stop where they meet.
//!
//! ```mermaid
//! graph TD
//!     dists[Dijkstra per objective:<br/>to the goal and from the start] --> fwd[Forward BOA*, order f1 f2]
//!     dists --> bwd[Backward BOA*, order f2 f1]
//!     fwd -- "last c2 bounds backward f2" --> bwd
//!     bwd -- "last c1 bounds forward f1" --> fwd
//!     fwd --> merge[Forward solutions, then backward ones reversed]
//!     bwd --> merge
//! ```
//!
//! Differences from the paper, all of which only cost speed:
//!
//! - The searches alternate one label at a time instead of running in
//!   parallel threads (the crate is `no_std`).
//! - The heuristics come from four full single-objective Dijkstra searches,
//!   not the paper's cost-bounded A* searches.
//! - The paper's "early solution update" (joining a label with its
//!   complementary shortest path) and "secondary heuristic tuning" are not
//!   implemented.
//! - The search stops as soon as either direction is done, instead of
//!   letting both run out; at that point every solution has been found
//!   (see [`boba_star`]).

mod dijkstra;

use alloc::vec::Vec;

use crate::boa::Search;
use crate::cost::Cost;
use crate::graph::{Graph, ReverseGraph};
use crate::output::{Output, Solution, Stats};

/// Finds one path for every Pareto-optimal cost vector from `start` to
/// `goal` by searching from both ends.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - [`ReverseGraph::predecessors`] mirrors [`Graph::successors`] exactly,
///   parallel edges included;
/// - the states reachable from `start`, and the states that reach `goal`,
///   are finite: the heuristics come from one-to-all Dijkstra searches over
///   both sets.
///
/// No heuristic is needed: BOBA* computes exact per-objective distances
/// itself. Solutions are sorted as in [`boa_star`](crate::boa_star): first
/// objective strictly ascending, second strictly descending. The counters
/// in [`Output::stats`] add up both directions and leave out the Dijkstra
/// searches.
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
    let to_goal = dijkstra::Distances::new(&goal, |s, out| graph.predecessors(s, out));
    let from_start = dijkstra::Distances::new(&start, |s, out| graph.successors(s, out));
    let forward_h = |s: &G::State| to_goal.get(s);
    let backward_h = |s: &G::State| from_start.get(s).map(swap);
    let reversed = Reversed(graph);

    let is_goal = equals(goal.clone());
    let is_start = equals(start.clone());

    let mut forward = Search::new(graph, &forward_h, is_goal, C::default());
    let mut backward = Search::new(&reversed, &backward_h, is_start, C::default());
    forward.push_start(start);
    backward.push_start(goal);

    loop {
        if !forward.step() {
            break;
        }

        backward.set_primary_bound(forward.secondary_bound());
        if !backward.step() {
            break;
        }

        forward.set_primary_bound(backward.secondary_bound());
    }

    merge(forward.into_output(), backward.into_output())
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
fn merge<S, C>(forward: Output<S, [C; 2]>, backward: Output<S, [C; 2]>) -> Output<S, [C; 2]> {
    let mut solutions = forward.solutions;

    solutions.extend(backward.solutions.into_iter().rev().map(|mut solution| {
        solution.path.reverse();
        Solution {
            cost: swap(solution.cost),
            path: solution.path,
        }
    }));

    Output {
        solutions,
        stats: Stats {
            expanded: forward.stats.expanded + backward.stats.expanded,
            generated: forward.stats.generated + backward.stats.generated,
        },
    }
}
