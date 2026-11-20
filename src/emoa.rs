//! EMOA*: enhanced multi-objective A*.
//!
//! Ren, Zhan, Rathinam, Likhachev, Choset: "Enhanced Multi-Objective A*
//! Using Balanced Binary Search Trees", `SoCS` 2022.
//!
//! EMOA* pops labels in lexicographic `f` order and, like NAMOA*dr, checks
//! dominance against expanded labels and solutions on objectives `1..N`
//! only. Unlike NAMOA*, it keeps no per-state frontier of open labels: a
//! new label is checked against expanded labels and solutions only, and is
//! checked again when popped. That trades a larger open list for cheaper
//! generation, which pays off when many labels share a state.
//!
//! The paper stores each truncated frontier in a balanced binary search
//! tree for any number of objectives, with a one-way `O(log n)` traversal
//! in its three-objective special case, TOA*. This crate keeps the
//! frontiers in a sorted `Vec` (see [`Front`](crate::Front)):
//!
//! - three objectives: a two-dimensional frontier with `O(log n)` binary
//!   search queries, as in TOA*, but an `O(n)` tail shift on insertion;
//! - four or more: a scan of the lexicographic prefix that can dominate,
//!   where the paper prunes a tree traversal.

use crate::cost::Cost;
use crate::frontier_search;
use crate::graph::{Graph, Heuristic};
use crate::output::Output;
use crate::pareto::TruncatedFront;

/// Finds one path for every Pareto-optimal cost vector from `start` to a
/// state that satisfies `is_goal`, for any number of objectives `N`.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is consistent in every objective
///   (`h(s) <= c(s, t) + h(t)` for every edge) and zero on every goal.
///
/// Solutions are sorted lexicographically ascending, at most one per cost
/// vector. Same cost vectors as [`namoa_star`](crate::namoa_star).
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, emoa_star};
///
/// /// A line 0 - 1 - 2 with two parallel edges per hop.
/// struct Line;
///
/// impl Graph for Line {
///     type State = u8;
///     type Weight = [u32; 3];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 3])>) {
///         if *state < 2 {
///             out.extend([(state + 1, [1, 2, 0]), (state + 1, [2, 1, 0])]);
///         }
///     }
/// }
///
/// let out = emoa_star(&Line, &ZeroHeuristic, 0, |s| *s == 2);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[2, 4, 0], [3, 3, 0], [4, 2, 0]]);
/// ```
pub fn emoa_star<G, H, C, P, const N: usize>(
    graph: &G,
    heuristic: &H,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    frontier_search::run::<G, H, C, P, TruncatedFront<C, N>, N>(
        graph, heuristic, start, is_goal, false,
    )
}
