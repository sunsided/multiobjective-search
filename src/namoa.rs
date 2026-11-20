//! NAMOA*: multi-objective A* with per-state Pareto frontiers, and its
//! dimensionality-reduced variant NAMOA*dr.
//!
//! Mandow, Pérez de la Cruz: "A New Approach to Multiobjective A* Search",
//! IJCAI 2005; "Multiobjective A* search with consistent heuristics",
//! Journal of the ACM 57(5), 2010. Pulido, Mandow, Pérez de la Cruz:
//! "Dimensionality reduction in multiobjective shortest path search",
//! Computers & Operations Research, 2015.
//!
//! Both keep, per state, the cost vectors of the labels still open and of
//! the labels already expanded. A new label survives only if no open label,
//! no expanded label and no solution weakly dominates it; it removes the
//! open labels it dominates. Labels are expanded in lexicographic `f`
//! order. The engine is shared with EMOA*, see
//! [`crate::frontier_search`].

use search_core::{Cost, Heuristic};

use crate::frontier_search;
use crate::graph::Graph;
use crate::output::Output;
use crate::pareto::{Front, TruncatedFront};

/// Finds one path for every Pareto-optimal cost vector from `start` to a
/// state that satisfies `is_goal`, for any number of objectives `N`.
///
/// This is the classic NAMOA*. It is the most permissive algorithm in the
/// crate and the slowest; prefer [`namoa_star_dr`] or
/// [`emoa_star`](crate::emoa_star) when the heuristic is consistent, and
/// [`boa_star`](crate::boa_star) for two objectives.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is admissible in every objective: it never overestimates
///   the cost to the nearest goal, and so is zero on every goal. It need
///   not be consistent;
/// - the graph is finite, or every edge costs at least some fixed `δ > 0`
///   in every objective. Otherwise the search may not terminate on an
///   infinite implicit graph (Mandow and Pérez de la Cruz 2010, Theorem
///   4.3, assumption 3).
///
/// With these, the result is the full Pareto-optimal set (Theorem 4.9),
/// and every solution recorded along the way is already Pareto-optimal
/// (Corollary 4.7). With a consistent heuristic, only paths that are
/// non-dominated at their state are expanded (Theorem 5.8).
///
/// With [`ZeroHeuristic`](crate::ZeroHeuristic) this is the multi-objective
/// label-setting algorithm (Martins 1984). Solutions are sorted
/// lexicographically ascending, at most one per cost vector.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, namoa_star};
///
/// /// Three routes from 0 to 3 trading time, money and risk.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 3];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 3])>) {
///         if *state == 0 {
///             out.extend([(1, [1, 5, 1]), (2, [4, 1, 1]), (3, [9, 9, 9])]);
///         } else if *state < 3 {
///             out.push((3, [1, 1, 3]));
///         }
///     }
/// }
///
/// let out = namoa_star(&Routes, ZeroHeuristic, 0, |s| *s == 3);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[2, 6, 4], [5, 2, 4]]);
/// ```
pub fn namoa_star<G, H, C, P, const N: usize>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    frontier_search::run::<G, H, C, P, Front<C, N>, N>(graph, heuristic, start, is_goal, true)
}

/// NAMOA*dr: [`namoa_star`] with dimensionality reduction, for consistent
/// heuristics.
///
/// With a consistent heuristic and lexicographic expansion order, every
/// label expanded at a state is lexicographically no greater than any
/// later label there, and every solution found is lexicographically no
/// greater than any later `f`. Dominance against expanded labels and
/// solutions then ignores the first objective, which saves one comparison
/// per check and, for three objectives, turns those checks into `O(log n)`
/// binary searches. Checks against open labels stay full.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is consistent in every objective
///   (`h(s) <= c(s, t) + h(t)` for every edge) and zero on every goal.
///
/// Output is identical in cost vectors to [`namoa_star`].
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, namoa_star_dr};
///
/// /// A diamond: two routes from 0 to 3.
/// struct Diamond;
///
/// impl Graph for Diamond {
///     type State = u8;
///     type Weight = [u32; 3];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 3])>) {
///         match state {
///             0 => out.extend([(1, [1, 2, 3]), (2, [3, 2, 1])]),
///             1 | 2 => out.push((3, [1, 1, 1])),
///             _ => {}
///         }
///     }
/// }
///
/// let out = namoa_star_dr(&Diamond, ZeroHeuristic, 0, |s| *s == 3);
/// assert_eq!(out.solutions.len(), 2);
/// assert_eq!(out.solutions[1].path, [0, 2, 3]);
/// ```
pub fn namoa_star_dr<G, H, C, P, const N: usize>(
    graph: &G,
    heuristic: H,
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
        graph, heuristic, start, is_goal, true,
    )
}
