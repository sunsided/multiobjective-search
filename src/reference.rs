//! A brute-force reference for cross-checking the search algorithms.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use search_core::Cost;

use crate::cost::{Dominates, add};
use crate::graph::Graph;
use crate::output::{Output, Solution, Stats};

/// Finds one path for every Pareto-optimal cost vector by enumerating
/// every simple path from `start`, for any number of objectives.
///
/// Meant for tests on small graphs: it takes time exponential in the
/// graph size, and recursion depth up to the longest simple path. With
/// non-negative costs every Pareto-optimal cost vector is the cost of
/// some simple path, so the result is exact. It shares no dominance code
/// with the searches beyond [`Dominates`], so a disagreement points at the
/// search. For a polynomial-per-label reference, use
/// [`namoa_star`](crate::namoa_star) with
/// [`ZeroHeuristic`](crate::ZeroHeuristic), the classic multi-objective
/// label-setting algorithm.
///
/// Paths do not continue through a goal: with non-negative costs, a
/// longer path to another goal is never cheaper. Solutions are sorted
/// lexicographically ascending, one per cost vector; among equal-cost
/// paths the first one found is kept. [`Output::stats`] counts the states
/// visited by the enumeration as `expanded`, and the paths reaching a goal
/// as `generated`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, brute_force, namoa_star};
///
/// /// Two parallel routes from 0 to 1, plus a dominated one.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 3];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 3])>) {
///         if *state == 0 {
///             out.extend([(1, [1, 2, 3]), (1, [3, 2, 1]), (1, [3, 3, 3])]);
///         }
///     }
/// }
///
/// let reference = brute_force(&Routes, 0, |s| *s == 1);
/// let search = namoa_star(&Routes, ZeroHeuristic, 0, |s| *s == 1);
/// assert_eq!(reference.solutions, search.solutions);
/// ```
pub fn brute_force<G, C, P, const N: usize>(
    graph: &G,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut enumeration = Enumeration {
        graph,
        is_goal,
        path: Vec::new(),
        on_path: BTreeSet::new(),
        found: Vec::new(),
        stats: Stats::default(),
    };
    enumeration.visit(start, [C::ZERO; N]);

    Output {
        solutions: non_dominated(enumeration.found),
        stats: enumeration.stats,
    }
}

/// State of the depth-first enumeration of simple paths.
struct Enumeration<'a, G: Graph, P, const N: usize> {
    /// The graph being enumerated.
    graph: &'a G,
    /// Goal test.
    is_goal: P,
    /// States of the current path, from the start.
    path: Vec<G::State>,
    /// The same states as `path`, for `O(log n)` membership.
    on_path: BTreeSet<G::State>,
    /// Every start-to-goal path found, dominated ones included.
    found: Vec<Solution<G::State, G::Weight>>,
    /// Effort counters.
    stats: Stats,
}

impl<G, C, P, const N: usize> Enumeration<'_, G, P, N>
where
    G: Graph<Weight = [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    /// Extends the current path by `state`, reached at cost `cost`, then
    /// records it if `state` is a goal or recurses into every successor
    /// not already on the path.
    fn visit(&mut self, state: G::State, cost: [C; N]) {
        self.stats.expanded += 1;
        self.path.push(state.clone());

        if (self.is_goal)(&state) {
            self.stats.generated += 1;
            self.found.push(Solution {
                cost,
                path: self.path.clone(),
            });
        } else {
            let mut successors = Vec::new();
            self.graph.successors(&state, &mut successors);
            self.on_path.insert(state);

            for (next, weight) in successors {
                if !self.on_path.contains(&next) {
                    self.visit(next, add(cost, weight));
                }
            }
        }

        if let Some(last) = self.path.pop() {
            self.on_path.remove(&last);
        }
    }
}

/// Keeps one solution per non-dominated cost vector, sorted
/// lexicographically by cost, with a naive `O(n²)` filter.
fn non_dominated<S, C: Cost, const N: usize>(
    found: Vec<Solution<S, [C; N]>>,
) -> Vec<Solution<S, [C; N]>> {
    let costs: Vec<_> = found.iter().map(|s| s.cost).collect();
    let mut kept: Vec<_> = found
        .into_iter()
        .filter(|s| !costs.iter().any(|c| c.dominates(&s.cost)))
        .collect();

    // Stable, so the first path found wins among equal costs.
    kept.sort_by_key(|s| s.cost);
    kept.dedup_by_key(|s| s.cost);

    kept
}
