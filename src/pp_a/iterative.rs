//! Iterative-PP-A*: the anytime preprint's PP-A* baseline.

use alloc::vec::Vec;

use search_core::{Budget, Cost, Heuristic, WideMul};

use crate::anytime_boa::{AnytimeFront, ApproxFactor};
use crate::epsilon::Epsilon;
use crate::graph::Graph;
use crate::output::{Solution, Stats};
use crate::pp_a::pp_a_star;

/// Iterative-PP-A*: [`pp_a_star`] run from scratch for every tolerance of
/// `schedule`, the anytime baseline of "Anytime Approximate Bi-Objective
/// Search" (anonymous preprint, crl.cs.technion.ac.il), which uses
/// `[0.01, 0.001, 0.0001, 0]`.
///
/// Each run with tolerance `ε` (on both objectives) returns solutions that
/// cover every solution within `1 + ε`. Whenever that lowers the factor,
/// `on_front` gets the run's solutions and `ε` as an [`ApproxFactor`]. The
/// solutions are mutually non-dominated but, before the last run, not
/// necessarily Pareto-optimal (PP-A* returns representatives). The result
/// is [`AnytimeFront::complete`] when the schedule ran to its end and that
/// end is zero; `budget` is checked between runs. Nothing is reused
/// between runs, which is what [`boa_star_anytime`](crate::boa_star_anytime)
/// improves on.
///
/// Requirements, not checked at run time: non-negative edge costs and a
/// `heuristic` consistent in both objectives and zero on every goal.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Epsilon, Graph, ZeroHeuristic, pp_a_star_iterative};
///
/// /// Four routes from 0 to 1 on a convex front.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [100, 40]), (1, [105, 30]), (1, [150, 20]), (1, [200, 19])]);
///         }
///     }
/// }
///
/// let schedule = [Epsilon::new(1, 10), Epsilon::ZERO];
/// let mut sizes = Vec::new();
/// let result = pp_a_star_iterative(
///     &Routes,
///     ZeroHeuristic,
///     0,
///     |s| *s == 1,
///     schedule,
///     |front, _| sizes.push(front.len()),
///     |_: &_| false,
/// );
///
/// assert!(result.complete);
/// assert_eq!(sizes, [2, 4]);
/// ```
#[expect(
    clippy::needless_pass_by_value,
    reason = "takes the start by value like every other search in the crate"
)]
pub fn pp_a_star_iterative<G, H, C, P>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
    schedule: impl IntoIterator<Item = Epsilon>,
    mut on_front: impl FnMut(&[Solution<G::State, [C; 2]>], ApproxFactor<C>),
    mut budget: impl Budget<Stats>,
) -> AnytimeFront<G::State, C>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let mut front = AnytimeFront {
        solutions: Vec::new(),
        factor: ApproxFactor::Infinite,
        complete: false,
        stats: Stats::default(),
    };
    let mut ran = false;

    for eps in schedule {
        if ran && budget.exhausted(&front.stats) {
            return front;
        }

        let first = !ran;
        ran = true;

        let out = pp_a_star(graph, &mut heuristic, start.clone(), &is_goal, [eps; 2]);
        front.stats.expanded += out.stats.expanded;
        front.stats.generated += out.stats.generated;
        front.complete = eps == Epsilon::ZERO;

        let factor = as_factor(eps, &out.solutions);
        if first || factor < front.factor {
            front.solutions = out.solutions;
            front.factor = factor;
            on_front(&front.solutions, factor);
        }
    }

    front
}

/// `eps` as an [`ApproxFactor`] over cost type `C`: `num · u / den · u` for
/// the smallest positive cost component `u` among `solutions`, since `C`
/// has no unit of its own. Zero when there is no such component (no
/// solution, or a zero-cost one that covers everything); infinite if the
/// scaled denominator saturates, which is a valid but loose bound.
pub(crate) fn as_factor<S, C: Cost + WideMul, const N: usize>(
    eps: Epsilon,
    solutions: &[Solution<S, [C; N]>],
) -> ApproxFactor<C> {
    let unit = solutions
        .iter()
        .flat_map(|s| s.cost)
        .filter(|&c| c > C::ZERO)
        .min();

    match unit {
        _ if eps.num() == 0 => ApproxFactor::Zero,
        None => ApproxFactor::Zero,
        Some(unit) => {
            let den = unit.saturating_times(eps.den());
            let num = unit.saturating_times(eps.num());
            if den.saturating_add(den) == den || num.saturating_add(num) == num {
                ApproxFactor::Infinite
            } else {
                ApproxFactor::Ratio { num, den }
            }
        }
    }
}
