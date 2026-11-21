//! A-A*pex: anytime A*pex over a falling tolerance schedule, reusing the
//! solutions of earlier runs.

use alloc::vec::Vec;

use search_core::{Budget, Cost, Heuristic, WideMul};

use crate::anytime_boa::{AnytimeFront, ApproxFactor};
use crate::apex::{MergeRule, Search};
use crate::epsilon::Epsilon;
use crate::graph::Graph;
use crate::output::{Solution, Stats};
use crate::pareto::Front;
use crate::pp_a::iterative::as_factor;

/// A-A*pex: anytime A*pex for any number of objectives. Runs A*pex for
/// each tolerance vector of `schedule` in turn and reports, after each run
/// that lowers the factor, the solutions so far and a factor they achieve.
///
/// The idea of Zhang et al. ("A-A*pex: Efficient Anytime Approximate
/// Multi-Objective Search", `SoCS` 2024, cited by the survey of Salzman et
/// al., arXiv 2510.25504): reuse earlier search effort when the tolerance
/// tightens. The paper was not available; this implements the reuse that
/// keeps A*pex's guarantee without its internals: every earlier solution
/// is a real path, so each run starts with them as known solutions and
/// prunes every pair one of them already covers within the new
/// tolerances (checked in every objective), and the reported front is the
/// non-dominated union. Pairs merged under a looser tolerance are not
/// reused, since their dropped paths cannot be recovered.
///
/// The factor reported for a run is its largest tolerance. The result is
/// complete when the schedule ran out with all tolerances zero (then the
/// front is exact). `budget` is checked between runs.
///
/// Requirements as for [`apex_star`](crate::apex_star).
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Epsilon, Graph, ZeroHeuristic, apex_star_anytime};
///
/// /// Three routes from 0 to 1 with three objectives.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 3];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 3])>) {
///         if *state == 0 {
///             out.extend([(1, [100, 100, 50]), (1, [104, 98, 51]), (1, [50, 200, 100])]);
///         }
///     }
/// }
///
/// let schedule = [[Epsilon::new(1, 10); 3], [Epsilon::ZERO; 3]];
/// let mut sizes = Vec::new();
/// let result = apex_star_anytime(&Routes, ZeroHeuristic, 0, |s| *s == 1, schedule, |front, _| sizes.push(front.len()), |_: &_| false);
/// assert!(result.complete && result.factor.is_zero());
/// assert_eq!(sizes, [2, 3]);
/// ```
#[expect(
    clippy::needless_pass_by_value,
    reason = "takes the start by value like every other search in the crate"
)]
pub fn apex_star_anytime<G, H, C, P, const N: usize>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
    schedule: impl IntoIterator<Item = [Epsilon; N]>,
    mut on_front: impl FnMut(&[Solution<G::State, [C; N]>], ApproxFactor<C>),
    mut budget: impl Budget<Stats>,
) -> AnytimeFront<G::State, C, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
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

        let mut search = Search::new(
            graph,
            &mut heuristic,
            &is_goal,
            eps,
            MergeRule::Greedy,
            false,
        );
        search.seeds = front.solutions.iter().map(|s| s.cost).collect();
        search.push_start(start.clone());
        search.run();
        let out = search.into_output();
        front.stats.expanded += out.stats.expanded;
        front.stats.generated += out.stats.generated;
        front.complete = eps.iter().all(|e| *e == Epsilon::ZERO);

        let solutions = union(core::mem::take(&mut front.solutions), out.solutions);
        let factor = as_factor(widest(eps), &solutions);
        front.solutions = solutions;
        if first || factor < front.factor {
            front.factor = factor;
            on_front(&front.solutions, factor);
        }
    }

    front
}

/// The largest tolerance of `eps`.
fn widest<const N: usize>(eps: [Epsilon; N]) -> Epsilon {
    eps.into_iter()
        .reduce(|a, b| {
            let larger =
                u64::from(b.num()) * u64::from(a.den()) > u64::from(a.num()) * u64::from(b.den());
            if larger { b } else { a }
        })
        .unwrap_or(Epsilon::ZERO)
}

/// The non-dominated union of two solution sets, sorted by cost.
fn union<S, C: Cost, const N: usize>(
    old: Vec<Solution<S, [C; N]>>,
    new: Vec<Solution<S, [C; N]>>,
) -> Vec<Solution<S, [C; N]>> {
    let mut all: Vec<_> = old.into_iter().chain(new).collect();
    all.sort_by_key(|s| s.cost);
    let mut front = Front::new();
    all.retain(|s| front.insert(s.cost));
    all
}
