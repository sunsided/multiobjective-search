//! A*pex tests: the multiplicative approximation guarantee for two to four
//! objectives, property-tested against the brute-force Pareto front.

use multiobjective_search::{
    Epsilon, MergeRule, Output, ZeroHeuristic, apex_star, apex_star_with, heuristic_fn,
};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::check::{check, check_paths};
use crate::fixtures::{EdgeList, Problem, exact_heuristic, problems};

/// Returns `true` if `x <= (1 + eps) * y`, in exact integer arithmetic.
fn within(eps: Epsilon, x: u32, y: u32) -> bool {
    u64::from(eps.den()) * u64::from(x) <= u64::from(eps.den() + eps.num()) * u64::from(y)
}

/// Checks the approximation contract of `out`: every Pareto-optimal cost
/// is covered within `eps`, and the solutions are real paths, mutually
/// non-dominated and lexicographically ascending.
fn check_apex<const N: usize>(
    problem: &Problem<N>,
    out: &Output<usize, [u32; N]>,
    eps: [Epsilon; N],
) -> Result<(), TestCaseError> {
    let front = pareto_costs(problem);
    let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();

    for s in &front {
        prop_assert!(
            costs
                .iter()
                .any(|r| (0..N).all(|i| within(eps[i], r[i], s[i]))),
            "{:?} not covered by {:?}",
            s,
            costs
        );
    }

    for (k, a) in costs.iter().enumerate() {
        for b in &costs[k + 1..] {
            prop_assert!(a < b);
            prop_assert!(!(0..N).all(|i| a[i] <= b[i]) && !(0..N).all(|i| b[i] <= a[i]));
        }
    }

    prop_assert_eq!(costs.is_empty(), front.is_empty());
    check_paths(problem, out)
}

/// Runs the guarantee check with the zero and the exact heuristic, and the
/// exactness check with zero tolerances, for every merge rule.
fn check_all<const N: usize>(problem: &Problem<N>, eps: [Epsilon; N]) -> Result<(), TestCaseError> {
    let h = exact_heuristic(problem);
    let goal = |v: &usize| problem.goals[*v];
    let (graph, start) = (&problem.graph, problem.start);
    let exact = heuristic_fn(|v: &usize| h[*v]);

    for rule in [MergeRule::Greedy, MergeRule::ReverseLex] {
        let run = |eps| {
            [
                apex_star_with(graph, ZeroHeuristic, start, goal, eps, rule),
                apex_star_with(graph, exact, start, goal, eps, rule),
            ]
        };

        for out in run(eps) {
            check_apex(problem, &out, eps)?;
        }

        for out in run([Epsilon::ZERO; N]) {
            check(problem, &out)?;
        }
    }

    Ok(())
}

/// A strategy for tolerances `num / den` up to 300 %, zero included.
fn epsilon() -> impl Strategy<Value = Epsilon> {
    (0..=3_u32, 1..=4_u32).prop_map(|(num, den)| Epsilon::new(num, den))
}

/// `problem` with every edge cost scaled to `25 c + 1`, so that small
/// tolerances matter.
fn scaled<const N: usize>(mut problem: Problem<N>) -> Problem<N> {
    for edge in &mut problem.graph.edges {
        edge.2 = edge.2.map(|c| c * 25 + 1);
    }

    problem
}

/// Ten percent in every objective.
const TEN_PERCENT: [Epsilon; 2] = [Epsilon::new(1, 10); 2];

/// The solution cost vectors of `out`.
fn costs(out: &Output<usize, [u32; 2]>) -> Vec<[u32; 2]> {
    out.solutions.iter().map(|s| s.cost).collect()
}

/// Two routes to state 1 merge there only under the paper's `f`-based
/// bound: `g = [20, 10]` is not within 10 % of the apex `[10, 10]`, but
/// `f = [120, 110]` is within 10 % of `[110, 110]`. One pair is expanded
/// at state 1 instead of two.
#[test]
fn merges_under_the_f_based_bound() {
    let graph = EdgeList {
        nodes: 3,
        edges: vec![(0, 1, [10, 20]), (0, 1, [20, 10]), (1, 2, [100, 100])],
    };
    let h = [[100, 100], [100, 100], [0, 0]];
    let heuristic = heuristic_fn(|v: &usize| Some(h[*v]));

    let out = apex_star(&graph, heuristic, 0, |v| *v == 2, TEN_PERCENT);

    // [20, 10] has the lexicographically smaller reverse cost, so it
    // represents the merged pair.
    assert_eq!(costs(&out), [[120, 110]]);
    assert_eq!(out.stats.expanded, 2);
}

/// A goal pair popped after a solution merges into it (paper line 11): the
/// solution `[10, 20]` and the later goal pair `[11, 11]` become one pair
/// with apex `[10, 11]` represented by `[11, 11]`.
#[test]
fn merges_goal_pairs_into_solutions() {
    let graph = EdgeList {
        nodes: 3,
        edges: vec![(0, 1, [10, 20]), (0, 2, [11, 1]), (2, 1, [0, 10])],
    };

    let out = apex_star(&graph, ZeroHeuristic, 0, |v| *v == 1, TEN_PERCENT);

    assert_eq!(costs(&out), [[11, 11]]);
    assert_eq!(out.solutions[0].path, [0, 2, 1]);
}

/// A solution that covers a pruned pair lowers its apex to that pair's `f`
/// (paper lines 20 to 22), which stops a later merge that would uncover
/// it. Solution `[2, 2, 1]` prunes `[3, 1, 1]` and its apex drops to
/// `[2, 1, 1]`; merging goal pair `[3, 4, 0]` would then need
/// `4 <= 2 · 1`, so both stay. Without the update the apex `[2, 2, 0]`
/// would let `[3, 4, 0]` represent both, and `[3, 1, 1]` would be lost.
#[test]
fn pruning_lowers_the_solution_apex() {
    let graph = EdgeList {
        nodes: 3,
        edges: vec![(1, 2, [3, 4, 0]), (1, 0, [2, 2, 1]), (1, 2, [3, 1, 1])],
    };
    let eps = [Epsilon::new(2, 1), Epsilon::new(1, 1), Epsilon::ZERO];

    let out = apex_star(&graph, ZeroHeuristic, 1, |v| *v != 1, eps);
    let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();

    assert_eq!(costs, [[2, 2, 1], [3, 4, 0]]);
}

/// The default rule is the paper's greedy G method, and it differs from the
/// L method: against the merged apex `[10, 10]`, `[10, 13]` exceeds it by
/// 60 % of the tolerance and `[14, 10]` by 80 %, so G keeps `[10, 13]`
/// where L keeps `[14, 10]`.
#[test]
fn greedy_is_the_default_and_differs_from_reverse_lex() {
    let graph = EdgeList {
        nodes: 2,
        edges: vec![(0, 1, [10, 13]), (0, 1, [14, 10])],
    };
    let eps = [Epsilon::new(1, 2); 2];
    let run = |rule| apex_star_with(&graph, ZeroHeuristic, 0, |v| *v == 1, eps, rule);

    assert_eq!(
        costs(&apex_star(&graph, ZeroHeuristic, 0, |v| *v == 1, eps)),
        [[10, 13]]
    );
    assert_eq!(costs(&run(MergeRule::Greedy)), [[10, 13]]);
    assert_eq!(costs(&run(MergeRule::ReverseLex)), [[14, 10]]);
}

proptest! {
    /// Two objectives.
    #[test]
    fn two_objectives(p in problems::<2>(), eps in prop::array::uniform(epsilon())) {
        check_all(&p, eps)?;
        check_all(&scaled(p), eps)?;
    }

    /// Three objectives.
    #[test]
    fn three_objectives(p in problems::<3>(), eps in prop::array::uniform(epsilon())) {
        check_all(&p, eps)?;
        check_all(&scaled(p), eps)?;
    }

    /// Four objectives.
    #[test]
    fn four_objectives(p in problems::<4>(), eps in prop::array::uniform(epsilon())) {
        check_all(&p, eps)?;
        check_all(&scaled(p), eps)?;
    }
}

/// Checks an A-A*pex run: every report covers the brute-force front within
/// its factor and is mutually non-dominated, the factors fall, and a
/// complete run is exact.
fn check_anytime<const N: usize>(
    problem: &Problem<N>,
    schedule: &[[Epsilon; N]],
    limit: Option<usize>,
) -> Result<(), TestCaseError> {
    use multiobjective_search::{ApproxFactor, Stats, apex_star_anytime};

    let front = pareto_costs(problem);
    let goal = |v: &usize| problem.goals[*v];
    let mut reports: Vec<(Vec<[u32; N]>, ApproxFactor<u32>)> = Vec::new();
    let result = apex_star_anytime(
        &problem.graph,
        ZeroHeuristic,
        problem.start,
        goal,
        schedule.iter().copied(),
        |s, f| reports.push((s.iter().map(|x| x.cost).collect(), f)),
        move |stats: &Stats| limit.is_some_and(|l| stats.expanded >= l),
    );

    for (costs, factor) in &reports {
        for s in &front {
            prop_assert!(
                costs
                    .iter()
                    .any(|r| (0..N).all(|i| factor.covers(r[i], s[i])))
            );
        }

        for (k, a) in costs.iter().enumerate() {
            for b in &costs[k + 1..] {
                prop_assert!(a < b && !(0..N).all(|i| a[i] <= b[i]));
            }
        }
    }

    for pair in reports.windows(2) {
        prop_assert!(pair[1].1 < pair[0].1);
    }

    if result.complete {
        let costs: Vec<_> = result.solutions.iter().map(|s| s.cost).collect();
        prop_assert_eq!(costs, front);
    }

    check_paths(
        problem,
        &Output {
            solutions: result.solutions,
            stats: result.stats,
        },
    )
}

proptest! {
    /// A-A*pex on falling schedules ending at zero, with and without a
    /// budget, two and three objectives.
    #[test]
    fn anytime_apex(p2 in problems::<2>(), p3 in problems::<3>(), steps in prop::collection::vec((1..=3_u32, 1..=6_u32), 0..3), limit in prop::option::of(0..30_usize)) {
        let mut eps: Vec<Epsilon> = steps.iter().map(|&(n, d)| Epsilon::new(n, d)).collect();
        eps.sort_by(|a, b| (u64::from(b.num()) * u64::from(a.den())).cmp(&(u64::from(a.num()) * u64::from(b.den()))));
        eps.push(Epsilon::ZERO);
        let s2: Vec<[Epsilon; 2]> = eps.iter().map(|&e| [e; 2]).collect();
        let s3: Vec<[Epsilon; 3]> = eps.iter().map(|&e| [e; 3]).collect();
        check_anytime(&scaled(p2), &s2, limit)?;
        check_anytime(&scaled(p3), &s3, limit)?;
    }
}
