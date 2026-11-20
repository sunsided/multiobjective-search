//! A-BOA*ε tests: every reported front meets its factor against the
//! brute-force front, factors strictly fall to zero, and the final front
//! equals BOA*'s, for all three variants, with and without a budget.

use multiobjective_search::{
    AnytimeFront, ApproxFactor, Heuristic, Solution, Stats, WeightedSum, ZeroHeuristic, boa_star,
    boa_star_anytime, boa_star_anytime_basic, boa_star_anytime_ws, heuristic_fn,
};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::check::check_paths;
use crate::fixtures::{Problem, exact_heuristic, problems};

/// The solutions `on_front` reports.
type Front = [Solution<usize, [u32; 2]>];

/// One `on_front` report: the costs and the factor.
type Report = (Vec<[u32; 2]>, ApproxFactor<u32>);

/// Returns `true` if `x <= (1 + num / den) * y` in exact integer arithmetic,
/// for the factor `num / den`, zero or infinite.
fn within(factor: ApproxFactor<u32>, x: u32, y: u32) -> bool {
    match factor {
        ApproxFactor::Zero => x <= y,
        ApproxFactor::Infinite => x <= y || y > 0,
        ApproxFactor::Ratio { num, den } => {
            u64::from(den) * u64::from(x) <= (u64::from(den) + u64::from(num)) * u64::from(y)
        }
    }
}

/// Checks one reported front: Pareto-optimal, sorted, and covering the
/// whole front within its factor.
fn check_report(
    front: &[[u32; 2]],
    costs: &[[u32; 2]],
    factor: ApproxFactor<u32>,
) -> Result<(), TestCaseError> {
    for c in costs {
        prop_assert!(front.contains(c), "{:?} is not Pareto-optimal", c);
    }

    prop_assert!(costs.windows(2).all(|w| w[0] < w[1]));
    for s in front {
        let covered = costs
            .iter()
            .any(|r| within(factor, r[0], s[0]) && within(factor, r[1], s[1]));
        prop_assert!(
            covered,
            "{:?} not covered by {:?} within {:?}",
            s,
            costs,
            factor
        );
    }

    Ok(())
}

/// Checks a run: every report, a strictly falling factor schedule, and a
/// final result that is BOA*'s front when complete.
fn check_run(
    problem: &Problem<2>,
    reports: &[Report],
    result: &AnytimeFront<usize, u32>,
) -> Result<(), TestCaseError> {
    let front = pareto_costs(problem);
    for (costs, factor) in reports {
        check_report(&front, costs, *factor)?;
    }

    for pair in reports.windows(2) {
        prop_assert!(pair[1].1 < pair[0].1, "factors not falling: {:?}", pair);
    }

    let costs: Vec<_> = result.solutions.iter().map(|s| s.cost).collect();
    check_report(&front, &costs, result.factor)?;
    if result.complete {
        prop_assert!(result.factor.is_zero());
        prop_assert_eq!(&costs, &front);
        if !front.is_empty() {
            prop_assert!(
                reports
                    .last()
                    .is_some_and(|(c, f)| f.is_zero() && *c == front)
            );
        }
    }

    let output = multiobjective_search::Output {
        solutions: result.solutions.clone(),
        stats: result.stats,
    };
    check_paths(problem, &output)
}

/// Collects reports into `reports`.
fn recorder(reports: &mut Vec<Report>) -> impl FnMut(&Front, ApproxFactor<u32>) + '_ {
    |front, factor| reports.push((front.iter().map(|s| s.cost).collect(), factor))
}

/// A budget of `limit` expansions; `None` is unlimited.
fn budget(limit: Option<usize>) -> impl FnMut(&Stats) -> bool {
    move |stats: &Stats| limit.is_some_and(|l| stats.expanded >= l)
}

/// The exact weighted-sum heuristic `min c1 + w c2` to a goal, by
/// Bellman-Ford; `None` where no goal is reachable.
fn weighted_sum(problem: &Problem<2>, w: u32) -> Vec<Option<u32>> {
    let mut dist: Vec<Option<u32>> = problem.goals.iter().map(|&g| g.then_some(0)).collect();
    for _ in 0..problem.graph.nodes {
        for &(from, to, c) in &problem.graph.edges {
            if let Some(d) = dist[to] {
                let candidate = d + c[0] + w * c[1];
                if dist[from].is_none_or(|cur| candidate < cur) {
                    dist[from] = Some(candidate);
                }
            }
        }
    }

    dist
}

/// Runs all three variants with `heuristic` and checks them.
fn check_variants<H: Heuristic<usize, [u32; 2]> + Copy>(
    problem: &Problem<2>,
    heuristic: H,
    d: u32,
    w: u32,
    limit: Option<usize>,
) -> Result<(), TestCaseError> {
    let (graph, start) = (&problem.graph, problem.start);
    let goal = |v: &usize| problem.goals[*v];

    let mut reports = Vec::new();
    let result = boa_star_anytime(
        graph,
        heuristic,
        start,
        goal,
        d,
        recorder(&mut reports),
        budget(limit),
    );
    check_run(problem, &reports, &result)?;

    let mut reports = Vec::new();
    let result = boa_star_anytime_basic(
        graph,
        heuristic,
        start,
        goal,
        d,
        recorder(&mut reports),
        budget(limit),
    );
    check_run(problem, &reports, &result)?;

    let sums = weighted_sum(problem, w);
    let weighted = WeightedSum {
        weight: w,
        heuristic: heuristic_fn(|v: &usize| sums[*v]),
    };
    let mut reports = Vec::new();
    let result = boa_star_anytime_ws(
        graph,
        heuristic,
        start,
        goal,
        d,
        weighted,
        recorder(&mut reports),
        budget(limit),
    );
    check_run(problem, &reports, &result)?;

    // A zero weighted-sum bound is admissible too, only weaker.
    let zero = WeightedSum {
        weight: w,
        heuristic: heuristic_fn(|_: &usize| Some(0_u32)),
    };
    let mut reports = Vec::new();
    let result = boa_star_anytime_ws(
        graph,
        heuristic,
        start,
        goal,
        d,
        zero,
        recorder(&mut reports),
        budget(limit),
    );
    check_run(problem, &reports, &result)
}

/// `problem` with every edge cost scaled to `25 c + 1`, so that the
/// approximation factors take many values.
fn scaled(mut problem: Problem<2>) -> Problem<2> {
    for edge in &mut problem.graph.edges {
        edge.2 = edge.2.map(|c| c * 25 + 1);
    }

    problem
}

/// The paper's Example 1: `DF((2, 3), (3, 2)) = 0.5`, and a complete run
/// on four routes ends with the exact front after the seeds.
#[test]
fn four_routes() {
    let problem = Problem {
        graph: crate::fixtures::EdgeList {
            nodes: 2,
            edges: vec![
                (0, 1, [10, 100]),
                (0, 1, [40, 45]),
                (0, 1, [42, 41]),
                (0, 1, [100, 10]),
            ],
        },
        start: 0,
        goals: vec![false, true],
    };
    let mut reports = Vec::new();
    let result = boa_star_anytime(
        &problem.graph,
        ZeroHeuristic,
        0,
        |v: &usize| *v == 1,
        4,
        recorder(&mut reports),
        |_: &Stats| false,
    );

    assert!(result.complete);
    assert_eq!(reports[0].0, [[10, 100], [100, 10]]);
    assert_eq!(
        reports.last().unwrap().0,
        [[10, 100], [40, 45], [42, 41], [100, 10]]
    );
    let exact = boa_star(&problem.graph, ZeroHeuristic, 0, |v: &usize| *v == 1);
    assert_eq!(result.solutions, exact.solutions);
}

proptest! {
    /// Complete runs with the zero and the exact heuristic.
    #[test]
    fn complete_runs(p in problems::<2>(), d in 2..=5_u32, w in 1..=3_u32) {
        let h = exact_heuristic(&p);
        check_variants(&p, ZeroHeuristic, d, w, None)?;
        check_variants(&p, heuristic_fn(|v: &usize| h[*v]), d, w, None)?;
        let p = scaled(p);
        let h = exact_heuristic(&p);
        check_variants(&p, ZeroHeuristic, d, w, None)?;
        check_variants(&p, heuristic_fn(|v: &usize| h[*v]), d, w, None)?;
    }

    /// Runs cut short by an expansion budget still report valid fronts.
    #[test]
    fn budgeted_runs(p in problems::<2>(), d in 2..=5_u32, limit in 0..30_usize) {
        let p = scaled(p);
        let h = exact_heuristic(&p);
        check_variants(&p, ZeroHeuristic, d, 1, Some(limit))?;
        check_variants(&p, heuristic_fn(|v: &usize| h[*v]), d, 1, Some(limit))?;
    }
}
