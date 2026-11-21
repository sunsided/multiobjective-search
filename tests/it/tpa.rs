//! TPA tests: exact fronts against brute force, every reported point
//! efficient, budgets, and the hypervolume indicator against a grid count.

use multiobjective_search::{Stats, ZeroHeuristic, heuristic_fn, hypervolume, tpa_paths};
use proptest::prelude::*;
use search_core::BigUint;

use crate::brute_force::pareto_costs;
use crate::check::check_paths;
use crate::fixtures::{Problem, exact_heuristic, problems};

/// Runs TPA with an optional expansion budget and checks it.
fn check<const N: usize>(problem: &Problem<N>, limit: Option<usize>) -> Result<(), TestCaseError> {
    let front = pareto_costs(problem);
    let h = exact_heuristic(problem);
    let goal = |v: &usize| problem.goals[*v];
    let mut reports: Vec<Vec<[u32; N]>> = Vec::new();
    let result = tpa_paths(
        &problem.graph,
        heuristic_fn(|v: &usize| h[*v]),
        problem.start,
        goal,
        |s, _| reports.push(s.iter().map(|x| x.cost).collect()),
        move |stats: &Stats| limit.is_some_and(|l| stats.expanded >= l),
    );

    for costs in &reports {
        for c in costs {
            prop_assert!(front.contains(c), "{:?} is not efficient", c);
        }
    }

    let costs: Vec<_> = result.solutions.iter().map(|s| s.cost).collect();
    let mut unique = costs.clone();
    unique.dedup();
    prop_assert_eq!(&unique, &costs);
    if result.complete {
        prop_assert_eq!(&costs, &front);
        prop_assert!(result.factor.is_zero());
    } else {
        prop_assert!(limit.is_some());
    }

    let zero = tpa_paths(
        &problem.graph,
        ZeroHeuristic,
        problem.start,
        goal,
        |_, _| {},
        |_: &Stats| false,
    );
    let costs: Vec<_> = zero.solutions.iter().map(|s| s.cost).collect();
    prop_assert_eq!(costs, front);
    check_paths(
        problem,
        &multiobjective_search::Output {
            solutions: result.solutions,
            stats: result.stats,
        },
    )
}

proptest! {
    /// Two to four objectives, complete and budgeted.
    #[test]
    fn tpa_matches_brute_force(p2 in problems::<2>(), p3 in problems::<3>(), p4 in problems::<4>(), limit in prop::option::of(0..20_usize)) {
        check(&p2, limit)?;
        check(&p3, limit)?;
        check(&p4, None)?;
    }

    /// The hypervolume equals a count of the dominated grid cells.
    #[test]
    fn hypervolume_counts_cells(points in prop::collection::vec(prop::array::uniform3(0..6_u32), 0..6)) {
        let reference = [6_u32; 3];
        let count = (0..6).flat_map(|x| (0..6).flat_map(move |y| (0..6).map(move |z| [x, y, z])))
            .filter(|c: &[u32; 3]| points.iter().any(|p| (0..3).all(|i| p[i] <= c[i])))
            .count();
        prop_assert_eq!(hypervolume(&points, reference), BigUint::from(count as u128));
    }
}
