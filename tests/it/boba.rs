//! BOBA* tests: fixed cases and property-based cross-checks against the
//! brute-force reference and BOA*.

use multiobjective_search::{Output, boa_star, boba_star};
use proptest::prelude::*;

use crate::check::check;
use crate::fixtures::{EdgeList, Problem, exact_heuristic, single_goal_problems};

/// Runs BOBA* on a single-goal problem.
///
/// # Panics
///
/// Panics if `problem` has no goal.
fn solve(problem: &Problem<2>) -> Output<usize, [u32; 2]> {
    let goal = problem.goals.iter().position(|&g| g).expect("one goal");
    boba_star(&problem.graph, problem.start, goal)
}

/// Builds a problem with a single goal.
fn problem(
    nodes: usize,
    edges: &[(usize, usize, [u32; 2])],
    start: usize,
    goal: usize,
) -> Problem<2> {
    Problem {
        graph: EdgeList {
            nodes,
            edges: edges.to_vec(),
        },
        start,
        goals: (0..nodes).map(|v| v == goal).collect(),
    }
}

/// A start that is the goal yields exactly the empty path at zero cost.
#[test]
fn start_is_goal() {
    let p = problem(2, &[(0, 1, [1, 1]), (1, 0, [0, 0])], 0, 0);
    let out = solve(&p);

    assert_eq!(out.solutions.len(), 1);
    assert_eq!(out.solutions[0].cost, [0, 0]);
    assert_eq!(out.solutions[0].path, [0]);
}

/// An unreachable goal yields no solutions and no labels.
#[test]
fn unreachable_goal() {
    let p = problem(3, &[(0, 1, [1, 1]), (2, 0, [1, 1])], 0, 2);
    let out = solve(&p);

    assert_eq!(out.solutions, []);
    assert_eq!(out.stats.generated, 0);
}

/// The hand-computed graph from the BOA* tests: three Pareto-optimal
/// routes, so at least one comes from each direction.
#[test]
fn hand_computed() {
    let edges = [
        (0, 1, [1, 4]),
        (1, 3, [1, 2]),
        (0, 2, [2, 1]),
        (2, 3, [3, 1]),
        (0, 3, [6, 6]),
        (1, 2, [0, 0]),
        (2, 1, [0, 0]),
    ];
    let found: Vec<_> = solve(&problem(4, &edges, 0, 3))
        .solutions
        .into_iter()
        .map(|s| (s.cost, s.path))
        .collect();

    assert_eq!(
        found,
        [
            ([2, 6], vec![0, 1, 3]),
            ([3, 3], vec![0, 2, 1, 3]),
            ([5, 2], vec![0, 2, 3])
        ]
    );
}

proptest! {
    /// BOBA* finds exactly the Pareto-optimal cost set, in staircase
    /// order, with valid paths (the backward half reversed and its costs
    /// swapped back).
    #[test]
    fn matches_brute_force(p in single_goal_problems::<2>()) {
        let out = solve(&p);
        for pair in out.solutions.windows(2) {
            prop_assert!(pair[0].cost[0] < pair[1].cost[0] && pair[0].cost[1] > pair[1].cost[1]);
        }

        check(&p, &out)?;
    }

    /// BOBA* and BOA* agree on the cost vectors.
    #[test]
    fn agrees_with_boa(p in single_goal_problems::<2>()) {
        let h = exact_heuristic(&p);
        let costs = |out: Output<usize, [u32; 2]>| -> Vec<_> {
            out.solutions.into_iter().map(|s| s.cost).collect()
        };
        let boa = boa_star(&p.graph, &|v: &usize| h[*v], p.start, |v| p.goals[*v]);

        prop_assert_eq!(costs(solve(&p)), costs(boa));
    }
}
