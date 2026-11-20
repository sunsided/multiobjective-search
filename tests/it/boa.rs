//! BOA* tests: fixed cases and a property-based cross-check against the
//! brute-force reference.

use multiobjective_search::{Heuristic, Output, ZeroHeuristic, boa_star};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::fixtures::{EdgeList, Problem, exact_heuristic, problems};

/// Runs BOA* on `problem` with the given heuristic.
fn solve<H: Heuristic<usize, [u32; 2]>>(
    problem: &Problem<2>,
    heuristic: &H,
) -> Output<usize, [u32; 2]> {
    boa_star(&problem.graph, heuristic, problem.start, |v| {
        problem.goals[*v]
    })
}

/// Runs BOA* with the exact backward-distance heuristic.
fn solve_informed(problem: &Problem<2>) -> Output<usize, [u32; 2]> {
    let h = exact_heuristic(problem);
    solve(problem, &|v: &usize| h[*v])
}

/// Checks every solution: cost set equals the reference, the order is
/// strictly increasing in the first objective, and each path is a real
/// start-to-goal path whose cost matches.
fn check(problem: &Problem<2>, out: &Output<usize, [u32; 2]>) -> Result<(), TestCaseError> {
    let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
    prop_assert_eq!(&costs, &pareto_costs(problem));

    for pair in costs.windows(2) {
        prop_assert!(pair[0][0] < pair[1][0] && pair[0][1] > pair[1][1]);
    }

    for solution in &out.solutions {
        prop_assert_eq!(solution.path.first(), Some(&problem.start));
        prop_assert!(problem.goals[*solution.path.last().unwrap()]);
        prop_assert!(path_costs(&problem.graph, &solution.path).contains(&solution.cost));
    }

    Ok(())
}

/// Every cost vector a path can have, over all choices among parallel
/// edges. Empty if some step has no edge.
fn path_costs(graph: &EdgeList<2>, path: &[usize]) -> Vec<[u32; 2]> {
    let mut costs = vec![[0, 0]];

    for step in path.windows(2) {
        let edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|(from, to, _)| *from == step[0] && *to == step[1])
            .map(|&(_, _, cost)| cost)
            .collect();

        costs = costs
            .iter()
            .flat_map(|c| edges.iter().map(move |e| [c[0] + e[0], c[1] + e[1]]))
            .collect();
    }

    costs
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

/// A start that is a goal yields exactly the empty path at zero cost.
#[test]
fn start_is_goal() {
    let p = problem(2, &[(0, 1, [1, 1]), (1, 0, [0, 0])], 0, 0);
    let out = solve(&p, &ZeroHeuristic);

    assert_eq!(out.solutions.len(), 1);
    assert_eq!(out.solutions[0].cost, [0, 0]);
    assert_eq!(out.solutions[0].path, [0]);
}

/// An unreachable goal yields no solutions, with and without heuristic.
#[test]
fn unreachable_goal() {
    let p = problem(3, &[(0, 1, [1, 1]), (2, 0, [1, 1])], 0, 2);

    assert_eq!(solve(&p, &ZeroHeuristic).solutions.len(), 0);

    let informed = solve_informed(&p);
    assert_eq!(informed.solutions.len(), 0);
    assert_eq!(informed.stats.generated, 0);
}

/// A hand-computed graph with a zero-cost cycle and dominated routes.
///
/// Routes from 0 to 3: `0-1-3` costs `[2, 6]`, `0-2-1-3` costs `[3, 3]`,
/// `0-2-3` costs `[5, 2]`. Dominated: the direct edge `[6, 6]` and
/// `0-1-2-3` at `[4, 5]`.
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
    let p = problem(4, &edges, 0, 3);

    for out in [solve(&p, &ZeroHeuristic), solve_informed(&p)] {
        let found: Vec<_> = out
            .solutions
            .iter()
            .map(|s| (s.cost, s.path.clone()))
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
}

proptest! {
    /// Uninformed BOA* finds exactly the Pareto-optimal cost set.
    #[test]
    fn zero_heuristic_matches_brute_force(p in problems::<2>()) {
        check(&p, &solve(&p, &ZeroHeuristic))?;
    }

    /// BOA* with an exact, consistent heuristic finds the same set.
    #[test]
    fn exact_heuristic_matches_brute_force(p in problems::<2>()) {
        check(&p, &solve_informed(&p))?;
    }
}
