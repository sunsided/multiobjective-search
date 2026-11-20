//! BOA* and approximate BOA* tests: fixed cases and property-based
//! cross-checks against the brute-force reference.

use multiobjective_search::{
    Heuristic, Output, ZeroHeuristic, boa_star, boa_star_approx, heuristic_fn,
};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::check::{check, check_paths};
use crate::fixtures::{EdgeList, Problem, exact_heuristic, problems};

/// Runs BOA* on `problem` with the given heuristic.
fn solve<H: Heuristic<usize, [u32; 2]>>(
    problem: &Problem<2>,
    heuristic: H,
) -> Output<usize, [u32; 2]> {
    boa_star(&problem.graph, heuristic, problem.start, |v| {
        problem.goals[*v]
    })
}

/// Runs BOA* with the exact backward-distance heuristic.
fn solve_informed(problem: &Problem<2>) -> Output<usize, [u32; 2]> {
    let h = exact_heuristic(problem);
    solve(problem, heuristic_fn(|v: &usize| h[*v]))
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
    let out = solve(&p, ZeroHeuristic);

    assert_eq!(out.solutions.len(), 1);
    assert_eq!(out.solutions[0].cost, [0, 0]);
    assert_eq!(out.solutions[0].path, [0]);
}

/// A heuristic passed as `&mut h` keeps its state after the search, so a
/// workspace survives the run.
#[test]
fn heuristic_by_mut_reference_keeps_state() {
    /// The zero heuristic that counts its calls.
    struct Counting(usize);

    impl Heuristic<usize, [u32; 2]> for Counting {
        fn estimate(&mut self, _state: &usize) -> Option<[u32; 2]> {
            self.0 += 1;
            Some([0, 0])
        }
    }

    let p = problem(2, &[(0, 1, [1, 2])], 0, 1);
    let mut h = Counting(0);
    let out = solve(&p, &mut h);

    assert_eq!(out.solutions.len(), 1);
    assert_eq!(h.0, 2);
}

/// An unreachable goal yields no solutions, with and without heuristic.
#[test]
fn unreachable_goal() {
    let p = problem(3, &[(0, 1, [1, 1]), (2, 0, [1, 1])], 0, 2);

    assert_eq!(solve(&p, ZeroHeuristic).solutions.len(), 0);

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

    for out in [solve(&p, ZeroHeuristic), solve_informed(&p)] {
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
        check(&p, &solve(&p, ZeroHeuristic))?;
    }

    /// BOA* with an exact, consistent heuristic finds the same set.
    #[test]
    fn exact_heuristic_matches_brute_force(p in problems::<2>()) {
        check(&p, &solve_informed(&p))?;
    }

    /// Two objectives come out strictly increasing in the first and so
    /// strictly decreasing in the second.
    #[test]
    fn output_is_a_staircase(p in problems::<2>()) {
        let out = solve(&p, ZeroHeuristic);
        for pair in out.solutions.windows(2) {
            prop_assert!(pair[0].cost[0] < pair[1].cost[0] && pair[0].cost[1] > pair[1].cost[1]);
        }
    }
}

/// Checks the guarantee of [`boa_star_approx`] with `slack`: every
/// solution is Pareto-optimal and a valid path, the output is a staircase,
/// and every Pareto-optimal cost is covered within `slack` in the second
/// objective.
fn check_approx<H: Heuristic<usize, [u32; 2]>>(
    problem: &Problem<2>,
    heuristic: H,
    slack: u32,
) -> Result<(), TestCaseError> {
    let goal = |v: &usize| problem.goals[*v];
    let out = boa_star_approx(&problem.graph, heuristic, problem.start, goal, slack);
    let reference = pareto_costs(problem);

    check_paths(problem, &out)?;
    for pair in out.solutions.windows(2) {
        prop_assert!(pair[0].cost[0] < pair[1].cost[0] && pair[0].cost[1] > pair[1].cost[1]);
    }

    for solution in &out.solutions {
        prop_assert!(reference.contains(&solution.cost));
    }

    for s in &reference {
        let covered = out
            .solutions
            .iter()
            .any(|r| r.cost[0] <= s[0] && r.cost[1] <= s[1] + slack);
        prop_assert!(covered, "{s:?} not covered within {slack}");
    }

    Ok(())
}

/// A tolerance of one drops a solution that is within one of its
/// predecessor in the second objective, and keeps both extremes.
#[test]
fn approx_drops_close_solution() {
    let edges = [(0, 1, [1, 3]), (0, 1, [2, 2]), (0, 1, [3, 1])];
    let p = problem(2, &edges, 0, 1);
    let costs = |slack| -> Vec<_> {
        boa_star_approx(&p.graph, ZeroHeuristic, 0, |v| *v == 1, slack)
            .solutions
            .into_iter()
            .map(|s| s.cost)
            .collect()
    };

    assert_eq!(costs(0), [[1, 3], [2, 2], [3, 1]]);
    assert_eq!(costs(1), [[1, 3], [3, 1]]);
    assert_eq!(costs(2), [[1, 3]]);
}

proptest! {
    /// The approximation guarantee holds for small tolerances, with the
    /// zero and the exact heuristic.
    #[test]
    fn approx_covers_front(p in problems::<2>(), slack in 0..=3_u32) {
        let h = exact_heuristic(&p);
        check_approx(&p, ZeroHeuristic, slack)?;
        check_approx(&p, heuristic_fn(|v: &usize| h[*v]), slack)?;
    }
}
