//! NAMOA*, NAMOA*dr and EMOA* tests: hand-computed cases, and
//! property-based cross-checks against the brute-force reference for two, three and four objectives, and
//! against BOA* for two.

use multiobjective_search::{
    Heuristic, Output, ZeroHeuristic, boa_star, emoa_star, heuristic_fn, namoa_star, namoa_star_dr,
};
use proptest::prelude::*;

use crate::check::check;
use crate::fixtures::{EdgeList, Problem, exact_heuristic, patchy_heuristic, problems};

/// The three `N`-objective algorithms under test.
#[derive(Debug, Clone, Copy)]
enum Algorithm {
    /// [`namoa_star`].
    Namoa,
    /// [`namoa_star_dr`].
    NamoaDr,
    /// [`emoa_star`].
    Emoa,
}

/// Runs `algorithm` on `problem` with `heuristic`.
fn solve<H: Heuristic<usize, [u32; N]>, const N: usize>(
    algorithm: Algorithm,
    problem: &Problem<N>,
    heuristic: H,
) -> Output<usize, [u32; N]> {
    let goal = |v: &usize| problem.goals[*v];
    let (graph, start) = (&problem.graph, problem.start);

    match algorithm {
        Algorithm::Namoa => namoa_star(graph, heuristic, start, goal),
        Algorithm::NamoaDr => namoa_star_dr(graph, heuristic, start, goal),
        Algorithm::Emoa => emoa_star(graph, heuristic, start, goal),
    }
}

/// Checks all three algorithms with the zero and the exact heuristic, and
/// classic NAMOA* also with an admissible, inconsistent heuristic.
fn check_all<const N: usize>(problem: &Problem<N>, zeroed: &[bool]) -> Result<(), TestCaseError> {
    let exact = exact_heuristic(problem);
    let exact = heuristic_fn(|v: &usize| exact[*v]);
    let patchy = patchy_heuristic(problem, zeroed);
    let patchy = heuristic_fn(|v: &usize| patchy[*v]);

    check(problem, &solve(Algorithm::Namoa, problem, ZeroHeuristic))?;
    check(problem, &solve(Algorithm::Namoa, problem, exact))?;
    check(problem, &solve(Algorithm::Namoa, problem, patchy))?;
    check(problem, &solve(Algorithm::NamoaDr, problem, ZeroHeuristic))?;
    check(problem, &solve(Algorithm::NamoaDr, problem, exact))?;
    check(problem, &solve(Algorithm::Emoa, problem, ZeroHeuristic))?;
    check(problem, &solve(Algorithm::Emoa, problem, exact))?;

    Ok(())
}

/// Builds a three-objective problem with a single goal.
fn problem(
    nodes: usize,
    edges: &[(usize, usize, [u32; 3])],
    start: usize,
    goal: usize,
) -> Problem<3> {
    Problem {
        graph: EdgeList {
            nodes,
            edges: edges.to_vec(),
        },
        start,
        goals: (0..nodes).map(|v| v == goal).collect(),
    }
}

/// Every algorithm on `problem` with the zero and the exact heuristic.
fn outputs(problem: &Problem<3>) -> Vec<Output<usize, [u32; 3]>> {
    let exact = exact_heuristic(problem);
    let exact = heuristic_fn(|v: &usize| exact[*v]);

    [Algorithm::Namoa, Algorithm::NamoaDr, Algorithm::Emoa]
        .into_iter()
        .flat_map(|a| [solve(a, problem, ZeroHeuristic), solve(a, problem, exact)])
        .collect()
}

/// A hand-computed three-objective graph from 0 to 3.
///
/// Routes: `0-1-3` and `0-1-2-3` both cost `[2, 3, 4]` (a zero-cost edge
/// makes a tie), the direct edge `[3, 3, 3]`, `0-2-3` costs `[4, 3, 2]`,
/// and a second direct edge `[5, 5, 5]` is dominated by all of them.
#[test]
fn hand_computed() {
    let edges = [
        (0, 1, [1, 2, 3]),
        (1, 3, [1, 1, 1]),
        (0, 2, [3, 2, 1]),
        (2, 3, [1, 1, 1]),
        (1, 2, [0, 0, 0]),
        (0, 3, [3, 3, 3]),
        (0, 3, [5, 5, 5]),
    ];
    let p = problem(4, &edges, 0, 3);

    for out in outputs(&p) {
        let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
        assert_eq!(costs, [[2, 3, 4], [3, 3, 3], [4, 3, 2]]);
        assert_eq!(out.solutions[1].path, [0, 3]);
        assert_eq!(out.solutions[2].path, [0, 2, 3]);
        check(&p, &out).unwrap();
    }
}

/// A start that is a goal yields exactly the empty path at zero cost.
#[test]
fn start_is_goal() {
    let p = problem(2, &[(0, 1, [1, 1, 1]), (1, 0, [0, 0, 0])], 0, 0);

    for out in outputs(&p) {
        assert_eq!(out.solutions.len(), 1);
        assert_eq!(out.solutions[0].cost, [0, 0, 0]);
        assert_eq!(out.solutions[0].path, [0]);
    }
}

/// An unreachable goal yields no solutions.
#[test]
fn unreachable_goal() {
    let p = problem(3, &[(0, 1, [1, 1, 1]), (2, 0, [1, 1, 1])], 0, 2);

    for out in outputs(&p) {
        assert_eq!(out.solutions, []);
    }
}

/// A problem paired with a random subset of nodes to zero the heuristic at.
fn with_zeroed<const N: usize>() -> impl Strategy<Value = (Problem<N>, Vec<bool>)> {
    problems::<N>().prop_flat_map(|p| {
        let nodes = p.graph.nodes;
        (Just(p), prop::collection::vec(any::<bool>(), nodes))
    })
}

proptest! {
    /// Two objectives.
    #[test]
    fn two_objectives((p, zeroed) in with_zeroed::<2>()) {
        check_all(&p, &zeroed)?;
    }

    /// Three objectives: the two-dimensional truncated frontier.
    #[test]
    fn three_objectives((p, zeroed) in with_zeroed::<3>()) {
        check_all(&p, &zeroed)?;
    }

    /// Four objectives: the linear truncated frontier.
    #[test]
    fn four_objectives((p, zeroed) in with_zeroed::<4>()) {
        check_all(&p, &zeroed)?;
    }

    /// For two objectives, all algorithms return the same cost vectors as
    /// BOA* with the exact heuristic.
    #[test]
    fn agrees_with_boa(p in problems::<2>()) {
        let h = exact_heuristic(&p);
        let h = heuristic_fn(|v: &usize| h[*v]);
        let costs = |out: Output<usize, [u32; 2]>| -> Vec<_> {
            out.solutions.into_iter().map(|s| s.cost).collect()
        };
        let expected = costs(boa_star(&p.graph, h, p.start, |v| p.goals[*v]));

        for algorithm in [Algorithm::Namoa, Algorithm::NamoaDr, Algorithm::Emoa] {
            prop_assert_eq!(costs(solve(algorithm, &p, h)), expected.clone());
        }
    }
}
