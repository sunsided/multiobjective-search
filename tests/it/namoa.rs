//! NAMOA*, NAMOA*dr and EMOA* tests: property-based cross-checks against
//! the brute-force reference for two, three and four objectives, and
//! against BOA* for two.

use multiobjective_search::{
    Heuristic, Output, ZeroHeuristic, boa_star, emoa_star, namoa_star, namoa_star_dr,
};
use proptest::prelude::*;

use crate::check::check;
use crate::fixtures::{Problem, exact_heuristic, patchy_heuristic, problems};

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
    heuristic: &H,
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
    let exact = |v: &usize| exact[*v];
    let patchy = patchy_heuristic(problem, zeroed);
    let patchy = |v: &usize| patchy[*v];

    check(problem, &solve(Algorithm::Namoa, problem, &ZeroHeuristic))?;
    check(problem, &solve(Algorithm::Namoa, problem, &exact))?;
    check(problem, &solve(Algorithm::Namoa, problem, &patchy))?;
    check(problem, &solve(Algorithm::NamoaDr, problem, &ZeroHeuristic))?;
    check(problem, &solve(Algorithm::NamoaDr, problem, &exact))?;
    check(problem, &solve(Algorithm::Emoa, problem, &ZeroHeuristic))?;
    check(problem, &solve(Algorithm::Emoa, problem, &exact))?;

    Ok(())
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
        let h = |v: &usize| h[*v];
        let costs = |out: Output<usize, [u32; 2]>| -> Vec<_> {
            out.solutions.into_iter().map(|s| s.cost).collect()
        };
        let expected = costs(boa_star(&p.graph, &h, p.start, |v| p.goals[*v]));

        for algorithm in [Algorithm::Namoa, Algorithm::NamoaDr, Algorithm::Emoa] {
            prop_assert_eq!(costs(solve(algorithm, &p, &h)), expected.clone());
        }
    }
}
