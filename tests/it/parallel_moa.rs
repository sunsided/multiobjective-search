//! Parallel MOA* tests: the exact front whatever the thread timing, each
//! case run several times, against brute force and against NAMOA*.

use multiobjective_search::{Output, ZeroHeuristic, heuristic_fn, namoa_star, parallel_moa_star};
use proptest::prelude::*;

use crate::check::{check, check_paths};
use crate::fixtures::{EdgeList, Problem, exact_heuristic, problems};

/// The cost vectors of an output.
fn costs<const N: usize>(out: &Output<usize, [u32; N]>) -> Vec<[u32; N]> {
    out.solutions.iter().map(|s| s.cost).collect()
}

/// Runs parallel MOA* `runs` times with the zero and the exact heuristic.
fn check_all<const N: usize>(problem: &Problem<N>, runs: usize) -> Result<(), TestCaseError> {
    let h = exact_heuristic(problem);
    let goal = |v: &usize| problem.goals[*v];
    for _ in 0..runs {
        check(
            problem,
            &parallel_moa_star(&problem.graph, ZeroHeuristic, problem.start, goal),
        )?;
        check(
            problem,
            &parallel_moa_star(
                &problem.graph,
                heuristic_fn(|v: &usize| h[*v]),
                problem.start,
                goal,
            ),
        )?;
    }

    Ok(())
}

/// A larger random graph: 40 nodes, up to 200 edges, costs up to 30.
fn larger<const N: usize>() -> impl Strategy<Value = EdgeList<N>> {
    let edge = (0..40_usize, 0..40_usize, prop::array::uniform(0..=30_u32));
    prop::collection::vec(edge, 0..=200).prop_map(|edges| EdgeList { nodes: 40, edges })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Two objectives.
    #[test]
    fn two_objectives(p in problems::<2>()) {
        check_all(&p, 3)?;
    }

    /// Three objectives.
    #[test]
    fn three_objectives(p in problems::<3>()) {
        check_all(&p, 3)?;
    }

    /// Four objectives.
    #[test]
    fn four_objectives(p in problems::<4>()) {
        check_all(&p, 3)?;
    }

    /// Larger three- and four-objective graphs against NAMOA*.
    #[test]
    fn larger_against_namoa(g3 in larger::<3>(), g4 in larger::<4>()) {
        let goal = |v: &usize| *v == 39;
        for _ in 0..3 {
            let expected = costs(&namoa_star(&g3, ZeroHeuristic, 0, goal));
            let out = parallel_moa_star(&g3, ZeroHeuristic, 0, goal);
            prop_assert_eq!(costs(&out), expected);
            check_paths(&Problem { graph: g3.clone(), start: 0, goals: (0..40).map(|v| v == 39).collect() }, &out)?;
            let expected = costs(&namoa_star(&g4, ZeroHeuristic, 0, goal));
            prop_assert_eq!(costs(&parallel_moa_star(&g4, ZeroHeuristic, 0, goal)), expected);
        }
    }
}
