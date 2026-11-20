//! SIP-MOSP tests: the exact Pareto front whatever the thread timing, on
//! small random graphs against brute force and on larger ones against
//! NAMOA*, for every worker configuration and snapshot index. Each case
//! runs several times.

use multiobjective_search::{
    FrontIndex, Output, SipConfig, ZeroHeuristic, heuristic_fn, namoa_star, sip_mosp,
};
use proptest::prelude::*;

use crate::check::{check, check_paths};
use crate::fixtures::{EdgeList, Problem, exact_heuristic, problems};

/// Worker counts and indexes to run every case with.
const CONFIGS: [SipConfig; 4] = [
    SipConfig {
        search_workers: 1,
        update_workers: 1,
        index: FrontIndex::None,
    },
    SipConfig {
        search_workers: 2,
        update_workers: 3,
        index: FrontIndex::Blocks,
    },
    SipConfig {
        search_workers: 4,
        update_workers: 2,
        index: FrontIndex::Tree,
    },
    SipConfig {
        search_workers: 3,
        update_workers: 1,
        index: FrontIndex::Blocks,
    },
];

/// The cost vectors of an output.
fn costs<const N: usize>(out: &Output<usize, [u32; N]>) -> Vec<[u32; N]> {
    out.solutions.iter().map(|s| s.cost).collect()
}

/// Runs every configuration `runs` times with the zero and the exact
/// heuristic and checks each result against brute force.
fn check_all<const N: usize>(problem: &Problem<N>, runs: usize) -> Result<(), TestCaseError> {
    let h = exact_heuristic(problem);
    let goal = |v: &usize| problem.goals[*v];
    let (graph, start) = (&problem.graph, problem.start);

    for config in CONFIGS {
        for _ in 0..runs {
            check(
                problem,
                &sip_mosp(graph, ZeroHeuristic, start, goal, config),
            )?;
            let exact = heuristic_fn(|v: &usize| h[*v]);
            check(problem, &sip_mosp(graph, exact, start, goal, config))?;
        }
    }

    Ok(())
}

/// A larger random graph: 40 nodes, up to 200 edges with costs up to 30,
/// start 0 and goal 39; big enough for indexed snapshots.
fn larger<const N: usize>() -> impl Strategy<Value = EdgeList<N>> {
    let edge = (0..40_usize, 0..40_usize, prop::array::uniform(0..=30_u32));
    prop::collection::vec(edge, 0..=200).prop_map(|edges| EdgeList { nodes: 40, edges })
}

/// A dense layered graph whose goal frontier has well over 64 vectors, so
/// snapshots are indexed and use deltas.
#[test]
fn large_front_is_exact() {
    // Every edge costs 3 in total, so all paths are mutually
    // non-dominated and the front is every distinct sum.
    let layers = 10;
    let mut edges = Vec::new();
    for layer in 0..layers {
        for cost in [[0, 1, 2], [1, 2, 0], [2, 0, 1], [1, 1, 1], [3, 0, 0]] {
            edges.push((layer, layer + 1, cost));
        }
    }

    let problem = Problem {
        graph: EdgeList {
            nodes: layers + 1,
            edges,
        },
        start: 0,
        goals: (0..=layers).map(|v| v == layers).collect(),
    };
    let goal = |v: &usize| *v == layers;
    let expected = costs(&namoa_star(&problem.graph, ZeroHeuristic, 0, goal));
    assert!(expected.len() > 64);

    for config in CONFIGS {
        for _ in 0..3 {
            let out = sip_mosp(&problem.graph, ZeroHeuristic, 0, goal, config);
            assert_eq!(costs(&out), expected);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Two objectives, three runs per configuration.
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
        check_all(&p, 2)?;
    }

    /// Larger three-objective graphs against NAMOA*.
    #[test]
    fn larger_against_namoa(graph in larger::<3>()) {
        let goal = |v: &usize| *v == 39;
        let expected = costs(&namoa_star(&graph, ZeroHeuristic, 0, goal));
        let problem = Problem { graph, start: 0, goals: (0..40).map(|v| v == 39).collect() };
        for config in CONFIGS {
            for _ in 0..2 {
                let out = sip_mosp(&problem.graph, ZeroHeuristic, 0, goal, config);
                prop_assert_eq!(costs(&out), expected.clone());
                check_paths(&problem, &out)?;
            }
        }
    }
}
