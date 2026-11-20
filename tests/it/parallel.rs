//! Multi-threaded BOBA* tests: the same front as the sequential version,
//! on small random graphs against brute force and on larger random graphs
//! against BOA*. Each case runs several times, since thread timing varies.

use multiobjective_search::{
    Output, ZeroHeuristic, boa_star, boba_star, boba_star_parallel, boba_star_with_parallel,
};
use proptest::prelude::*;

use crate::check::{check, check_paths};
use crate::fixtures::{EdgeList, Problem, single_goal_problems};

/// The cost vectors of an output.
fn costs(out: &Output<usize, [u32; 2]>) -> Vec<[u32; 2]> {
    out.solutions.iter().map(|s| s.cost).collect()
}

/// A strategy for larger random graphs: 30 nodes, up to 120 edges with
/// costs up to 20, start 0 and goal 29.
fn larger() -> impl Strategy<Value = EdgeList<2>> {
    let edge = (0..30_usize, 0..30_usize, prop::array::uniform(0..=20_u32));
    prop::collection::vec(edge, 0..=120).prop_map(|edges| EdgeList { nodes: 30, edges })
}

proptest! {
    /// Both parallel entry points match the brute-force front and the
    /// sequential BOBA*, with valid paths.
    #[test]
    fn matches_sequential(p in single_goal_problems::<2>()) {
        let goal = p.goals.iter().position(|&g| g).expect("one goal");
        let sequential = costs(&boba_star(&p.graph, p.start, goal));

        for _ in 0..4 {
            let parallel = boba_star_parallel(&p.graph, p.start, goal);
            check(&p, &parallel)?;
            prop_assert_eq!(&costs(&parallel), &sequential);

            let blind = boba_star_with_parallel(&p.graph, p.start, goal, ZeroHeuristic, ZeroHeuristic);
            check(&p, &blind)?;
        }
    }

    /// On larger graphs the parallel front equals BOA*'s.
    #[test]
    fn matches_boa_on_larger_graphs(graph in larger()) {
        let expected = costs(&boa_star(&graph, ZeroHeuristic, 0, |v| *v == 29));
        let problem = Problem { goals: (0..30).map(|v| v == 29).collect(), start: 0, graph };

        for _ in 0..4 {
            let out = boba_star_parallel(&problem.graph, 0, 29);
            prop_assert_eq!(&costs(&out), &expected);
            check_paths(&problem, &out)?;
        }
    }
}
