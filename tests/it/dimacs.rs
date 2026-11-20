//! DIMACS loader tests: round trips through the text format, and searches
//! on the loaded graph against the brute-force reference.

use std::fmt::Write;

use multiobjective_search::{RoadGraph, ZeroHeuristic, boa_star, boba_star};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::fixtures::{Problem, single_goal_problems};

/// Writes `problem`'s graph as a distance and a time `.gr` file, with node
/// ids shifted to start at 1.
fn to_dimacs(problem: &Problem<2>) -> (String, String) {
    let graph = &problem.graph;
    let mut files = [String::new(), String::new()];

    for (objective, file) in files.iter_mut().enumerate() {
        writeln!(file, "c objective {objective}").unwrap();
        writeln!(file, "p sp {} {}", graph.nodes, graph.edges.len()).unwrap();
        for &(from, to, cost) in &graph.edges {
            writeln!(file, "a {} {} {}", from + 1, to + 1, cost[objective]).unwrap();
        }
    }

    let [distance, time] = files;
    (distance, time)
}

proptest! {
    /// Parsing the written files gives the graph built directly from the
    /// arcs, and BOA* and BOBA* on it find the brute-force Pareto front.
    #[test]
    fn round_trip_and_search(p in single_goal_problems::<2>()) {
        let (distance, time) = to_dimacs(&p);
        let graph = RoadGraph::from_dimacs(&distance, &time).expect("valid files");

        let nodes = u32::try_from(p.graph.nodes).unwrap();
        let arcs: Vec<_> = p
            .graph
            .edges
            .iter()
            .map(|&(a, b, c)| (u32::try_from(a + 1).unwrap(), u32::try_from(b + 1).unwrap(), c))
            .collect();
        prop_assert_eq!(&graph, &RoadGraph::from_arcs(nodes, &arcs));

        let start = u32::try_from(p.start + 1).unwrap();
        let goal = u32::try_from(p.goals.iter().position(|&g| g).unwrap() + 1).unwrap();
        let expected = pareto_costs(&p);
        let costs = |out: multiobjective_search::Output<u32, [u32; 2]>| -> Vec<_> {
            out.solutions.into_iter().map(|s| s.cost).collect()
        };

        prop_assert_eq!(&costs(boa_star(&graph, ZeroHeuristic, start, |s| *s == goal)), &expected);
        prop_assert_eq!(&costs(boba_star(&graph, start, goal)), &expected);
    }
}
