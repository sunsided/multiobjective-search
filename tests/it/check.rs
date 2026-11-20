//! Output checks shared by every algorithm's tests.

use multiobjective_search::Output;
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::fixtures::{EdgeList, Problem};

/// Checks an output against the brute-force reference: the cost list
/// equals the reference (so it is lexicographically sorted, Pareto-optimal
/// and complete), and each path is a real start-to-goal path whose cost
/// matches.
pub fn check<const N: usize>(
    problem: &Problem<N>,
    out: &Output<usize, [u32; N]>,
) -> Result<(), TestCaseError> {
    let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
    prop_assert_eq!(&costs, &pareto_costs(problem));
    check_paths(problem, out)
}

/// Checks that every solution path starts at the start, ends in a goal,
/// uses only real edges and can have the reported cost.
pub fn check_paths<const N: usize>(
    problem: &Problem<N>,
    out: &Output<usize, [u32; N]>,
) -> Result<(), TestCaseError> {
    for solution in &out.solutions {
        prop_assert_eq!(solution.path.first(), Some(&problem.start));
        prop_assert!(problem.goals[*solution.path.last().unwrap()]);
        prop_assert!(path_costs(&problem.graph, &solution.path).contains(&solution.cost));
    }

    Ok(())
}

/// Every cost vector a path can have, over all choices among parallel
/// edges. Empty if some step has no edge.
fn path_costs<const N: usize>(graph: &EdgeList<N>, path: &[usize]) -> Vec<[u32; N]> {
    let mut costs = vec![[0; N]];

    for step in path.windows(2) {
        let edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|(from, to, _)| *from == step[0] && *to == step[1])
            .map(|&(_, _, cost)| cost)
            .collect();

        costs = costs
            .iter()
            .flat_map(|c| {
                edges
                    .iter()
                    .map(move |e| core::array::from_fn(|i| c[i] + e[i]))
            })
            .collect();
    }

    costs
}
