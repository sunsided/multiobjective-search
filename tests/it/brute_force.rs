//! Brute-force reference: enumerate every simple path, keep the
//! non-dominated costs.
//!
//! With non-negative costs, every Pareto-optimal cost vector is the cost of
//! some simple path (removing a cycle never increases a cost), so this is
//! exact. It deliberately avoids the crate's dominance code.

use crate::fixtures::Problem;

/// The Pareto-optimal cost vectors from start to any goal, sorted
/// lexicographically ascending, without duplicates.
pub fn pareto_costs<const N: usize>(problem: &Problem<N>) -> Vec<[u32; N]> {
    let mut costs = Vec::new();
    let mut visited = vec![false; problem.graph.nodes];
    collect_path_costs(problem, problem.start, [0; N], &mut visited, &mut costs);

    non_dominated(&costs)
}

/// Depth-first enumeration of simple paths from `node`, pushing the cost of
/// every path that ends in a goal.
fn collect_path_costs<const N: usize>(
    problem: &Problem<N>,
    node: usize,
    cost: [u32; N],
    visited: &mut [bool],
    out: &mut Vec<[u32; N]>,
) {
    if problem.goals[node] {
        out.push(cost);
    }

    visited[node] = true;

    for &(from, to, edge) in &problem.graph.edges {
        if from == node && !visited[to] {
            let next = core::array::from_fn(|i| cost[i] + edge[i]);
            collect_path_costs(problem, to, next, visited, out);
        }
    }

    visited[node] = false;
}

/// Naive `O(n²)` filter: keeps each vector that no other, different vector
/// matches or beats in every objective. Sorted, without duplicates.
pub fn non_dominated<const N: usize>(costs: &[[u32; N]]) -> Vec<[u32; N]> {
    let mut kept: Vec<[u32; N]> = costs
        .iter()
        .copied()
        .filter(|c| !costs.iter().any(|o| o != c && (0..N).all(|i| o[i] <= c[i])))
        .collect();

    kept.sort_unstable();
    kept.dedup();

    kept
}
