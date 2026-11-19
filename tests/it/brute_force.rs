//! Brute-force reference: enumerate every simple path, keep the
//! non-dominated costs.
//!
//! With non-negative costs, every Pareto-optimal cost vector is the cost of
//! some simple path (removing a cycle never increases a cost), so this is
//! exact. It deliberately avoids the crate's dominance code.

use crate::fixtures::Problem;

/// The Pareto-optimal cost vectors from start to any goal, sorted
/// ascending, without duplicates.
pub fn pareto_costs(problem: &Problem) -> Vec<[u32; 2]> {
    let mut costs = Vec::new();
    let mut visited = vec![false; problem.graph.nodes];
    collect_path_costs(problem, problem.start, [0, 0], &mut visited, &mut costs);

    non_dominated(&costs)
}

/// Depth-first enumeration of simple paths from `node`, pushing the cost of
/// every path that ends in a goal.
fn collect_path_costs(
    problem: &Problem,
    node: usize,
    cost: [u32; 2],
    visited: &mut [bool],
    out: &mut Vec<[u32; 2]>,
) {
    if problem.goals[node] {
        out.push(cost);
    }

    visited[node] = true;

    for &(from, to, edge) in &problem.graph.edges {
        if from == node && !visited[to] {
            let next = [cost[0] + edge[0], cost[1] + edge[1]];
            collect_path_costs(problem, to, next, visited, out);
        }
    }

    visited[node] = false;
}

/// Naive `O(n²)` filter: keeps each vector that no other vector beats in
/// one objective while matching or beating it in the other.
pub fn non_dominated(costs: &[[u32; 2]]) -> Vec<[u32; 2]> {
    let mut kept: Vec<[u32; 2]> = costs
        .iter()
        .copied()
        .filter(|c| !costs.iter().any(|o| o[0] <= c[0] && o[1] <= c[1] && o != c))
        .collect();

    kept.sort_unstable();
    kept.dedup();

    kept
}
