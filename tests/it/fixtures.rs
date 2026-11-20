//! Test graphs, proptest strategies for them, and exact heuristics.

use multiobjective_search::Graph;
use proptest::prelude::*;

/// Largest number of nodes in a generated graph. Brute force is exponential.
const MAX_NODES: usize = 7;

/// Largest number of edges in a generated graph.
const MAX_EDGES: usize = 20;

/// Largest cost of one edge in one objective. Small, so that ties and
/// zero-cost cycles are common.
const MAX_COST: u32 = 4;

/// A directed multigraph over nodes `0..nodes` with `N`-objective costs.
#[derive(Debug, Clone)]
pub struct EdgeList<const N: usize> {
    /// Number of nodes. At least 1.
    pub nodes: usize,
    /// Edges as `(from, to, cost)`. Self-loops and parallel edges allowed.
    pub edges: Vec<(usize, usize, [u32; N])>,
}

impl<const N: usize> Graph for EdgeList<N> {
    type State = usize;
    type Weight = [u32; N];

    fn successors(&self, state: &usize, out: &mut Vec<(usize, [u32; N])>) {
        out.extend(
            self.edges
                .iter()
                .filter(|(from, _, _)| from == state)
                .map(|&(_, to, cost)| (to, cost)),
        );
    }
}

/// A search problem: graph, start node and goal set.
#[derive(Debug, Clone)]
pub struct Problem<const N: usize> {
    /// The graph.
    pub graph: EdgeList<N>,
    /// Start node, in `0..graph.nodes`.
    pub start: usize,
    /// `goals[v]` is `true` if node `v` is a goal. Length `graph.nodes`.
    pub goals: Vec<bool>,
}

/// Generates small random problems with one or more goals (possibly none
/// reachable, possibly the start).
pub fn problems<const N: usize>() -> impl Strategy<Value = Problem<N>> {
    (1..=MAX_NODES).prop_flat_map(|nodes| {
        (
            edges::<N>(nodes),
            0..nodes,
            prop::collection::vec(prop::bool::weighted(0.3), nodes),
        )
            .prop_map(move |(edges, start, goals)| Problem {
                graph: EdgeList { nodes, edges },
                start,
                goals,
            })
    })
}

/// Random edge lists over `nodes` nodes with costs in `0..=MAX_COST`.
fn edges<const N: usize>(nodes: usize) -> impl Strategy<Value = Vec<(usize, usize, [u32; N])>> {
    let edge = (0..nodes, 0..nodes, prop::array::uniform(0..=MAX_COST));
    prop::collection::vec(edge, 0..=MAX_EDGES)
}

/// Exact remaining cost per objective, computed independently for each
/// objective by backward Bellman-Ford from the goal set.
///
/// Entry `v` is `None` if no goal is reachable from `v`. Exact distances
/// are consistent and zero on goals.
pub fn exact_heuristic<const N: usize>(problem: &Problem<N>) -> Vec<Option<[u32; N]>> {
    let per_objective: Vec<_> = (0..N).map(|i| backward_distances(problem, i)).collect();

    (0..problem.graph.nodes)
        .map(|v| {
            let mut h = [0; N];
            for (i, dist) in per_objective.iter().enumerate() {
                h[i] = dist[v]?;
            }

            Some(h)
        })
        .collect()
}

/// Shortest distance from every node to the goal set in one objective.
fn backward_distances<const N: usize>(problem: &Problem<N>, objective: usize) -> Vec<Option<u32>> {
    let mut dist: Vec<Option<u32>> = problem
        .goals
        .iter()
        .map(|&goal| goal.then_some(0))
        .collect();

    for _ in 0..problem.graph.nodes {
        for &(from, to, cost) in &problem.graph.edges {
            let Some(d) = dist[to] else { continue };
            let candidate = d + cost[objective];
            if dist[from].is_none_or(|current| candidate < current) {
                dist[from] = Some(candidate);
            }
        }
    }

    dist
}
