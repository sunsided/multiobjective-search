//! Test graphs, a proptest strategy for them, and an exact heuristic.

use multiobjective_search::Graph;
use proptest::prelude::*;

/// Largest number of nodes in a generated graph. Brute force is exponential.
const MAX_NODES: usize = 7;

/// Largest number of edges in a generated graph.
const MAX_EDGES: usize = 20;

/// Largest cost of one edge in one objective. Small, so that ties and
/// zero-cost cycles are common.
const MAX_COST: u32 = 4;

/// A directed multigraph over nodes `0..nodes` with two-objective costs.
#[derive(Debug, Clone)]
pub struct EdgeList {
    /// Number of nodes. At least 1.
    pub nodes: usize,
    /// Edges as `(from, to, cost)`. Self-loops and parallel edges allowed.
    pub edges: Vec<(usize, usize, [u32; 2])>,
}

impl Graph for EdgeList {
    type State = usize;
    type Weight = [u32; 2];

    fn successors(&self, state: &usize, out: &mut Vec<(usize, [u32; 2])>) {
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
pub struct Problem {
    /// The graph.
    pub graph: EdgeList,
    /// Start node, in `0..graph.nodes`.
    pub start: usize,
    /// `goals[v]` is `true` if node `v` is a goal. Length `graph.nodes`.
    pub goals: Vec<bool>,
}

/// Generates small random problems with one or more goals (possibly none
/// reachable, possibly the start).
pub fn problems() -> impl Strategy<Value = Problem> {
    (1..=MAX_NODES).prop_flat_map(|nodes| {
        let edge = (0..nodes, 0..nodes, [0..=MAX_COST, 0..=MAX_COST]);
        (
            prop::collection::vec(edge, 0..=MAX_EDGES),
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

/// Exact remaining cost per objective, computed independently for each
/// objective by backward Bellman-Ford from the goal set.
///
/// Entry `v` is `None` if no goal is reachable from `v`. Exact distances
/// are consistent and zero on goals, as BOA* requires.
pub fn exact_heuristic(problem: &Problem) -> Vec<Option<[u32; 2]>> {
    let first = backward_distances(problem, 0);
    let second = backward_distances(problem, 1);

    first
        .into_iter()
        .zip(second)
        .map(|(a, b)| Some([a?, b?]))
        .collect()
}

/// Shortest distance from every node to the goal set in one objective.
fn backward_distances(problem: &Problem, objective: usize) -> Vec<Option<u32>> {
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
