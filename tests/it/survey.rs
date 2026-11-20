//! Tests for the problem classes taken from the Craveirinha et al. survey:
//! minsum–maxmin paths, constrained shortest paths and the ranking method,
//! each against exhaustive enumeration of simple paths.

use multiobjective_search::{
    Graph, Stats, ZeroHeuristic, constrained_shortest_path, heuristic_fn, minsum_maxmin, ranking,
};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::check::check;
use crate::fixtures::{EdgeList, Problem, exact_heuristic, problems};

/// Every simple path from `node` to a goal, as the list of its edge costs.
fn simple_paths(
    problem: &Problem<2>,
    node: usize,
    visited: &mut Vec<bool>,
    edges: &mut Vec<[u32; 2]>,
    out: &mut Vec<Vec<[u32; 2]>>,
) {
    if problem.goals[node] {
        out.push(edges.clone());
    }

    visited[node] = true;
    for &(from, to, cost) in &problem.graph.edges {
        if from == node && !visited[to] {
            edges.push(cost);
            simple_paths(problem, to, visited, edges, out);
            edges.pop();
        }
    }

    visited[node] = false;
}

/// All simple start–goal paths of `problem`.
fn all_paths(problem: &Problem<2>) -> Vec<Vec<[u32; 2]>> {
    let mut out = Vec::new();
    simple_paths(
        problem,
        problem.start,
        &mut vec![false; problem.graph.nodes],
        &mut Vec::new(),
        &mut out,
    );
    out
}

/// The graph of `problem` with `(cost, capacity)` edges.
struct Capacities<'a>(&'a EdgeList<2>);

impl Graph for Capacities<'_> {
    type State = usize;
    type Weight = (u32, u32);

    fn successors(&self, state: &usize, out: &mut Vec<(usize, (u32, u32))>) {
        let mut edges = Vec::new();
        self.0.successors(state, &mut edges);
        out.extend(edges.into_iter().map(|(t, c)| (t, (c[0], c[1]))));
    }
}

/// `a` is at least as good as `b`: no larger cost, no smaller bottleneck
/// (`None` is unbounded).
fn width_dominates(a: (u32, Option<u32>), b: (u32, Option<u32>)) -> bool {
    let wider = match (a.1, b.1) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(x), Some(y)) => x >= y,
    };

    a.0 <= b.0 && wider
}

proptest! {
    /// `minsum_maxmin` returns exactly the non-dominated (sum, bottleneck)
    /// pairs, by cost ascending.
    #[test]
    fn minsum_maxmin_matches_enumeration(p in problems::<2>()) {
        let pairs: Vec<(u32, Option<u32>)> = all_paths(&p)
            .iter()
            .map(|edges| (edges.iter().map(|c| c[0]).sum(), edges.iter().map(|c| c[1]).min()))
            .collect();
        let mut front: Vec<_> = pairs
            .iter()
            .copied()
            .filter(|&a| !pairs.iter().any(|&b| b != a && width_dominates(b, a)))
            .collect();
        front.sort_by_key(|&(c, _)| c);
        front.dedup();

        let goal = |v: &usize| p.goals[*v];
        let out = minsum_maxmin(&Capacities(&p.graph), heuristic_fn(|_: &usize| Some(0_u32)), p.start, goal);
        let found: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
        prop_assert_eq!(&found, &front);

        for s in &out.solutions {
            prop_assert_eq!(s.path.first(), Some(&p.start));
            prop_assert!(p.goals[*s.path.last().unwrap()]);
        }
    }

    /// `constrained_shortest_path` is the lexicographic optimum among paths
    /// within the budget, for every budget.
    #[test]
    fn constrained_matches_enumeration(p in problems::<2>(), budget in 0..=12_u32) {
        let expected = all_paths(&p)
            .iter()
            .map(|edges| edges.iter().fold([0, 0], |s, c| [s[0] + c[0], s[1] + c[1]]))
            .filter(|c| c[1] <= budget)
            .min();
        let h = exact_heuristic(&p);
        let goal = |v: &usize| p.goals[*v];
        let mut stats = Stats::default();

        let zero = constrained_shortest_path(&p.graph, ZeroHeuristic, p.start, goal, budget, &mut stats);
        prop_assert_eq!(zero.as_ref().map(|s| s.cost), expected);
        let exact = heuristic_fn(|v: &usize| h[*v]);
        let informed = constrained_shortest_path(&p.graph, exact, p.start, goal, budget, &mut stats);
        prop_assert_eq!(informed.map(|s| s.cost), expected);
    }

    /// `ranking` returns the Pareto front.
    #[test]
    fn ranking_matches_brute_force(p in problems::<2>()) {
        let h = exact_heuristic(&p);
        let goal = |v: &usize| p.goals[*v];
        check(&p, &ranking(&p.graph, ZeroHeuristic, p.start, goal))?;
        check(&p, &ranking(&p.graph, heuristic_fn(|v: &usize| h[*v]), p.start, goal))?;
        prop_assert!(!pareto_costs(&p).is_empty() || ranking(&p.graph, ZeroHeuristic, p.start, goal).solutions.is_empty());
    }
}
