//! Benchmark-suite generator tests: the paper's instance definitions, and
//! the exact searches against brute force on small generated instances.

use multiobjective_search::{
    BenchmarkGraph, Graph, ReverseGraph, ZeroHeuristic, boa_star, guards_grid, namoa_star_dr,
    netmaker, netmaker_queries, random_grid, road_extended,
};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::fixtures::{EdgeList, Problem};

/// A generated graph as a brute-force problem.
fn problem<const N: usize>(graph: &BenchmarkGraph<N>, start: u32, goal: u32) -> Problem<N> {
    let edges = graph
        .arcs()
        .map(|(u, v, c)| (u as usize, v as usize, c))
        .collect();
    Problem {
        graph: EdgeList {
            nodes: graph.node_count() as usize,
            edges,
        },
        start: start as usize,
        goals: (0..graph.node_count()).map(|v| v == goal).collect(),
    }
}

/// States reachable from `from`, forwards or backwards.
fn reach<const N: usize>(graph: &BenchmarkGraph<N>, from: u32, backwards: bool) -> usize {
    let mut seen = vec![false; graph.node_count() as usize];
    let mut stack = vec![from];
    seen[from as usize] = true;
    let mut edges = Vec::new();
    while let Some(u) = stack.pop() {
        if backwards {
            graph.predecessors(&u, &mut edges);
        } else {
            graph.successors(&u, &mut edges);
        }

        for (v, _) in edges.drain(..) {
            if !seen[v as usize] {
                seen[v as usize] = true;
                stack.push(v);
            }
        }
    }
    seen.iter().filter(|&&s| s).count()
}

proptest! {
    /// Random grids: shape, cost range, determinism, and exact fronts.
    #[test]
    fn random_grids(w in 1..4_u32, h in 1..3_u32, seed: u64) {
        let (grid, (s, t)) = random_grid::<2>(w, h, seed);
        prop_assert_eq!(grid.node_count(), w * h + 2);
        prop_assert_eq!(grid.arc_count(), (2 * (w - 1) * h + 2 * w * (h - 1) + 2 * h) as usize);
        prop_assert!(grid.arcs().all(|(_, _, c)| c.iter().all(|&x| (1..=10).contains(&x))));
        prop_assert_eq!(&random_grid::<2>(w, h, seed).0, &grid);

        let costs: Vec<_> = boa_star(&grid, ZeroHeuristic, s, |v| *v == t).solutions.iter().map(|x| x.cost).collect();
        prop_assert_eq!(costs, pareto_costs(&problem(&grid, s, t)));
        let (grid3, (s, t)) = random_grid::<3>(w, h, seed);
        let costs: Vec<_> = namoa_star_dr(&grid3, ZeroHeuristic, s, |v| *v == t).solutions.iter().map(|x| x.cost).collect();
        prop_assert_eq!(costs, pareto_costs(&problem(&grid3, s, t)));
    }

    /// NetMaker: strongly connected, out-degrees and cost bands as defined,
    /// deterministic; queries in the first and last tenth.
    #[test]
    fn netmaker_instances(n in 10..60_u32, locality in 2..12_u32, seed: u64) {
        let graph = netmaker(n, 1, 6, locality, seed);
        prop_assert_eq!(&netmaker(n, 1, 6, locality, seed), &graph);
        prop_assert_eq!(reach(&graph, 0, false), n as usize);
        prop_assert_eq!(reach(&graph, 0, true), n as usize);

        let mut edges = Vec::new();
        for u in 0..n {
            graph.successors(&u, &mut edges);
            prop_assert!(!edges.is_empty() && edges.len() <= 6);
            let banded = |c: [u32; 3]| {
                let mut c = c;
                c.sort_unstable();
                c[0] <= 333 && (334..=666).contains(&c[1]) && c[2] >= 667
            };
            prop_assert_eq!(edges.iter().filter(|e| banded(e.1)).count(), 1);
            for &(v, c) in edges.iter().filter(|e| !banded(e.1)) {
                prop_assert!(v != u && u.abs_diff(v) <= locality / 2 && c.iter().all(|&x| (1..=99).contains(&x)));
            }
            let mut targets: Vec<_> = edges.iter().map(|e| e.0).collect();
            targets.sort_unstable();
            targets.dedup();
            prop_assert_eq!(targets.len(), edges.len());
            edges.clear();
        }

        for (s, t) in netmaker_queries(n, 5, seed) {
            prop_assert!(s < n / 10 && t >= n - n / 10 && t < n);
        }
    }
}

/// A guard sees along open lines only; exposure costs follow the move
/// type, and diagonal moves need both side cells.
#[test]
fn guards_grid_costs() {
    let map = "type octile\nheight 3\nwidth 3\nmap\n...\n.@.\n...\n";
    let graph = guards_grid(map, &[(0, 0)]).unwrap();
    let mut edges = Vec::new();

    // From (0,0) right to (1,0): seen, cost 10 and exposure 1.
    graph.successors(&0, &mut edges);
    assert!(edges.contains(&(1, [10, 1])));
    // No diagonal through the wall at (1,1).
    assert!(edges.iter().all(|e| e.0 != 4));
    edges.clear();

    // (2,2) is hidden behind the wall: moving there costs no exposure.
    graph.successors(&5, &mut edges);
    assert!(edges.contains(&(8, [10, 0])));
    // The blocked cell has no edges.
    edges.clear();
    graph.successors(&4, &mut edges);
    assert_eq!(edges, []);
    assert!(guards_grid("height 1\nwidth 2\nmap\n.\n", &[]).is_none());
}

/// DIMACS-Extended: elevation difference, summed out-degrees and hops.
#[test]
fn road_extension() {
    let arcs = [(0, 1, [5, 7]), (1, 0, [5, 7]), (1, 2, [3, 4])];
    let graph = road_extended(3, &arcs, &[10, 4, 9]);
    let all: Vec<_> = graph.arcs().collect();
    assert_eq!(
        all,
        [
            (0, 1, [5, 7, 6, 3, 1]),
            (1, 0, [5, 7, 6, 3, 1]),
            (1, 2, [3, 4, 5, 2, 1])
        ]
    );
    let three: Vec<_> = graph.project([0, 1, 2]).arcs().map(|a| a.2).collect();
    assert_eq!(three[0], [5, 7, 6]);
}
