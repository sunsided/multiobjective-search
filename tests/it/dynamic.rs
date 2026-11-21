//! Dynamic search tests: after every edge change, the repaired front
//! equals a fresh NAMOA* on the current graph, and every path is real.

use multiobjective_search::{DynamicGraph, DynamicSearch, Output, ZeroHeuristic, namoa_star};
use proptest::prelude::*;

use crate::check::check_paths;
use crate::fixtures::{EdgeList, Problem, problems};

/// One edge change: re-cost, remove or add.
#[derive(Debug, Clone)]
enum Change<const N: usize> {
    /// New cost for the edge at this position (modulo the edge count).
    Cost(usize, [u32; N]),
    /// Remove the edge at this position.
    Remove(usize),
    /// Add an edge.
    Add(usize, usize, [u32; N]),
}

/// A strategy for changes over up to seven nodes.
fn changes<const N: usize>() -> impl Strategy<Value = Vec<Change<N>>> {
    let cost = prop::array::uniform(0..=4_u32);
    let change = prop_oneof![
        (any::<usize>(), cost.clone()).prop_map(|(e, c)| Change::Cost(e, c)),
        any::<usize>().prop_map(Change::Remove),
        (0..7_usize, 0..7_usize, cost).prop_map(|(a, b, c)| Change::Add(a, b, c)),
    ];
    prop::collection::vec(change, 0..8)
}

/// The live edges of `graph` as an edge list.
fn snapshot<const N: usize>(graph: &DynamicGraph<u32, N>, ids: &[usize]) -> EdgeList<N> {
    let edges = ids
        .iter()
        .filter_map(|&e| graph.edge(e))
        .map(|(a, b, c)| (a as usize, b as usize, c))
        .collect();
    EdgeList {
        nodes: graph.node_count(),
        edges,
    }
}

/// `x` as a state.
fn small(x: usize) -> u32 {
    u32::try_from(x).expect("small graphs")
}

/// Checks that the repaired front of `search` equals a fresh NAMOA* on its
/// current graph, and that every path is real; `ids` are all edge ids ever
/// added, live or removed.
fn compare<P, const N: usize>(
    problem: &Problem<N>,
    search: &DynamicSearch<u32, P, N>,
    ids: &[usize],
) -> Result<(), TestCaseError>
where
    P: Fn(&u32) -> bool,
{
    let fresh = namoa_star(
        search.graph(),
        ZeroHeuristic,
        small(problem.start),
        |s: &u32| problem.goals[*s as usize],
    );
    let got = search.front();
    let costs = |o: &Output<u32, [u32; N]>| o.solutions.iter().map(|s| s.cost).collect::<Vec<_>>();
    prop_assert_eq!(costs(&got), costs(&fresh));

    let current = Problem {
        graph: snapshot(search.graph(), ids),
        start: problem.start,
        goals: problem.goals.clone(),
    };
    let solutions = got
        .solutions
        .iter()
        .map(|s| multiobjective_search::Solution {
            cost: s.cost,
            path: s.path.iter().map(|&v| v as usize).collect(),
        })
        .collect();

    check_paths(
        &current,
        &Output {
            solutions,
            stats: got.stats,
        },
    )
}

/// Applies `changes` one by one and checks the front after each.
fn check<const N: usize>(problem: &Problem<N>, changes: &[Change<N>]) -> Result<(), TestCaseError> {
    let mut graph = DynamicGraph::new(small(problem.graph.nodes));
    let mut ids: Vec<usize> = problem
        .graph
        .edges
        .iter()
        .map(|&(a, b, c)| graph.add_edge(small(a), small(b), c))
        .collect();
    let goals = problem.goals.clone();
    let mut search = DynamicSearch::new(graph, small(problem.start), move |s: &u32| {
        goals[*s as usize]
    });

    compare(problem, &search, &ids)?;
    for change in changes {
        let live: Vec<usize> = ids
            .iter()
            .copied()
            .filter(|&e| search.graph().edge(e).is_some())
            .collect();
        match *change {
            Change::Cost(k, c) if !live.is_empty() => search.set_cost(live[k % live.len()], c),
            Change::Remove(k) if !live.is_empty() => search.remove_edge(live[k % live.len()]),
            Change::Add(a, b, c) => {
                let n = problem.graph.nodes;
                ids.push(search.add_edge(small(a % n), small(b % n), c));
            }
            _ => continue,
        }

        compare(problem, &search, &ids)?;
    }

    Ok(())
}

proptest! {
    /// Two and three objectives, random change sequences.
    #[test]
    fn repairs_match_fresh_searches(p2 in problems::<2>(), c2 in changes::<2>(), p3 in problems::<3>(), c3 in changes::<3>()) {
        check(&p2, &c2)?;
        check(&p3, &c3)?;
    }
}
