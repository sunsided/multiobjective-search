//! Dynamic search tests: after every edge change, the repaired front
//! equals a fresh NAMOA* on the current graph, and every path is real,
//! with and without a heuristic.

use multiobjective_search::{
    DynamicGraph, DynamicSearch, Heuristic, IdealPoint, Output, ZeroHeuristic, heuristic_fn,
    namoa_star,
};
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
fn compare<P, H, const N: usize>(
    problem: &Problem<N>,
    search: &DynamicSearch<u32, P, N, H>,
    ids: &[usize],
) -> Result<(), TestCaseError>
where
    P: Fn(&u32) -> bool,
    H: Heuristic<u32, [u32; N]>,
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

/// Applies `changes` one by one to a search guided by `heuristic` and
/// checks the front after each; returns the graph after every change, the
/// first included.
fn check<H, const N: usize>(
    problem: &Problem<N>,
    changes: &[Change<N>],
    heuristic: H,
) -> Result<Vec<DynamicGraph<u32, N>>, TestCaseError>
where
    H: Heuristic<u32, [u32; N]>,
{
    let mut graph = DynamicGraph::new(small(problem.graph.nodes));
    let mut ids: Vec<usize> = problem
        .graph
        .edges
        .iter()
        .map(|&(a, b, c)| graph.add_edge(small(a), small(b), c))
        .collect();
    let goals = problem.goals.clone();
    let mut search =
        DynamicSearch::with_heuristic(graph, heuristic, small(problem.start), move |s: &u32| {
            goals[*s as usize]
        });

    compare(problem, &search, &ids)?;
    let mut graphs = vec![search.graph().clone()];
    for change in changes {
        if apply(&mut search, &mut ids, problem.graph.nodes, change) {
            compare(problem, &search, &ids)?;
            graphs.push(search.graph().clone());
        }
    }

    Ok(graphs)
}

/// Applies one change to `search`, recording added edge ids in `ids`;
/// `false` if there was no live edge to change.
fn apply<P, H, const N: usize>(
    search: &mut DynamicSearch<u32, P, N, H>,
    ids: &mut Vec<usize>,
    nodes: usize,
    change: &Change<N>,
) -> bool
where
    P: Fn(&u32) -> bool,
    H: Heuristic<u32, [u32; N]>,
{
    let live: Vec<usize> = ids
        .iter()
        .copied()
        .filter(|&e| search.graph().edge(e).is_some())
        .collect();
    match *change {
        Change::Cost(k, c) if !live.is_empty() => search.set_cost(live[k % live.len()], c),
        Change::Remove(k) if !live.is_empty() => search.remove_edge(live[k % live.len()]),
        Change::Add(a, b, c) => ids.push(search.add_edge(small(a % nodes), small(b % nodes), c)),
        _ => return false,
    }

    true
}

/// The componentwise minimum of the ideal-point heuristic over `graphs`:
/// admissible on every one of them, and in general inconsistent. `None`
/// only for states that reach no goal in any graph.
fn envelope<const N: usize>(
    problem: &Problem<N>,
    graphs: &[DynamicGraph<u32, N>],
) -> Vec<Option<[u32; N]>> {
    let goals: Vec<u32> = (0..problem.graph.nodes)
        .filter(|&s| problem.goals[s])
        .map(small)
        .collect();
    let mut table = vec![None; problem.graph.nodes];
    for graph in graphs {
        let mut ideal = IdealPoint::new(graph, goals.iter().copied());
        for (s, entry) in table.iter_mut().enumerate() {
            let Some(h) = ideal.estimate(&small(s)) else {
                continue;
            };

            *entry = Some(entry.map_or(h, |old: [u32; N]| {
                core::array::from_fn(|i| old[i].min(h[i]))
            }));
        }
    }

    table
}

/// Checks `changes` without a heuristic, then again guided by the envelope
/// of the ideal-point heuristic over every graph the first run saw.
fn check_both<const N: usize>(
    problem: &Problem<N>,
    changes: &[Change<N>],
) -> Result<(), TestCaseError> {
    let graphs = check(problem, changes, ZeroHeuristic)?;
    let table = envelope(problem, &graphs);
    check(problem, changes, heuristic_fn(|s: &u32| table[*s as usize]))?;
    Ok(())
}

/// A heuristic that points at the goal saves the expansion of a detour
/// that blind search expands before the first goal label exists.
#[test]
fn heuristic_skips_detours() {
    let mut graph = DynamicGraph::new(4);
    graph.add_edge(0, 1, [1_u32, 1]);
    graph.add_edge(1, 3, [1, 1]);
    graph.add_edge(0, 2, [1, 1]);
    graph.add_edge(2, 3, [5, 5]);
    let is_goal = |s: &u32| *s == 3;

    let blind = DynamicSearch::new(graph.clone(), 0, is_goal);
    let h = heuristic_fn(|s: &u32| Some([[2, 2], [1, 1], [5, 5], [0, 0]][*s as usize]));
    let guided = DynamicSearch::with_heuristic(graph, h, 0, is_goal);

    assert_eq!(
        blind.front(),
        Output {
            stats: blind.stats(),
            ..guided.front()
        }
    );
    assert_eq!(blind.stats().expanded, 3);
    assert_eq!(guided.stats().expanded, 2);
}

proptest! {
    /// Two and three objectives, random change sequences.
    #[test]
    fn repairs_match_fresh_searches(p2 in problems::<2>(), c2 in changes::<2>(), p3 in problems::<3>(), c3 in changes::<3>()) {
        check_both(&p2, &c2)?;
        check_both(&p3, &c3)?;
    }
}
