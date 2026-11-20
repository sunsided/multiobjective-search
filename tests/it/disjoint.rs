//! Disjoint and dissimilar path tests against exhaustive enumeration of
//! simple paths by edge identity.

use multiobjective_search::{
    Disjointness, disjoint_pair, dissimilar_paths, supported_disjoint_pairs,
};
use proptest::prelude::*;

use crate::fixtures::{EdgeList, problems};

/// Every simple path from `from` that ends at its first goal, as edge
/// indices with the visited nodes.
fn paths(
    graph: &EdgeList<2>,
    from: usize,
    goal: &dyn Fn(usize) -> bool,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    fn walk(
        graph: &EdgeList<2>,
        at: usize,
        goal: &dyn Fn(usize) -> bool,
        nodes: &mut Vec<usize>,
        edges: &mut Vec<usize>,
        out: &mut Vec<(Vec<usize>, Vec<usize>)>,
    ) {
        if goal(at) {
            out.push((nodes.clone(), edges.clone()));
            return;
        }

        for (i, &(f, t, _)) in graph.edges.iter().enumerate() {
            if f == at && !nodes.contains(&t) {
                nodes.push(t);
                edges.push(i);
                walk(graph, t, goal, nodes, edges, out);
                nodes.pop();
                edges.pop();
            }
        }
    }

    let mut out = Vec::new();
    walk(
        graph,
        from,
        goal,
        &mut vec![from],
        &mut Vec::new(),
        &mut out,
    );
    out
}

/// The summed cost vector of edge indices.
fn cost(graph: &EdgeList<2>, edges: &[usize]) -> [u32; 2] {
    edges.iter().fold([0, 0], |s, &e| {
        [s[0] + graph.edges[e].2[0], s[1] + graph.edges[e].2[1]]
    })
}

/// All disjoint pairs from `s` to `t` as summed cost vectors.
fn pair_costs(graph: &EdgeList<2>, s: usize, t: usize, kind: Disjointness) -> Vec<[u32; 2]> {
    let all = paths(graph, s, &|v| v == t);
    let mut out = Vec::new();
    for (i, (na, ea)) in all.iter().enumerate() {
        for (nb, eb) in &all[i + 1..] {
            let disjoint = match kind {
                Disjointness::Edges => ea.iter().all(|e| !eb.contains(e)),
                Disjointness::States => {
                    na[1..na.len() - 1]
                        .iter()
                        .all(|v| !nb[1..nb.len() - 1].contains(v))
                        && ea.iter().all(|e| !eb.contains(e))
                }
            };
            if disjoint {
                let (a, b) = (cost(graph, ea), cost(graph, eb));
                out.push([a[0] + b[0], a[1] + b[1]]);
            }
        }
    }

    out
}

/// Hull vertices of a sorted 2D front.
fn hull(front: &[[u32; 2]]) -> Vec<[u32; 2]> {
    let cross = |o: [u32; 2], a: [u32; 2], b: [u32; 2]| {
        (i64::from(a[0]) - i64::from(o[0])) * (i64::from(b[1]) - i64::from(o[1]))
            - (i64::from(a[1]) - i64::from(o[1])) * (i64::from(b[0]) - i64::from(o[0]))
    };
    let mut h: Vec<[u32; 2]> = Vec::new();
    for &p in front {
        while h.len() >= 2 && cross(h[h.len() - 2], h[h.len() - 1], p) <= 0 {
            h.pop();
        }
        h.push(p);
    }
    h
}

/// Checks that a returned pair is two valid, disjoint paths from `s` to `t`.
fn check_pair(
    graph: &EdgeList<2>,
    s: usize,
    t: usize,
    pair: &multiobjective_search::PathPair<usize, usize>,
    kind: Disjointness,
) -> Result<(), TestCaseError> {
    for (nodes, edges) in [
        (&pair.first, &pair.first_edges),
        (&pair.second, &pair.second_edges),
    ] {
        prop_assert_eq!(nodes.first(), Some(&s));
        prop_assert_eq!(nodes.last(), Some(&t));
        prop_assert_eq!(nodes.len(), edges.len() + 1);
        for (k, &e) in edges.iter().enumerate() {
            prop_assert_eq!(
                (graph.edges[e].0, graph.edges[e].1),
                (nodes[k], nodes[k + 1])
            );
        }
    }

    prop_assert!(
        pair.first_edges
            .iter()
            .all(|e| !pair.second_edges.contains(e))
    );
    if kind == Disjointness::States {
        let inner = |n: &Vec<usize>| n[1..n.len() - 1].to_vec();
        prop_assert!(
            inner(&pair.first)
                .iter()
                .all(|v| !inner(&pair.second).contains(v))
        );
    }

    Ok(())
}

proptest! {
    /// Suurballe's pair is the cheapest disjoint pair, both kinds, and the
    /// supported bi-objective pairs are the hull vertices.
    #[test]
    fn disjoint_pairs_match_enumeration(p in problems::<2>(), t in 0..7_usize) {
        let graph = &p.graph;
        if graph.nodes < 2 {
            return Ok(());
        }

        let (s, t) = (p.start, (p.start + 1 + t % (graph.nodes - 1)) % graph.nodes);

        for kind in [Disjointness::Edges, Disjointness::States] {
            let costs = pair_costs(graph, s, t, kind);
            let best = costs.iter().map(|c| c[0]).min();
            let got = disjoint_pair(graph, s, t, |w: &[u32; 2]| w[0], kind);
            prop_assert_eq!(got.as_ref().map(|g| g.1), best);
            if let Some((pair, total)) = &got {
                check_pair(graph, s, t, pair, kind)?;
                prop_assert_eq!(cost(graph, &pair.first_edges)[0] + cost(graph, &pair.second_edges)[0], *total);
            }

            let mut front: Vec<_> = costs.iter().copied().filter(|a| !costs.iter().any(|b| b != a && b[0] <= a[0] && b[1] <= a[1])).collect();
            front.sort_unstable();
            front.dedup();
            let supported = supported_disjoint_pairs(graph, &s, &t, kind);
            let found: Vec<_> = supported.iter().map(|(_, c)| *c).collect();
            prop_assert_eq!(found, hull(&front));
            for (pair, c) in &supported {
                check_pair(graph, s, t, pair, kind)?;
                let (a, b) = (cost(graph, &pair.first_edges), cost(graph, &pair.second_edges));
                prop_assert_eq!([a[0] + b[0], a[1] + b[1]], *c);
            }
        }
    }

    /// The greedy dissimilar paths match a greedy pass over all simple
    /// paths sorted by (cost, edge sequence).
    #[test]
    fn dissimilar_matches_enumeration(p in problems::<2>(), k in 1..5_usize, shared in 0..3_usize) {
        let graph = &p.graph;
        let goal = |v: usize| p.goals[v];
        let mut all: Vec<(u32, Vec<usize>)> = paths(graph, p.start, &goal).into_iter().map(|(_, e)| (cost(graph, &e)[0], e)).collect();
        all.sort();
        let mut kept: Vec<(u32, Vec<usize>)> = Vec::new();
        for (c, e) in all {
            if kept.len() < k && kept.iter().all(|(_, f)| e.iter().filter(|x| f.contains(x)).count() <= shared) {
                kept.push((c, e));
            }
        }

        let got = dissimilar_paths(graph, p.start, |v: &usize| p.goals[*v], |w: &[u32; 2]| w[0], k, shared);
        let got: Vec<(u32, Vec<usize>)> = got.into_iter().map(|(_, e, c)| (c, e)).collect();
        prop_assert_eq!(got, kept);
    }
}
