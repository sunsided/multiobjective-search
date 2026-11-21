//! Fuzzes the DIMACS loader: arbitrary text never panics, and a graph that
//! loads has exactly the arcs of its files, reachable in both directions.
//!
//! The input is split at NUL bytes into a distance `.gr`, a time `.gr` and
//! an optional `.co` file.

#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;
use multiobjective_search::{Graph, ReverseGraph, RoadGraph};

/// The largest node count a problem line may declare.
///
/// The graph stores one row offset per declared node, so `p sp 4000000000 0`
/// is a valid 18-byte file that allocates tens of gigabytes. That is the
/// format's cost, not a parser bug; the cap keeps libFuzzer's memory limit
/// from reporting it.
const MAX_NODES: u64 = 1 << 16;

/// Returns `true` if a problem line in `text` declares more than
/// [`MAX_NODES`] nodes (its first number; the arc count stays unlimited).
fn declares_huge_graph(text: &str) -> bool {
    text.lines()
        .filter(|line| line.split_whitespace().next() == Some("p"))
        .filter_map(|line| line.split_whitespace().find_map(|f| f.parse::<u64>().ok()))
        .any(|nodes| nodes > MAX_NODES)
}

/// Counts the arc lines (`a ...`) of a `.gr` file.
fn arc_lines(text: &str) -> usize {
    text.lines()
        .filter(|line| line.split_whitespace().next() == Some("a"))
        .count()
}

/// Checks that every arc is listed once from its source and once from its
/// target, and that out-of-range ids have no arcs.
fn check_adjacency(graph: &RoadGraph) {
    let mut out = Vec::new();
    let mut forward = 0;
    let mut backward = 0;

    for node in 0..=graph.node_count().saturating_add(1) {
        out.clear();
        graph.successors(&node, &mut out);
        forward += out.len();

        out.clear();
        graph.predecessors(&node, &mut out);
        backward += out.len();
        assert!(
            out.iter()
                .all(|&(from, _)| (1..=graph.node_count()).contains(&from))
        );
    }

    assert_eq!(forward, graph.arc_count());
    assert_eq!(backward, graph.arc_count());
}

fuzz_target!(|text: &str| {
    if declares_huge_graph(text) {
        return;
    }

    let mut files = text.split('\0');
    let distance = files.next().unwrap_or_default();
    let time = files.next().unwrap_or_default();
    let co = files.next();

    let Ok(graph) = RoadGraph::from_dimacs(distance, time) else {
        return;
    };

    assert_eq!(graph.arc_count(), arc_lines(distance));
    check_adjacency(&graph);

    if let Some(co) = co
        && let Ok(graph) = graph.with_coordinates(co)
    {
        assert_eq!(graph.coordinates(0), None);
        assert_eq!(
            graph.coordinates(graph.node_count().saturating_add(1)),
            None
        );
    }
});
