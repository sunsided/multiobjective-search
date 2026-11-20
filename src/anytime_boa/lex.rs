//! The lexicographically optimal solutions that seed A-BOA*ε's first
//! interval (Algorithm 2, lines 1 and 2).

use alloc::collections::{BTreeSet, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::{Cost, Heuristic};

use crate::cost::add;
use crate::graph::Graph;
use crate::output::{Solution, Stats};

/// A* in lexicographic order: the solution with the smallest
/// `(c[first], c[1 - first])`, or `None` if no goal is reachable.
///
/// With a consistent heuristic the first label popped at a state carries
/// its lexicographically smallest `g`, so each state is expanded once.
/// Expansions and generations are added to `stats`.
pub(crate) fn lex_first<G, H, C, P>(
    graph: &G,
    heuristic: &mut H,
    start: &G::State,
    is_goal: &P,
    first: usize,
    stats: &mut Stats,
) -> Option<Solution<G::State, [C; 2]>>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let key = |f: [C; 2]| [f[first], f[1 - first]];
    let mut nodes: Vec<(G::State, [C; 2], Option<usize>)> = Vec::new();
    let mut open = BinaryHeap::new();
    let mut closed = BTreeSet::new();
    let mut successors = Vec::new();

    let h = heuristic.estimate(start)?;
    nodes.push((start.clone(), [C::ZERO; 2], None));
    open.push(Reverse((key(h), 0)));
    stats.generated += 1;

    while let Some(Reverse((_, index))) = open.pop() {
        let (here, g, _) = nodes[index].clone();
        if !closed.insert(here.clone()) {
            continue;
        }

        if is_goal(&here) {
            return Some(rebuild(&nodes, index));
        }

        stats.expanded += 1;
        graph.successors(&here, &mut successors);
        for (next, weight) in successors.drain(..) {
            if closed.contains(&next) {
                continue;
            }

            let Some(h) = heuristic.estimate(&next) else {
                continue;
            };

            let g_next = add(g, weight);
            nodes.push((next, g_next, Some(index)));
            open.push(Reverse((key(add(g_next, h)), nodes.len() - 1)));
            stats.generated += 1;
        }
    }

    None
}

/// Rebuilds the solution ending in node `index`.
fn rebuild<S: Clone, C: Copy>(
    nodes: &[(S, [C; 2], Option<usize>)],
    index: usize,
) -> Solution<S, [C; 2]> {
    let mut path = Vec::new();
    let mut current = Some(index);

    while let Some(i) = current {
        path.push(nodes[i].0.clone());
        current = nodes[i].2;
    }

    path.reverse();

    Solution {
        cost: nodes[index].1,
        path,
    }
}
