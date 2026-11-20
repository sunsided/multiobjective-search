//! Resource-constrained shortest paths: the cheapest path whose second
//! cost stays within a budget.
//!
//! A classic constrained relative of the bi-objective problem (the
//! "constrained shortest path" class surveyed by Craveirinha et al.,
//! Algorithms 17(6):222, 2024); solved here by BOA*'s pruning with the
//! budget as an extra bound, stopping at the first solution.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::{Cost, Heuristic};

use crate::cost::add;
use crate::graph::Graph;
use crate::output::{Solution, Stats};

/// Finds the path from `start` to a state that satisfies `is_goal` with
/// the smallest first cost among those whose second cost is at most
/// `budget`, the smallest second cost breaking ties; `None` if no path
/// fits the budget.
///
/// Labels leave the open list in lexicographic `f` order. A label is
/// dropped when `f2 > budget`, or, as in BOA*, when an earlier label at its
/// state had a second cost no larger (it reaches everything this one does,
/// no dearer). The first goal label is the answer: every label still open
/// costs at least as much in the first objective.
///
/// Requirements, not checked at run time: non-negative costs and a
/// `heuristic` consistent in both objectives and zero on every goal.
/// Expansion counters go to `stats`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, Stats, ZeroHeuristic, constrained_shortest_path};
///
/// /// Three routes from 0 to 1: (time, fuel).
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [1, 9]), (1, [3, 5]), (1, [8, 1])]);
///         }
///     }
/// }
///
/// let mut stats = Stats::default();
/// let best = constrained_shortest_path(&Routes, ZeroHeuristic, 0, |s| *s == 1, 6, &mut stats);
/// assert_eq!(best.map(|s| s.cost), Some([3, 5]));
/// ```
pub fn constrained_shortest_path<G, H, C, P>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
    budget: C,
    stats: &mut Stats,
) -> Option<Solution<G::State, [C; 2]>>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut nodes: Vec<(G::State, [C; 2], Option<usize>)> = Vec::new();
    let mut open = BinaryHeap::new();
    let mut g2_min: BTreeMap<G::State, C> = BTreeMap::new();
    let mut successors = Vec::new();

    let h = heuristic.estimate(&start)?;
    if h[1] > budget {
        return None;
    }

    nodes.push((start, [C::ZERO; 2], None));
    open.push(Reverse((h, 0)));
    stats.generated += 1;

    while let Some(Reverse((_, index))) = open.pop() {
        let (here, g, _) = nodes[index].clone();
        if g2_min.get(&here).is_some_and(|&m| g[1] >= m) {
            continue;
        }

        g2_min.insert(here.clone(), g[1]);
        if is_goal(&here) {
            return Some(rebuild(&nodes, index));
        }

        stats.expanded += 1;
        graph.successors(&here, &mut successors);
        for (next, weight) in successors.drain(..) {
            let g_next = add(g, weight);
            if g2_min.get(&next).is_some_and(|&m| g_next[1] >= m) {
                continue;
            }

            let Some(h) = heuristic.estimate(&next) else {
                continue;
            };

            let f = add(g_next, h);
            if f[1] > budget {
                continue;
            }

            nodes.push((next, g_next, Some(index)));
            open.push(Reverse((f, nodes.len() - 1)));
            stats.generated += 1;
        }
    }

    None
}

/// Rebuilds the solution ending in node `index`.
pub(crate) fn rebuild<S: Clone, C: Copy, const N: usize>(
    nodes: &[(S, [C; N], Option<usize>)],
    index: usize,
) -> Solution<S, [C; N]> {
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
