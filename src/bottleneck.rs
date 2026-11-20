//! Minsum–maxmin paths: one additive cost to minimize and one bottleneck
//! capacity to maximize.
//!
//! Martins, "On a special class of bicriterion path problems", EJOR 1984;
//! Gandibleux, Beugnies, Randriamasy, "Martins' algorithm revisited for
//! multi-objective shortest path problems with a `MaxMin` cost function",
//! 4OR 2006. Surveyed by Craveirinha et al., Algorithms 17(6):222, 2024.

mod multi;
mod ratio;

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::{Cost, Heuristic};

use crate::graph::Graph;
use crate::output::{Output, Solution, Stats};

pub use multi::{SumsAndWidths, minsum_maxmin_multi};
pub use ratio::{WidePath, min_cost_capacity_ratio, quickest_path};

/// The capacity of a path: the smallest capacity of its edges, or `None`
/// (unbounded) for the empty path.
pub type Bottleneck<K> = Option<K>;

/// Finds one path for every Pareto-optimal `(cost, bottleneck)` pair from
/// `start` to a state that satisfies `is_goal`: the cost (the sum of the
/// edges' first components) as small as possible, the bottleneck (the
/// smallest of their second components) as large as possible.
///
/// Labels leave the open list in increasing `cost + heuristic`, and among
/// equal ones in decreasing bottleneck. A label is then dominated exactly
/// when an earlier label at its state, or an earlier solution, has a
/// bottleneck at least as large, as in BOA*: one value per state suffices.
/// A min–max objective (a largest delay to minimize) is the same problem
/// with `K = core::cmp::Reverse<_>`.
///
/// Requirements, not checked at run time: non-negative costs and a
/// `heuristic` on the cost that is consistent and zero on every goal.
///
/// Solutions are sorted by cost ascending, bottleneck strictly descending,
/// as `(cost, bottleneck)`; a start that is a goal gives the unbounded
/// bottleneck `None`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, heuristic_fn, minsum_maxmin};
///
/// /// Two routes from 0 to 2: cheap and narrow, or dear and wide.
/// struct Links;
///
/// impl Graph for Links {
///     type State = u8;
///     type Weight = (u32, u32);
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, (u32, u32))>) {
///         match state {
///             0 => out.extend([(1, (1, 10)), (2, (5, 100))]),
///             1 => out.push((2, (1, 100))),
///             _ => {}
///         }
///     }
/// }
///
/// let out = minsum_maxmin(&Links, heuristic_fn(|_: &u8| Some(0)), 0, |s| *s == 2);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [(2, Some(10)), (5, Some(100))]);
/// ```
pub fn minsum_maxmin<G, H, C, K, P>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
) -> Output<G::State, (C, Bottleneck<K>)>
where
    G: Graph<Weight = (C, K)>,
    H: Heuristic<G::State, C>,
    C: Cost,
    K: Copy + Ord,
    P: Fn(&G::State) -> bool,
{
    let mut nodes: Vec<Node<G::State, C, K>> = Vec::new();
    let mut open = BinaryHeap::new();
    let mut best: BTreeMap<G::State, Bottleneck<K>> = BTreeMap::new();
    let mut solved: Option<Bottleneck<K>> = None;
    let mut solutions = Vec::new();
    let mut successors = Vec::new();
    let mut stats = Stats::default();

    if let Some(h) = heuristic.estimate(&start) {
        nodes.push(Node {
            state: start,
            cost: C::ZERO,
            width: None,
            parent: None,
        });
        open.push(Reverse((h, Reverse(Width(None)), 0)));
        stats.generated += 1;
    }

    while let Some(Reverse((_, _, index))) = open.pop() {
        let (here, cost, width) = {
            let node = &nodes[index];
            (node.state.clone(), node.cost, node.width)
        };

        if covers(solved, width) || best.get(&here).is_some_and(|&b| covers(Some(b), width)) {
            continue;
        }

        best.insert(here.clone(), width);
        if is_goal(&here) {
            solved = Some(width);
            solutions.push(rebuild(&nodes, index));
            continue;
        }

        stats.expanded += 1;
        graph.successors(&here, &mut successors);
        for (next, (edge_cost, capacity)) in successors.drain(..) {
            let narrowed = Some(width.map_or(capacity, |w| w.min(capacity)));
            if covers(solved, narrowed)
                || best.get(&next).is_some_and(|&b| covers(Some(b), narrowed))
            {
                continue;
            }

            let Some(h) = heuristic.estimate(&next) else {
                continue;
            };

            let g = cost + edge_cost;
            nodes.push(Node {
                state: next,
                cost: g,
                width: narrowed,
                parent: Some(index),
            });
            open.push(Reverse((g + h, Reverse(Width(narrowed)), nodes.len() - 1)));
            stats.generated += 1;
        }
    }

    Output { solutions, stats }
}

/// One label in the arena.
struct Node<S, C, K> {
    /// The state the path ends in.
    state: S,
    /// Sum of the edge costs.
    cost: C,
    /// Smallest edge capacity; `None` for the empty path.
    width: Bottleneck<K>,
    /// Arena index of the path without its last edge.
    parent: Option<usize>,
}

/// A bottleneck ordered with `None` (unbounded) as the largest value.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Width<K>(Bottleneck<K>);

impl<K: Ord> PartialOrd for Width<K> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord> Ord for Width<K> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        match (&self.0, &other.0) {
            (None, None) => core::cmp::Ordering::Equal,
            (None, Some(_)) => core::cmp::Ordering::Greater,
            (Some(_), None) => core::cmp::Ordering::Less,
            (Some(a), Some(b)) => a.cmp(b),
        }
    }
}

/// Returns `true` if an earlier label with bottleneck `known` (and no
/// larger cost) makes one with bottleneck `width` redundant: `known`
/// is at least as wide. `None` for `known` means "no such label".
fn covers<K: Ord + Copy>(known: Option<Bottleneck<K>>, width: Bottleneck<K>) -> bool {
    known.is_some_and(|k| Width(k) >= Width(width))
}

/// Rebuilds the solution ending in node `index`.
fn rebuild<S: Clone, C: Copy, K: Copy>(
    nodes: &[Node<S, C, K>],
    index: usize,
) -> Solution<S, (C, Bottleneck<K>)> {
    let mut path = Vec::new();
    let mut current = Some(index);

    while let Some(i) = current {
        path.push(nodes[i].state.clone());
        current = nodes[i].parent;
    }

    path.reverse();

    Solution {
        cost: (nodes[index].cost, nodes[index].width),
        path,
    }
}
