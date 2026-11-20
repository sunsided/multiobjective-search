//! Several additive costs together with several bottleneck capacities.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::{Cost, Heuristic};

use crate::bottleneck::{Bottleneck, Width};
use crate::cost::{add, weakly_dominates};
use crate::graph::Graph;
use crate::output::{Output, Solution, Stats};

/// The costs of a path in [`minsum_maxmin_multi`]: `N` sums and `M`
/// bottlenecks.
pub type SumsAndWidths<C, K, const N: usize, const M: usize> = ([C; N], [Bottleneck<K>; M]);

/// Finds one path for every Pareto-optimal cost from `start` to a state
/// that satisfies `is_goal`, with `N` additive costs to minimize and `M`
/// bottleneck capacities to maximize (edge weights `(sums, capacities)`).
///
/// This generalizes [`minsum_maxmin`](crate::minsum_maxmin) to the
/// problems with several maxmin objectives surveyed by Craveirinha et al.
/// (Algorithms 17(6):222, 2024; Pinto et al. 2009, Gandibleux et al.
/// 2006, Bornstein et al. 2012). Labels leave the open list in
/// lexicographic `f` order, wider bottlenecks first among equal ones, and
/// are dropped when an earlier label at their state, or an earlier
/// solution against `f`, is no worse in every sum and every bottleneck.
/// That keeps one path per non-dominated cost (the minimal complete set):
/// with bottlenecks, an efficient path may run through a weakly efficient
/// sub-path, which is why the dominance test compares whole labels rather
/// than truncated ones.
///
/// Requirements, not checked at run time: non-negative costs and a
/// `heuristic` on the sums that is consistent and zero on every goal.
///
/// Solutions are sorted lexicographically by their sums.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, minsum_maxmin_multi};
///
/// /// Two links from 0 to 1: one wide in bandwidth, one in reliability.
/// struct Links;
///
/// impl Graph for Links {
///     type State = u8;
///     type Weight = ([u32; 1], [u32; 2]);
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, ([u32; 1], [u32; 2]))>) {
///         if *state == 0 {
///             out.extend([(1, ([5], [100, 1])), (1, ([5], [1, 100])), (1, ([9], [1, 1]))]);
///         }
///     }
/// }
///
/// let out = minsum_maxmin_multi(&Links, ZeroHeuristic, 0, |s| *s == 1);
/// assert_eq!(out.solutions.len(), 2); // the third link is dominated
/// ```
pub fn minsum_maxmin_multi<G, H, C, K, P, const N: usize, const M: usize>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
) -> Output<G::State, SumsAndWidths<C, K, N, M>>
where
    G: Graph<Weight = ([C; N], [K; M])>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost,
    K: Copy + Ord,
    P: Fn(&G::State) -> bool,
{
    let mut nodes: Vec<Node<G::State, C, K, N, M>> = Vec::new();
    let mut open = BinaryHeap::new();
    let mut closed: BTreeMap<G::State, Vec<Label<C, K, N, M>>> = BTreeMap::new();
    let mut found: Vec<Label<C, K, N, M>> = Vec::new();
    let mut solutions = Vec::new();
    let mut successors = Vec::new();
    let mut stats = Stats::default();

    if let Some(h) = heuristic.estimate(&start) {
        nodes.push(Node {
            state: start,
            label: ([C::ZERO; N], [None; M]),
            parent: None,
        });
        open.push(Reverse((h, Reverse([Width(None); M]), 0)));
        stats.generated += 1;
    }

    while let Some(Reverse((f, _, index))) = open.pop() {
        let (here, label) = (nodes[index].state.clone(), nodes[index].label);
        if covered(&found, &(f, label.1)) || closed.get(&here).is_some_and(|c| covered(c, &label)) {
            continue;
        }

        closed.entry(here.clone()).or_default().push(label);
        if is_goal(&here) {
            found.push(label);
            solutions.push(rebuild(&nodes, index));
            continue;
        }

        stats.expanded += 1;
        graph.successors(&here, &mut successors);
        for (next, (sums, capacities)) in successors.drain(..) {
            let child: Label<C, K, N, M> = (
                add(label.0, sums),
                core::array::from_fn(|i| {
                    Some(label.1[i].map_or(capacities[i], |w| w.min(capacities[i])))
                }),
            );
            if closed.get(&next).is_some_and(|c| covered(c, &child)) {
                continue;
            }

            let Some(h) = heuristic.estimate(&next) else {
                continue;
            };

            let f = add(child.0, h);
            if covered(&found, &(f, child.1)) {
                continue;
            }

            nodes.push(Node {
                state: next,
                label: child,
                parent: Some(index),
            });
            open.push(Reverse((f, Reverse(child.1.map(Width)), nodes.len() - 1)));
            stats.generated += 1;
        }
    }

    Output { solutions, stats }
}

/// Sums and bottlenecks of a label.
type Label<C, K, const N: usize, const M: usize> = SumsAndWidths<C, K, N, M>;

/// One label in the arena.
struct Node<S, C, K, const N: usize, const M: usize> {
    /// The state the path ends in.
    state: S,
    /// Its sums and bottlenecks.
    label: Label<C, K, N, M>,
    /// Arena index of the path without its last edge.
    parent: Option<usize>,
}

/// Returns `true` if some label in `known` is no worse than `label`: no
/// larger in every sum, no narrower in every bottleneck.
fn covered<C: Cost, K: Copy + Ord, const N: usize, const M: usize>(
    known: &[Label<C, K, N, M>],
    label: &Label<C, K, N, M>,
) -> bool {
    known.iter().any(|k| {
        weakly_dominates(&k.0, &label.0) && (0..M).all(|i| Width(k.1[i]) >= Width(label.1[i]))
    })
}

/// Rebuilds the solution ending in node `index`.
fn rebuild<S: Clone, C: Copy, K: Copy, const N: usize, const M: usize>(
    nodes: &[Node<S, C, K, N, M>],
    index: usize,
) -> Solution<S, Label<C, K, N, M>> {
    let mut path = Vec::new();
    let mut current = Some(index);

    while let Some(i) = current {
        path.push(nodes[i].state.clone());
        current = nodes[i].parent;
    }

    path.reverse();

    Solution {
        cost: nodes[index].label,
        path,
    }
}
