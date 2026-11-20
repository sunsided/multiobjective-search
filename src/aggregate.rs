//! Multi-objective search over objective-aggregation functions: solution
//! objectives computed from hidden search objectives.
//!
//! Peer, Weiss, Alterovitz, Salzman: "Generalizing Multi-Objective Search
//! via Objective-Aggregation Functions", arXiv 2509.22085.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::{Cost, Heuristic};

use crate::cost::add;
use crate::epsilon::Epsilon;
use crate::graph::Graph;
use crate::output::{Output, Solution, Stats};
use crate::pareto::Front;

/// The functions that turn edge costs into `M` hidden objectives and those
/// into `K` solution objectives.
///
/// - `extend` is the path-cost extension `F_ext(g, c)`: the hidden cost of
///   a path extended by an edge of weight `c`. It must be monotone in `g`
///   (`g <= g'` component-wise gives `F_ext(g, c) <= F_ext(g', c)`) and
///   never decrease `g`. Plain addition is the usual additive case.
/// - `aggregate` is the objective aggregation `F_agg`: the solution cost of
///   a path's hidden cost. It must be monotone non-decreasing. The
///   identity is the usual case.
#[derive(Debug, Clone, Copy)]
pub struct Aggregation<X, A> {
    /// `F_ext`.
    pub extend: X,
    /// `F_agg`.
    pub aggregate: A,
}

/// The result of [`aggregated_search`]: the solutions with their
/// aggregated costs, and the hidden cost of each, in the same order.
pub type Aggregated<S, C, const M: usize, const K: usize> = (Output<S, [C; K]>, Vec<[C; M]>);

/// A solution with its hidden cost.
type WithHidden<S, C, const M: usize, const K: usize> = (Solution<S, [C; K]>, [C; M]);

/// `(k, m)`-MOS-A*: the paths from `start` to a goal whose aggregated
/// solution costs `F_agg(c_ext(π))` are Pareto-optimal, or, with tolerances
/// `eps`, a set that covers each of them within `1 + eps[i]`.
///
/// The paper's Algorithm 1 with its changes: the open list is ordered
/// lexicographically by `F_agg(f)` with `f = g + h` in the hidden space,
/// labels are pruned against labels expanded at their state by the hidden
/// `g` (full weak dominance), and against solutions by the aggregated
/// costs, with the tolerances for the approximate variant. Under the
/// monotonicity requirements the result is the aggregated front
/// (Theorem 1). Since the hidden dimension `M` is usually larger than `K`,
/// this is much faster than the straw man that computes the hidden front
/// first.
///
/// Requirements, not checked at run time: the conditions of
/// [`Aggregation`], and a `heuristic` in the hidden space with
/// `g + h(s) <=` the hidden cost of every completion through `s` (zero on
/// goals). The zero heuristic always qualifies.
///
/// Solutions are sorted by aggregated cost, with their hidden costs
/// alongside; dominated or duplicate aggregated costs are dropped.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Aggregation, Epsilon, Graph, ZeroHeuristic, aggregated_search};
///
/// /// Two routes from 0 to 2 through two risky zones (hidden objectives:
/// /// length, and the worst exposure in each zone).
/// struct Map;
///
/// impl Graph for Map {
///     type State = u8;
///     type Weight = [u32; 3];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 3])>) {
///         match state {
///             0 => out.extend([(1, [1, 5, 0]), (1, [3, 1, 1])]),
///             1 => out.push((2, [1, 0, 4])),
///             _ => {}
///         }
///     }
/// }
///
/// // Length adds up; each zone keeps its worst exposure; the solution
/// // objectives are length and total exposure.
/// let functions = Aggregation {
///     extend: |g: &[u32; 3], c: &[u32; 3]| [g[0] + c[0], g[1].max(c[1]), g[2].max(c[2])],
///     aggregate: |h: &[u32; 3]| [h[0], h[1] + h[2]],
/// };
/// let (out, hidden) = aggregated_search(&Map, ZeroHeuristic, 0, |s| *s == 2, functions, [Epsilon::ZERO; 2]);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[2, 9], [4, 5]]);
/// assert_eq!(hidden, [[2, 5, 4], [4, 1, 4]]);
/// ```
pub fn aggregated_search<G, H, X, A, C, P, const M: usize, const K: usize>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
    functions: Aggregation<X, A>,
    eps: [Epsilon; K],
) -> Aggregated<G::State, C, M, K>
where
    G: Graph,
    H: Heuristic<G::State, [C; M]>,
    X: Fn(&[C; M], &G::Weight) -> [C; M],
    A: Fn(&[C; M]) -> [C; K],
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let Aggregation { extend, aggregate } = functions;
    let mut nodes: Vec<(G::State, [C; M], Option<usize>)> = Vec::new();
    let mut open = BinaryHeap::new();
    let mut closed: BTreeMap<G::State, Front<C, M>> = BTreeMap::new();
    let mut found: Vec<[C; K]> = Vec::new();
    let mut solutions: Vec<WithHidden<G::State, C, M, K>> = Vec::new();
    let mut successors = Vec::new();
    let mut stats = Stats::default();
    let covered = |found: &[[C; K]], f: &[C; K]| {
        found
            .iter()
            .any(|s| (0..K).all(|i| eps[i].within(s[i], f[i])))
    };

    if let Some(h) = heuristic.estimate(&start) {
        let g = [C::ZERO; M];
        nodes.push((start, g, None));
        open.push(Reverse((aggregate(&add(g, h)), add(g, h), 0)));
        stats.generated += 1;
    }

    while let Some(Reverse((key, _, index))) = open.pop() {
        let (here, g) = (nodes[index].0.clone(), nodes[index].1);
        if covered(&found, &key) || closed.get(&here).is_some_and(|c| c.weakly_dominates(&g)) {
            continue;
        }

        closed.entry(here.clone()).or_default().insert(g);
        if is_goal(&here) {
            found.push(aggregate(&g));
            solutions.push((rebuild(&nodes, index, aggregate(&g)), g));
            continue;
        }

        stats.expanded += 1;
        graph.successors(&here, &mut successors);
        for (next, weight) in successors.drain(..) {
            let g_next = extend(&g, &weight);
            if closed
                .get(&next)
                .is_some_and(|c| c.weakly_dominates(&g_next))
            {
                continue;
            }

            let Some(h) = heuristic.estimate(&next) else {
                continue;
            };

            let f = add(g_next, h);
            let key = aggregate(&f);
            if !covered(&found, &key) {
                nodes.push((next, g_next, Some(index)));
                open.push(Reverse((key, f, nodes.len() - 1)));
                stats.generated += 1;
            }
        }
    }

    solutions.sort_by_key(|(s, _)| s.cost);
    let mut front = Front::new();
    solutions.retain(|(s, _)| front.insert(s.cost));
    let (solutions, hidden) = solutions.into_iter().unzip();

    (Output { solutions, stats }, hidden)
}

/// Rebuilds the path ending in label `index` with solution cost `cost`.
fn rebuild<S: Clone, C, const M: usize, const K: usize>(
    nodes: &[(S, [C; M], Option<usize>)],
    index: usize,
    cost: [C; K],
) -> Solution<S, [C; K]> {
    let mut path = Vec::new();
    let mut current = Some(index);

    while let Some(i) = current {
        path.push(nodes[i].0.clone());
        current = nodes[i].2;
    }

    path.reverse();

    Solution { cost, path }
}
