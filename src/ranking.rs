//! The ranking method for bi-objective shortest paths: enumerate loopless
//! paths in order of the first objective and keep the efficient ones.
//!
//! Clímaco and Martins, "A bicriterion shortest path algorithm", EJOR 1982;
//! the ranking class of Craveirinha et al., Algorithms 17(6):222, 2024
//! (Algorithm 2 there, "Generic biobjective ranking method").

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::{Cost, Heuristic};

use crate::anytime_boa::lex::lex_first;
use crate::constrained::rebuild;
use crate::cost::add;
use crate::graph::Graph;
use crate::output::{Output, Stats};

/// Finds one path for every Pareto-optimal cost vector by ranking: loopless
/// paths from `start` to a goal are generated in lexicographic `(c1, c2)`
/// order (a best-first k-shortest-path enumeration), and a path is kept when
/// its second cost is below every kept one. The enumeration stops after the
/// first cost of the lexicographically `(c2, c1)`-optimal path, beyond
/// which nothing is efficient.
///
/// This is the classic ranking approach, exponential in the worst case and
/// much slower than [`boa_star`](crate::boa_star); it is here as the
/// reference the survey describes and for comparison. Goal states end a
/// path.
///
/// Requirements, not checked at run time: non-negative costs and a
/// `heuristic` consistent in both objectives and zero on every goal.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, boa_star, ranking};
///
/// /// A diamond: two routes from 0 to 3.
/// struct Diamond;
///
/// impl Graph for Diamond {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         match state {
///             0 => out.extend([(1, [1, 3]), (2, [3, 1])]),
///             1 | 2 => out.push((3, [1, 1])),
///             _ => {}
///         }
///     }
/// }
///
/// let ranked = ranking(&Diamond, ZeroHeuristic, 0, |s| *s == 3);
/// let exact = boa_star(&Diamond, ZeroHeuristic, 0, |s| *s == 3);
/// assert_eq!(ranked.solutions, exact.solutions);
/// ```
pub fn ranking<G, H, C, P>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; 2]>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut stats = Stats::default();
    let mut solutions = Vec::new();
    let Some(last) = lex_first(graph, &mut heuristic, &start, &is_goal, 1, &mut stats) else {
        return Output { solutions, stats };
    };
    let bound = last.cost[0];

    let mut nodes: Vec<(G::State, [C; 2], Option<usize>)> = Vec::new();
    let mut open = BinaryHeap::new();
    let mut best_c2: Option<C> = None;
    let mut successors = Vec::new();

    if let Some(h) = heuristic.estimate(&start) {
        nodes.push((start, [C::ZERO; 2], None));
        open.push(Reverse((h, 0)));
        stats.generated += 1;
    }

    while let Some(Reverse((f, index))) = open.pop() {
        if f[0] > bound {
            break;
        }

        let (here, g, _) = nodes[index].clone();
        if is_goal(&here) {
            if best_c2.is_none_or(|b| g[1] < b) {
                best_c2 = Some(g[1]);
                solutions.push(rebuild(&nodes, index));
            }

            continue;
        }

        stats.expanded += 1;
        graph.successors(&here, &mut successors);
        for (next, weight) in successors.drain(..) {
            if on_path(&nodes, index, &next) {
                continue;
            }

            let Some(h) = heuristic.estimate(&next) else {
                continue;
            };

            let g_next = add(g, weight);
            nodes.push((next, g_next, Some(index)));
            open.push(Reverse((add(g_next, h), nodes.len() - 1)));
            stats.generated += 1;
        }
    }

    Output { solutions, stats }
}

/// Returns `true` if `state` already lies on the path ending in `index`.
fn on_path<S: PartialEq, C>(nodes: &[(S, [C; 2], Option<usize>)], index: usize, state: &S) -> bool {
    let mut current = Some(index);
    while let Some(i) = current {
        if nodes[i].0 == *state {
            return true;
        }

        current = nodes[i].2;
    }

    false
}
