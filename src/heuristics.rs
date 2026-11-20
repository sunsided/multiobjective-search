//! Heuristic builders: the ideal-point heuristic and differential
//! heuristics, as used across the multi-objective search literature
//! (Salzman, Hernández Ulloa, Felner, Koenig, "Multi-Objective Search:
//! Algorithms, Applications, and Emerging Directions", arXiv 2510.25504,
//! Section 3.2).

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::{Cost, Heuristic, WideMul};

use crate::graph::ReverseGraph;

/// The ideal-point heuristic `h_ideal(v) = (h_1(v), ..., h_N(v))`, with
/// `h_i` the exact shortest distance from `v` to a goal in objective `i`
/// alone. Consistent and zero on goals; a state that reaches no goal has no
/// estimate (a dead end).
///
/// Built by one backward Dijkstra per objective from the goal set, so the
/// graph's reachable part must be finite.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, Heuristic, IdealPoint, ReverseGraph};
///
/// /// 0 → 1 → 2 and 0 → 2 directly.
/// struct Line;
///
/// const EDGES: [(u8, u8, [u32; 2]); 3] = [(0, 1, [1, 5]), (1, 2, [1, 5]), (0, 2, [9, 1])];
///
/// impl Graph for Line {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, s: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         out.extend(EDGES.iter().filter(|e| e.0 == *s).map(|e| (e.1, e.2)));
///     }
/// }
///
/// impl ReverseGraph for Line {
///     fn predecessors(&self, s: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         out.extend(EDGES.iter().filter(|e| e.1 == *s).map(|e| (e.0, e.2)));
///     }
/// }
///
/// let mut h = IdealPoint::new(&Line, [2]);
/// assert_eq!(h.estimate(&0), Some([2, 1])); // the best of each objective
/// assert_eq!(h.estimate(&2), Some([0, 0]));
/// ```
#[derive(Debug, Clone)]
pub struct IdealPoint<S, C, const N: usize> {
    /// The estimate per state that reaches a goal.
    table: BTreeMap<S, [C; N]>,
}

impl<S: Clone + Ord, C: Cost, const N: usize> IdealPoint<S, C, N> {
    /// Computes the heuristic toward `goals` on `graph`.
    pub fn new<G>(graph: &G, goals: impl IntoIterator<Item = S>) -> Self
    where
        G: ReverseGraph<State = S, Weight = [C; N]>,
    {
        let goals: Vec<S> = goals.into_iter().collect();
        let per_objective: Vec<BTreeMap<S, C>> = (0..N)
            .map(|i| {
                dijkstra(
                    goals.iter().cloned(),
                    |s, out| graph.predecessors(s, out),
                    i,
                )
            })
            .collect();

        let table = per_objective[0]
            .keys()
            .map(|s| (s.clone(), core::array::from_fn(|i| per_objective[i][s])))
            .collect();

        Self { table }
    }
}

impl<S: Ord, C: Cost, const N: usize> Heuristic<S, [C; N]> for IdealPoint<S, C, N> {
    fn estimate(&mut self, state: &S) -> Option<[C; N]> {
        self.table.get(state).copied()
    }
}

impl<S: Ord, C: Cost, const N: usize> Heuristic<S, [C; N]> for &IdealPoint<S, C, N> {
    fn estimate(&mut self, state: &S) -> Option<[C; N]> {
        self.table.get(state).copied()
    }
}

/// Single-objective Dijkstra from `sources` over `edges` in objective
/// `objective`: the distance to every state reached.
fn dijkstra<S: Clone + Ord, C: Cost, const N: usize>(
    sources: impl IntoIterator<Item = S>,
    edges: impl Fn(&S, &mut Vec<(S, [C; N])>),
    objective: usize,
) -> BTreeMap<S, C> {
    let mut dist = BTreeMap::new();
    let mut heap: BinaryHeap<Reverse<(C, S)>> =
        sources.into_iter().map(|s| Reverse((C::ZERO, s))).collect();
    let mut buffer = Vec::new();

    while let Some(Reverse((d, s))) = heap.pop() {
        if dist.contains_key(&s) {
            continue;
        }

        edges(&s, &mut buffer);
        for (t, w) in buffer.drain(..) {
            if !dist.contains_key(&t) {
                heap.push(Reverse((d + w[objective], t)));
            }
        }

        dist.insert(s, d);
    }

    dist
}

/// Differential heuristics (Goldberg and Harrelson 2005) per objective:
/// exact distances from and to a few landmark states, giving for any goal
/// `g` the consistent lower bound
/// `h_i(v) = max_L max(d_i(L, g) - d_i(L, v), d_i(v, L) - d_i(g, L), 0)`
/// by the triangle inequality. It is consistent along every edge whose
/// head can reach the goal (a term undefined for lack of a path is left
/// out).
///
/// The memory-based heuristics the survey discusses for multi-objective
/// search; Zhang et al. 2023 extend them to multi-valued bi-objective
/// forms. Build once per graph with [`DifferentialHeuristic::new`], then
/// aim at a goal with [`DifferentialHeuristic::towards`].
///
/// # Examples
///
/// ```
/// use multiobjective_search::{DifferentialHeuristic, Graph, Heuristic, ReverseGraph};
///
/// /// A path 0 - 1 - 2 - 3 in both directions, unit costs.
/// struct Path;
///
/// impl Graph for Path {
///     type State = u8;
///     type Weight = [u32; 1];
///
///     fn successors(&self, s: &u8, out: &mut Vec<(u8, [u32; 1])>) {
///         if *s > 0 { out.push((s - 1, [1])); }
///         if *s < 3 { out.push((s + 1, [1])); }
///     }
/// }
///
/// impl ReverseGraph for Path {
///     fn predecessors(&self, s: &u8, out: &mut Vec<(u8, [u32; 1])>) {
///         self.successors(s, out);
///     }
/// }
///
/// let dh = DifferentialHeuristic::new(&Path, [0]);
/// let mut h = dh.towards(&3);
/// assert_eq!(h.estimate(&1), Some([2])); // exact here: the landmark is an end
/// ```
#[derive(Debug, Clone)]
pub struct DifferentialHeuristic<S, C, const N: usize> {
    /// Per landmark and objective: distances from the landmark.
    from: Vec<Vec<BTreeMap<S, C>>>,
    /// Per landmark and objective: distances to the landmark.
    to: Vec<Vec<BTreeMap<S, C>>>,
}

impl<S: Clone + Ord, C: Cost + WideMul, const N: usize> DifferentialHeuristic<S, C, N> {
    /// Computes the landmark distances: two Dijkstras per landmark and
    /// objective, over the finite reachable graph.
    pub fn new<G>(graph: &G, landmarks: impl IntoIterator<Item = S>) -> Self
    where
        G: ReverseGraph<State = S, Weight = [C; N]>,
    {
        let (mut from, mut to) = (Vec::new(), Vec::new());
        for landmark in landmarks {
            from.push(
                (0..N)
                    .map(|i| dijkstra([landmark.clone()], |s, out| graph.successors(s, out), i))
                    .collect(),
            );
            to.push(
                (0..N)
                    .map(|i| dijkstra([landmark.clone()], |s, out| graph.predecessors(s, out), i))
                    .collect(),
            );
        }

        Self { from, to }
    }

    /// The heuristic toward `goal`.
    #[must_use]
    pub fn towards(&self, goal: &S) -> Towards<'_, S, C, N> {
        Towards {
            landmarks: self,
            goal: goal.clone(),
        }
    }
}

/// [`DifferentialHeuristic`] aimed at one goal.
#[derive(Debug, Clone)]
pub struct Towards<'a, S, C, const N: usize> {
    /// The landmark distances.
    landmarks: &'a DifferentialHeuristic<S, C, N>,
    /// The goal.
    goal: S,
}

impl<S: Ord, C: Cost + WideMul, const N: usize> Heuristic<S, [C; N]> for Towards<'_, S, C, N> {
    fn estimate(&mut self, state: &S) -> Option<[C; N]> {
        let lm = self.landmarks;
        let bound = |i: usize| {
            let mut best = C::ZERO;
            for l in 0..lm.from.len() {
                // d(L, g) - d(L, v) and d(v, L) - d(g, L), where defined.
                let pairs = [
                    (lm.from[l][i].get(&self.goal), lm.from[l][i].get(state)),
                    (lm.to[l][i].get(state), lm.to[l][i].get(&self.goal)),
                ];
                for (big, small) in pairs {
                    if let (Some(&big), Some(&small)) = (big, small) {
                        if big > small {
                            best = best.max(big.abs_diff(small));
                        }
                    }
                }
            }

            best
        };

        Some(core::array::from_fn(bound))
    }
}
