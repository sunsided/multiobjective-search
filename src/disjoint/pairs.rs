//! Exact bi-objective fronts of path pairs, by label setting over the
//! space of partial pairs.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use search_core::Cost;

use crate::boa::boa_star;
use crate::disjoint::{CostedPair, Disjointness, PathPair};
use crate::graph::{EdgeGraph, Graph, ZeroHeuristic};
use crate::output::Solution;

/// The full Pareto-optimal front of disjoint path pairs from `start` to
/// `target`, the costs `(c1, c2)` summed over both paths: one pair per
/// non-dominated total, sorted by cost.
///
/// The bicriteria disjoint path pair problem of Clímaco and Pascoal
/// (2009), supported and non-supported pairs alike. A pair is built as a
/// path in the space of partial pairs: the first path from `start` to
/// `target`, then the second, each state remembering the edges (or, for
/// [`Disjointness::States`], the states) already used, so that a pair is a
/// path there and its cost is the sum of both. Both paths are kept simple,
/// which loses nothing: cutting a cycle never raises a cost. BOA* over that space returns
/// exactly the non-dominated totals. The space grows exponentially with
/// the graph, as every exact method for this NP-hard problem does; use it
/// on small graphs, and [`supported_disjoint_pairs`](crate::supported_disjoint_pairs)
/// for the polynomial supported part.
///
/// Requirements: non-negative costs, a finite graph, `start != target`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Disjointness, EdgeGraph, Graph, disjoint_pair_front};
///
/// /// Three parallel routes from 0 to 1.
/// struct Routes;
///
/// const EDGES: [[u32; 2]; 3] = [[1, 9], [5, 5], [9, 1]];
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend(EDGES.iter().map(|&w| (1, w)));
///         }
///     }
/// }
///
/// impl EdgeGraph for Routes {
///     type Edge = usize;
///
///     fn edges(&self, state: &u8, out: &mut Vec<(usize, u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend(EDGES.iter().enumerate().map(|(i, &w)| (i, 1, w)));
///         }
///     }
/// }
///
/// let front = disjoint_pair_front(&Routes, &0, &1, Disjointness::Edges);
/// let costs: Vec<_> = front.iter().map(|(_, c)| *c).collect();
/// assert_eq!(costs, [[6, 14], [10, 10], [14, 6]]); // (10, 10) is not extreme
/// ```
pub fn disjoint_pair_front<G, C>(
    graph: &G,
    start: &G::State,
    target: &G::State,
    kind: Disjointness,
) -> Vec<CostedPair<G::State, G::Edge, [C; 2]>>
where
    G: EdgeGraph<Weight = [C; 2]>,
    C: Cost,
{
    let mode = match kind {
        Disjointness::Edges => Mode::Edges,
        Disjointness::States => Mode::States,
    };

    solve(graph, start, target, mode, |w: &[C; 2]| *w, C::ZERO)
}

/// The maximally disjoint pairs: the Pareto-optimal trade-offs between the
/// total cost of two paths from `start` to `target` and the number of
/// edges they share, as `(pair, total cost, shared edges)` sorted by cost.
///
/// The relaxed maximally disjoint path pair problem (Pascoal and Clímaco
/// 2020) is a bi-objective two-unit flow: every edge carries one unit at
/// its cost, or two units at twice its cost and one shared edge. The
/// shared-edge count is small and integral, so the front is solved exactly
/// as an ε-constraint family over it at once: BOA* over the space of
/// partial pairs with objectives `(cost, shared)`, the first path edge-
/// simple, the second allowed onto the first's edges at a count of one each.
/// The pair with zero shared edges, if any, is
/// [`disjoint_pair`](crate::disjoint_pair)'s.
///
/// Requirements: non-negative `cost`, a finite graph, `start != target`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{EdgeGraph, Graph, maximally_disjoint_pairs};
///
/// /// A bridge 0 → 1 that both paths must cross, then two ways to 2.
/// struct Bridge;
///
/// const EDGES: [(u8, u8, u32); 3] = [(0, 1, 1), (1, 2, 1), (1, 2, 3)];
///
/// impl Graph for Bridge {
///     type State = u8;
///     type Weight = u32;
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, u32)>) {
///         out.extend(EDGES.iter().filter(|e| e.0 == *state).map(|e| (e.1, e.2)));
///     }
/// }
///
/// impl EdgeGraph for Bridge {
///     type Edge = usize;
///
///     fn edges(&self, state: &u8, out: &mut Vec<(usize, u8, u32)>) {
///         out.extend(EDGES.iter().enumerate().filter(|(_, e)| e.0 == *state).map(|(i, e)| (i, e.1, e.2)));
///     }
/// }
///
/// let front = maximally_disjoint_pairs(&Bridge, &0, &2, |w: &u32| *w);
/// let trade: Vec<_> = front.iter().map(|(_, c, s)| (*c, *s)).collect();
/// assert_eq!(trade, [(4, 2), (6, 1)]); // share both edges, or only the bridge
/// ```
pub fn maximally_disjoint_pairs<G, C, F>(
    graph: &G,
    start: &G::State,
    target: &G::State,
    cost: F,
) -> Vec<SharedPair<G::State, G::Edge, C>>
where
    G: EdgeGraph,
    C: Cost + From<u8>,
    F: Fn(&G::Weight) -> C,
{
    let pairs = solve(
        graph,
        start,
        target,
        Mode::Shared,
        |w| [cost(w), C::ZERO],
        C::from(1),
    );
    pairs
        .into_iter()
        .map(|(pair, [total, _])| {
            let shared = pair
                .first_edges
                .iter()
                .filter(|e| pair.second_edges.contains(e))
                .count();
            (pair, total, shared)
        })
        .collect()
}

/// A pair with its total cost and its number of shared edges.
pub type SharedPair<S, E, C> = (PathPair<S, E>, C, usize);

/// What the second path may share with the first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// No edge.
    Edges,
    /// No state but the ends.
    States,
    /// Edges, each counted in the second objective.
    Shared,
}

/// A partial pair: the first path so far, or the whole first path and the
/// second so far.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Partial<S, E> {
    /// `false` while the first path is built.
    second: bool,
    /// The current end.
    at: S,
    /// The edges of the first path.
    first_edges: BTreeSet<E>,
    /// The edges of the second path.
    second_edges: BTreeSet<E>,
    /// The states of the first path.
    first_states: BTreeSet<S>,
    /// The states of the second path.
    second_states: BTreeSet<S>,
    /// The edge that led here; `None` at either start.
    via: Option<E>,
}

/// The graph of partial pairs.
struct Pairs<'a, G: Graph, F, C> {
    /// The underlying graph.
    graph: &'a G,
    /// Where both paths start.
    start: G::State,
    /// Where both paths end.
    target: G::State,
    /// The sharing rule.
    mode: Mode,
    /// Edge cost vectors.
    cost: F,
    /// One shared edge in the second objective.
    one: C,
}

impl<G, F, C> Graph for Pairs<'_, G, F, C>
where
    G: EdgeGraph,
    F: Fn(&G::Weight) -> [C; 2],
    C: Cost,
{
    type State = Partial<G::State, G::Edge>;
    type Weight = [C; 2];

    fn successors(&self, state: &Self::State, out: &mut Vec<(Self::State, [C; 2])>) {
        if state.at == self.target {
            if !state.second {
                let mut next = state.clone();
                next.second = true;
                next.at = self.start.clone();
                next.via = None;
                next.second_states.insert(self.start.clone());
                out.push((next, [C::ZERO; 2]));
            }

            return;
        }

        let mut edges = Vec::new();
        self.graph.edges(&state.at, &mut edges);
        for (edge, to, weight) in edges {
            let mut cost = (self.cost)(&weight);
            let mut next = state.clone();

            if state.second {
                // Each path is simple: removing a cycle never costs more
                // and never adds a shared element.
                if next.second_states.contains(&to) {
                    continue;
                }

                let inner = to != self.target && to != self.start;
                let blocked = match self.mode {
                    Mode::Edges => next.first_edges.contains(&edge),
                    Mode::States => {
                        next.first_edges.contains(&edge)
                            || (inner && next.first_states.contains(&to))
                    }
                    Mode::Shared => false,
                };
                if blocked {
                    continue;
                }

                if self.mode == Mode::Shared && next.first_edges.contains(&edge) {
                    cost[1] = self.one;
                }

                next.second_edges.insert(edge.clone());
                next.second_states.insert(to.clone());
            } else {
                if next.first_states.contains(&to) {
                    continue;
                }

                next.first_edges.insert(edge.clone());
                next.first_states.insert(to.clone());
            }

            next.at = to;
            next.via = Some(edge);
            out.push((next, cost));
        }
    }
}

/// Runs BOA* over the partial pairs and maps the solutions back.
fn solve<G, F, C>(
    graph: &G,
    start: &G::State,
    target: &G::State,
    mode: Mode,
    cost: F,
    one: C,
) -> Vec<CostedPair<G::State, G::Edge, [C; 2]>>
where
    G: EdgeGraph,
    F: Fn(&G::Weight) -> [C; 2],
    C: Cost,
{
    if start == target {
        return Vec::new();
    }

    let pairs = Pairs {
        graph,
        start: start.clone(),
        target: target.clone(),
        mode,
        cost,
        one,
    };
    let root = Partial {
        second: false,
        at: start.clone(),
        first_edges: BTreeSet::new(),
        second_edges: BTreeSet::new(),
        first_states: BTreeSet::from([start.clone()]),
        second_states: BTreeSet::new(),
        via: None,
    };
    let goal = |s: &Partial<G::State, G::Edge>| s.second && s.at == *target;

    boa_star(&pairs, ZeroHeuristic, root, goal)
        .solutions
        .into_iter()
        .map(|solution| (unpack(&solution), solution.cost))
        .collect()
}

/// The two paths of a pair-space solution.
fn unpack<S: Clone, E: Clone, C>(solution: &Solution<Partial<S, E>, C>) -> PathPair<S, E> {
    let mut pair = PathPair {
        first: Vec::new(),
        second: Vec::new(),
        first_edges: Vec::new(),
        second_edges: Vec::new(),
    };

    for state in &solution.path {
        let (states, edges) = if state.second {
            (&mut pair.second, &mut pair.second_edges)
        } else {
            (&mut pair.first, &mut pair.first_edges)
        };

        states.push(state.at.clone());
        edges.extend(state.via.clone());
    }

    pair
}
