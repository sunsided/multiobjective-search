//! Disjoint and dissimilar paths on graphs with edge identities.
//!
//! Suurballe 1974 and Suurballe and Tarjan 1984 (the shortest pair of
//! disjoint paths, the form Bhandari 1999 popularized), the bicriteria
//! disjoint pairs of Clímaco and Pascoal 2009 and the dissimilar paths of
//! Moghanni, Pascoal and Godinho 2022, as surveyed by Craveirinha et al.,
//! Algorithms 17(6):222, 2024, Section 2.2.3(g).

mod pairs;

use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};

use search_core::{Cost, WideMul};

use crate::graph::{EdgeGraph, Graph};

pub use pairs::{SharedPair, disjoint_pair_front, maximally_disjoint_pairs};

/// A disjoint pair with its total cost.
pub type CostedPair<S, E, C> = (PathPair<S, E>, C);

/// A path as its states, its edges and its cost.
pub type RankedPath<S, E, C> = (Vec<S>, Vec<E>, C);

/// Which elements two paths must not share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Disjointness {
    /// No common edge.
    Edges,
    /// No common state other than the two ends.
    States,
}

/// Two disjoint paths between the same ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathPair<S, E> {
    /// The states of the first path, start and target included.
    pub first: Vec<S>,
    /// The states of the second path.
    pub second: Vec<S>,
    /// The edges of the first path, in order.
    pub first_edges: Vec<E>,
    /// The edges of the second path, in order.
    pub second_edges: Vec<E>,
}

/// The pair of disjoint paths from `start` to `target` with the smallest
/// total scalar cost `cost(weight)`, by Suurballe's algorithm; `None` if
/// no two such paths exist.
///
/// A first Dijkstra gives distances `d` and a shortest path `P1`. Every
/// other edge gets the reduced cost `d(u) + c - d(v) >= 0`, and the edges
/// of `P1` are replaced by zero-cost reversed copies; a second Dijkstra on
/// that residual graph gives `P2`. Dropping the edges `P2` crosses
/// backwards from `P1 ∪ P2` leaves the cheapest pair. For
/// [`Disjointness::States`] every state but the ends is split into an
/// entry and an exit joined by one edge, which makes state-disjoint pairs
/// edge-disjoint. Zero-cost cycles left in the union are cut.
///
/// Requirements, not checked at run time: non-negative costs and a finite
/// reachable graph.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Disjointness, EdgeGraph, Graph, disjoint_pair};
///
/// /// The trap graph: the shortest path 0-1-2-3 blocks any disjoint second.
/// struct Trap;
///
/// const EDGES: [(u8, u8, u32); 5] = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 2, 2), (1, 3, 2)];
///
/// impl Graph for Trap {
///     type State = u8;
///     type Weight = u32;
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, u32)>) {
///         out.extend(EDGES.iter().filter(|e| e.0 == *state).map(|e| (e.1, e.2)));
///     }
/// }
///
/// impl EdgeGraph for Trap {
///     type Edge = usize;
///
///     fn edges(&self, state: &u8, out: &mut Vec<(usize, u8, u32)>) {
///         out.extend(EDGES.iter().enumerate().filter(|(_, e)| e.0 == *state).map(|(i, e)| (i, e.1, e.2)));
///     }
/// }
///
/// let (pair, total) = disjoint_pair(&Trap, 0, 3, |w: &u32| *w, Disjointness::Edges).unwrap();
/// assert_eq!(total, 6);
/// assert_eq!((pair.first, pair.second), (vec![0, 1, 3], vec![0, 2, 3]));
/// ```
pub fn disjoint_pair<G, C, F>(
    graph: &G,
    start: G::State,
    target: G::State,
    cost: F,
    kind: Disjointness,
) -> Option<CostedPair<G::State, G::Edge, C>>
where
    G: EdgeGraph,
    C: Cost + WideMul,
    F: Fn(&G::Weight) -> C,
{
    match kind {
        Disjointness::Edges => {
            let net = Network::explore(graph, start, |w| cost(w));
            let (pair, total) = net.suurballe(&target)?;
            Some((pair, total))
        }
        Disjointness::States => {
            let split = Split {
                graph,
                start: start.clone(),
                target: target.clone(),
            };
            let cost = |w: &Option<G::Weight>| w.as_ref().map_or(C::ZERO, &cost);
            let net = Network::explore(&split, (start, true), cost);
            let (pair, total) = net.suurballe(&(target, false))?;
            Some((unsplit(&pair), total))
        }
    }
}

/// The explicit reachable graph, with dense state indices (the start is 0).
struct Network<S, E, C> {
    /// States by index.
    states: Vec<S>,
    /// Index by state.
    index: BTreeMap<S, usize>,
    /// Edges `(identity, from, to, cost)`.
    arcs: Vec<(E, usize, usize, C)>,
    /// Outgoing edge indices per state.
    out: Vec<Vec<usize>>,
}

/// An edge of the residual graph: an original edge, or one of `P1`
/// reversed.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Step {
    /// Along edge `.0`.
    Forward(usize),
    /// Against edge `.0`.
    Backward(usize),
}

impl<S: Clone + Ord, E: Clone, C: Cost + WideMul> Network<S, E, C> {
    /// Explores everything reachable from `start`.
    fn explore<G: EdgeGraph<State = S, Edge = E>>(
        graph: &G,
        start: S,
        cost: impl Fn(&G::Weight) -> C,
    ) -> Self {
        let mut net = Self {
            states: Vec::new(),
            index: BTreeMap::new(),
            arcs: Vec::new(),
            out: Vec::new(),
        };
        net.add(start);
        let mut buffer = Vec::new();
        let mut i = 0;

        while i < net.states.len() {
            graph.edges(&net.states[i].clone(), &mut buffer);
            for (edge, to, weight) in buffer.drain(..) {
                let j = net.add(to);
                if j != i {
                    net.out[i].push(net.arcs.len());
                    net.arcs.push((edge, i, j, cost(&weight)));
                }
            }

            i += 1;
        }

        net
    }

    /// The index of `state`, added if new.
    fn add(&mut self, state: S) -> usize {
        if let Some(&i) = self.index.get(&state) {
            return i;
        }

        self.index.insert(state.clone(), self.states.len());
        self.states.push(state);
        self.out.push(Vec::new());
        self.states.len() - 1
    }

    /// Dijkstra from state 0 over `steps(u)`, each `(step, to, cost)`.
    fn dijkstra(
        &self,
        steps: impl Fn(usize) -> Vec<(Step, usize, C)>,
    ) -> (Vec<Option<C>>, Vec<Option<Step>>) {
        let n = self.states.len();
        let (mut dist, mut parent) = (alloc::vec![None; n], alloc::vec![None; n]);
        let mut done = alloc::vec![false; n];
        let mut heap = BinaryHeap::from([Reverse((C::ZERO, 0_usize, None::<Step>))]);

        while let Some(Reverse((d, u, via))) = heap.pop() {
            if done[u] {
                continue;
            }

            done[u] = true;
            dist[u] = Some(d);
            parent[u] = via;
            for (step, v, c) in steps(u) {
                if !done[v] {
                    heap.push(Reverse((d + c, v, Some(step))));
                }
            }
        }

        (dist, parent)
    }

    /// The tail of a residual step.
    fn tail(&self, step: Step) -> usize {
        match step {
            Step::Forward(a) => self.arcs[a].1,
            Step::Backward(a) => self.arcs[a].2,
        }
    }

    /// The steps from state 0 to `to` along `parent`.
    fn trace(&self, parent: &[Option<Step>], to: usize) -> Vec<Step> {
        let mut steps = Vec::new();
        let mut at = to;
        while let Some(step) = parent[at] {
            steps.push(step);
            at = self.tail(step);
        }

        steps.reverse();
        steps
    }

    /// Suurballe's algorithm from state 0 to `target`.
    fn suurballe(&self, target: &S) -> Option<(PathPair<S, E>, C)> {
        let t = *self.index.get(target)?;
        let original = |u: usize| {
            self.out[u]
                .iter()
                .map(|&a| (Step::Forward(a), self.arcs[a].2, self.arcs[a].3))
                .collect()
        };
        let (dist, parent) = self.dijkstra(original);
        dist[t]?;
        if t == 0 {
            return None;
        }

        let first: Vec<usize> = self
            .trace(&parent, t)
            .into_iter()
            .map(|s| match s {
                Step::Forward(a) | Step::Backward(a) => a,
            })
            .collect();
        let on_first: BTreeSet<usize> = first.iter().copied().collect();

        // Incoming P1 edges per head, to add their reversals.
        let mut reversed: Vec<Vec<usize>> = alloc::vec![Vec::new(); self.states.len()];
        for &a in &first {
            reversed[self.arcs[a].2].push(a);
        }

        let residual = |u: usize| {
            let mut steps: Vec<(Step, usize, C)> = self.out[u]
                .iter()
                .filter(|a| !on_first.contains(a))
                .map(|&a| {
                    let (_, from, to, c) = self.arcs[a];
                    let reduced =
                        (dist[from].expect("reachable") + c).abs_diff(dist[to].expect("reachable"));
                    (Step::Forward(a), to, reduced)
                })
                .collect();
            steps.extend(
                reversed[u]
                    .iter()
                    .map(|&a| (Step::Backward(a), self.arcs[a].1, C::ZERO)),
            );
            steps
        };
        let (dist2, parent2) = self.dijkstra(residual);
        dist2[t]?;

        let mut used = on_first.clone();
        for step in self.trace(&parent2, t) {
            match step {
                Step::Forward(a) => used.insert(a),
                Step::Backward(a) => used.remove(&a),
            };
        }

        let mut remaining: Vec<Vec<usize>> = alloc::vec![Vec::new(); self.states.len()];
        for &a in &used {
            remaining[self.arcs[a].1].push(a);
        }

        let one = self.walk(&mut remaining, t);
        let two = self.walk(&mut remaining, t);
        let total = one
            .iter()
            .chain(&two)
            .fold(C::ZERO, |sum, &a| sum + self.arcs[a].3);
        let states = |arcs: &[usize]| {
            core::iter::once(self.states[0].clone())
                .chain(arcs.iter().map(|&a| self.states[self.arcs[a].2].clone()))
                .collect()
        };
        let edges = |arcs: &[usize]| arcs.iter().map(|&a| self.arcs[a].0.clone()).collect();

        let mut pair = PathPair {
            first: states(&one),
            second: states(&two),
            first_edges: edges(&one),
            second_edges: edges(&two),
        };
        if pair.second < pair.first {
            core::mem::swap(&mut pair.first, &mut pair.second);
            core::mem::swap(&mut pair.first_edges, &mut pair.second_edges);
        }

        Some((pair, total))
    }

    /// Follows unused edges of `remaining` from state 0 to `t`, cutting any
    /// cycle it closes; returns the edges.
    fn walk(&self, remaining: &mut [Vec<usize>], t: usize) -> Vec<usize> {
        let mut arcs = Vec::new();
        let mut position = BTreeMap::from([(0_usize, 0_usize)]);
        let mut at = 0;

        while at != t {
            let a = remaining[at]
                .pop()
                .expect("the union is a flow of two units");
            at = self.arcs[a].2;
            arcs.push(a);
            if let Some(&p) = position.get(&at) {
                // A cycle: drop it.
                for dropped in arcs.drain(p..) {
                    position.remove(&self.arcs[dropped].2);
                }
                position.insert(at, p);
            } else {
                position.insert(at, arcs.len());
            }
        }

        arcs
    }
}

/// `G` with every state but the ends split into an entry `(s, false)` and an
/// exit `(s, true)`, joined by an edge of no weight.
struct Split<'a, G: Graph> {
    /// The underlying graph.
    graph: &'a G,
    /// The start, which only has an exit.
    start: G::State,
    /// The target, which only has an entry.
    target: G::State,
}

/// An edge of [`Split`]: an original edge, or a state's inner edge.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum SplitEdge<E, S> {
    /// An original edge, from an exit to an entry.
    Edge(E),
    /// From the entry to the exit of a state.
    Inner(S),
}

impl<G: EdgeGraph> Graph for Split<'_, G> {
    type State = (G::State, bool);
    type Weight = Option<G::Weight>;

    fn successors(&self, state: &Self::State, out: &mut Vec<(Self::State, Self::Weight)>) {
        let mut edges = Vec::new();
        self.edges(state, &mut edges);
        out.extend(edges.into_iter().map(|(_, s, w)| (s, w)));
    }
}

impl<G: EdgeGraph> EdgeGraph for Split<'_, G> {
    type Edge = SplitEdge<G::Edge, G::State>;

    fn edges(&self, state: &Self::State, out: &mut Vec<(Self::Edge, Self::State, Self::Weight)>) {
        let (s, exit) = state;
        if !exit {
            if *s != self.target {
                out.push((SplitEdge::Inner(s.clone()), (s.clone(), true), None));
            }

            return;
        }

        let mut edges = Vec::new();
        self.graph.edges(s, &mut edges);
        for (e, to, w) in edges {
            if to != self.start {
                out.push((SplitEdge::Edge(e), (to, false), Some(w)));
            }
        }
    }
}

/// Maps a pair on [`Split`] back to the original graph.
fn unsplit<S: Clone + PartialEq + Ord, E: Clone>(
    pair: &PathPair<(S, bool), SplitEdge<E, S>>,
) -> PathPair<S, E> {
    let states = |path: &[(S, bool)]| {
        let mut out: Vec<S> = Vec::new();
        for (s, _) in path {
            if out.last() != Some(s) {
                out.push(s.clone());
            }
        }

        out
    };
    let edges = |edges: &[SplitEdge<E, S>]| {
        edges
            .iter()
            .filter_map(|e| match e {
                SplitEdge::Edge(e) => Some(e.clone()),
                SplitEdge::Inner(_) => None,
            })
            .collect::<Vec<_>>()
    };

    let mut out = PathPair {
        first: states(&pair.first),
        second: states(&pair.second),
        first_edges: edges(&pair.first_edges),
        second_edges: edges(&pair.second_edges),
    };
    if out.second < out.first {
        core::mem::swap(&mut out.first, &mut out.second);
        core::mem::swap(&mut out.first_edges, &mut out.second_edges);
    }

    out
}

/// The disjoint pairs whose total cost vectors are the vertices of the
/// convex hull of the bi-objective front of disjoint pairs: the extreme
/// supported pairs, sorted by total cost.
///
/// The bicriteria disjoint path pair problem (Clímaco and Pascoal 2009)
/// asks for the pairs whose summed costs `(c1, c2)` are non-dominated.
/// This finds its extreme supported points by dichotomic search, each step
/// one [`disjoint_pair`] run on the weighted sum `w1 · c1 + w2 · c2`
/// (lexicographic ends by large weights), in `u128` arithmetic: exact while
/// the costs summed over all edges stay below `2^63`. The non-supported
/// pairs between them are not searched; finding them needs a ranking of
/// pairs, which this crate does not implement.
///
/// Requirements as for [`disjoint_pair`].
///
/// # Panics
///
/// Panics if a run that must succeed (a pair is known to exist) fails,
/// which only a graph whose edges change between calls can cause.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Disjointness, EdgeGraph, Graph, supported_disjoint_pairs};
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
/// let pairs = supported_disjoint_pairs(&Routes, &0, &1, Disjointness::Edges);
/// let costs: Vec<_> = pairs.iter().map(|(_, c)| *c).collect();
/// // (10, 10) lies on the segment between the two: supported, not extreme.
/// assert_eq!(costs, [[6, 14], [14, 6]]);
/// ```
pub fn supported_disjoint_pairs<G, C>(
    graph: &G,
    start: &G::State,
    target: &G::State,
    kind: Disjointness,
) -> Vec<CostedPair<G::State, G::Edge, [C; 2]>>
where
    G: EdgeGraph<Weight = [C; 2]>,
    C: Cost + Into<u64>,
{
    let solve = |w: [u128; 2]| {
        let cost =
            move |c: &[C; 2]| w[0] * u128::from(c[0].into()) + w[1] * u128::from(c[1].into());
        disjoint_pair(graph, start.clone(), target.clone(), cost, kind).map(|(pair, _)| {
            let total = pair_cost(graph, &pair);
            (pair, total)
        })
    };

    // Large enough that one unit of the main objective outweighs the other.
    let big = u128::from(u64::MAX) + 1;
    let Some(left) = solve([big, 1]) else {
        return Vec::new();
    };
    let right = solve([1, big]).expect("a pair exists");
    if left.1 == right.1 {
        return alloc::vec![left];
    }

    let mut found = alloc::vec![left];
    let mut pending = alloc::vec![right];
    while let Some(b) = pending.pop() {
        let a = found.last().expect("starts with the left end").1;
        let w = [
            u128::from(a[1].into() - b.1[1].into()),
            u128::from(b.1[0].into() - a[0].into()),
        ];
        let p = solve(w).expect("a pair exists");
        let value = |c: [C; 2]| w[0] * u128::from(c[0].into()) + w[1] * u128::from(c[1].into());
        if value(p.1) < value(a) {
            pending.push(b);
            pending.push(p);
        } else {
            found.push(b);
        }
    }

    found
}

/// The summed weights of both paths of `pair`.
fn pair_cost<G, C>(graph: &G, pair: &PathPair<G::State, G::Edge>) -> [C; 2]
where
    G: EdgeGraph<Weight = [C; 2]>,
    C: Cost,
{
    let mut total = [C::ZERO; 2];
    for (path, edges) in [
        (&pair.first, &pair.first_edges),
        (&pair.second, &pair.second_edges),
    ] {
        let mut buffer = Vec::new();
        for (state, edge) in path.iter().zip(edges) {
            graph.edges(state, &mut buffer);
            let w = buffer
                .iter()
                .find(|(e, _, _)| e == edge)
                .expect("the edge leaves its state")
                .2;
            total = [total[0] + w[0], total[1] + w[1]];
            buffer.clear();
        }
    }

    total
}

/// Up to `k` loopless paths from `start` to a goal in order of the scalar
/// `cost(weight)`, each sharing at most `max_shared` edges with every
/// earlier one: the greedy k-shortest dissimilar paths.
///
/// Paths are ranked best-first by `(cost, edge sequence)`, so equal costs
/// are taken in a fixed order, and a path is kept when it is dissimilar
/// enough from all kept ones (in the spirit of Moghanni et al. 2022, who
/// trade cost against dissimilarity). Each result is `(states, edges,
/// cost)`. Exponential in the worst case, like every ranking.
///
/// Requirements: non-negative costs; goals end paths.
///
/// # Panics
///
/// Never: partial paths are never empty.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{EdgeGraph, Graph, dissimilar_paths};
///
/// /// 0 → 1 by two edges, then 1 → 2 by two edges.
/// struct Ladder;
///
/// const EDGES: [(u8, u8, u32); 4] = [(0, 1, 1), (0, 1, 2), (1, 2, 1), (1, 2, 2)];
///
/// impl Graph for Ladder {
///     type State = u8;
///     type Weight = u32;
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, u32)>) {
///         out.extend(EDGES.iter().filter(|e| e.0 == *state).map(|e| (e.1, e.2)));
///     }
/// }
///
/// impl EdgeGraph for Ladder {
///     type Edge = usize;
///
///     fn edges(&self, state: &u8, out: &mut Vec<(usize, u8, u32)>) {
///         out.extend(EDGES.iter().enumerate().filter(|(_, e)| e.0 == *state).map(|(i, e)| (i, e.1, e.2)));
///     }
/// }
///
/// // Fully disjoint: the cheapest path, then the one sharing nothing.
/// let paths = dissimilar_paths(&Ladder, 0, |s| *s == 2, |w: &u32| *w, 3, 0);
/// let costs: Vec<_> = paths.iter().map(|p| p.2).collect();
/// assert_eq!(costs, [2, 4]);
/// ```
pub fn dissimilar_paths<G, C, P, F>(
    graph: &G,
    start: G::State,
    is_goal: P,
    cost: F,
    k: usize,
    max_shared: usize,
) -> Vec<RankedPath<G::State, G::Edge, C>>
where
    G: EdgeGraph,
    C: Cost,
    P: Fn(&G::State) -> bool,
    F: Fn(&G::Weight) -> C,
{
    let mut kept: Vec<RankedPath<G::State, G::Edge, C>> = Vec::new();
    let mut open = BinaryHeap::new();
    open.push(Reverse(Partial {
        cost: C::ZERO,
        edges: Vec::new(),
        states: alloc::vec![start],
    }));
    let mut buffer = Vec::new();

    while let Some(Reverse(partial)) = open.pop() {
        if kept.len() >= k {
            break;
        }

        let here = partial.states.last().expect("never empty").clone();
        if is_goal(&here) {
            let similar = kept.iter().any(|(_, edges, _)| {
                partial.edges.iter().filter(|e| edges.contains(e)).count() > max_shared
            });
            if !similar {
                kept.push((partial.states, partial.edges, partial.cost));
            }

            continue;
        }

        graph.edges(&here, &mut buffer);
        for (edge, to, weight) in buffer.drain(..) {
            if partial.states.contains(&to) {
                continue;
            }

            let mut next = Partial {
                cost: partial.cost + cost(&weight),
                edges: partial.edges.clone(),
                states: partial.states.clone(),
            };
            next.edges.push(edge);
            next.states.push(to);
            open.push(Reverse(next));
        }
    }

    kept
}

/// A loopless partial path, ordered by cost and then edge sequence.
struct Partial<S, E, C> {
    /// Cost so far.
    cost: C,
    /// Edges so far.
    edges: Vec<E>,
    /// States so far; never empty.
    states: Vec<S>,
}

impl<S, E: Ord, C: Ord> PartialEq for Partial<S, E, C> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<S, E: Ord, C: Ord> Eq for Partial<S, E, C> {}

impl<S, E: Ord, C: Ord> PartialOrd for Partial<S, E, C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S, E: Ord, C: Ord> Ord for Partial<S, E, C> {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.cost, &self.edges).cmp(&(&other.cost, &other.edges))
    }
}
