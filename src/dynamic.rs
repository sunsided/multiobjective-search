//! Dynamic multi-objective search: a graph whose edges change between
//! queries, and a search that repairs its labels instead of starting over.
//!
//! The dynamic MOS setting of the survey by Salzman et al. (arXiv
//! 2510.25504, Section 3.2), with the incremental repair of LPA* and D*
//! Lite (Koenig and Likhachev) as carried to multiple objectives by
//! multi-objective path-based D* Lite (Ren et al. 2022).

use alloc::collections::{BTreeSet, VecDeque};
use alloc::vec::Vec;

use search_core::Cost;

use crate::cost::{add, weakly_dominates};
use crate::graph::{EdgeGraph, Graph, ReverseGraph};
use crate::output::{Output, Solution, Stats};
use crate::pareto::Front;

/// An explicit directed graph over states `0..nodes` whose edges can be
/// added, re-costed and removed. Edge identities are the indices
/// [`DynamicGraph::add_edge`] returns; a removed edge keeps its index.
#[derive(Debug, Clone)]
pub struct DynamicGraph<C, const N: usize> {
    /// Edges by identity: `(from, to, cost)`, `None` once removed.
    edges: Vec<Option<(u32, u32, [C; N])>>,
    /// Outgoing edge identities per state.
    out: Vec<Vec<usize>>,
    /// Incoming edge identities per state.
    into: Vec<Vec<usize>>,
}

impl<C: Cost, const N: usize> DynamicGraph<C, N> {
    /// A graph over `nodes` states without edges.
    ///
    /// # Panics
    ///
    /// Never on 32- and 64-bit targets.
    #[must_use]
    pub fn new(nodes: u32) -> Self {
        let n = usize::try_from(nodes).expect("u32 fits in usize");
        Self {
            edges: Vec::new(),
            out: alloc::vec![Vec::new(); n],
            into: alloc::vec![Vec::new(); n],
        }
    }

    /// Adds an edge and returns its identity.
    ///
    /// # Panics
    ///
    /// Panics if a state is out of range.
    pub fn add_edge(&mut self, from: u32, to: u32, cost: [C; N]) -> usize {
        let id = self.edges.len();
        self.edges.push(Some((from, to, cost)));
        self.out[index(from)].push(id);
        self.into[index(to)].push(id);
        id
    }

    /// The edge `id`, if it exists.
    #[must_use]
    pub fn edge(&self, id: usize) -> Option<(u32, u32, [C; N])> {
        self.edges.get(id).copied().flatten()
    }

    /// The number of states.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.out.len()
    }
}

/// `state` as an index.
fn index(state: u32) -> usize {
    usize::try_from(state).expect("u32 fits in usize")
}

impl<C: Cost, const N: usize> Graph for DynamicGraph<C, N> {
    type State = u32;
    type Weight = [C; N];

    fn successors(&self, state: &u32, out: &mut Vec<(u32, [C; N])>) {
        out.extend(
            self.out[index(*state)]
                .iter()
                .filter_map(|&e| self.edges[e])
                .map(|(_, to, c)| (to, c)),
        );
    }
}

impl<C: Cost, const N: usize> ReverseGraph for DynamicGraph<C, N> {
    fn predecessors(&self, state: &u32, out: &mut Vec<(u32, [C; N])>) {
        out.extend(
            self.into[index(*state)]
                .iter()
                .filter_map(|&e| self.edges[e])
                .map(|(from, _, c)| (from, c)),
        );
    }
}

impl<C: Cost, const N: usize> EdgeGraph for DynamicGraph<C, N> {
    type Edge = usize;

    fn edges(&self, state: &u32, out: &mut Vec<(usize, u32, [C; N])>) {
        out.extend(
            self.out[index(*state)]
                .iter()
                .filter_map(|&e| self.edges[e].map(|(_, to, c)| (e, to, c))),
        );
    }
}

/// One label: a path from the start, in the label tree.
#[derive(Debug, Clone)]
struct Label<C, const N: usize> {
    /// The state it ends in.
    state: u32,
    /// Its cost under the current edge costs.
    cost: [C; N],
    /// The label it extends; `None` for the start.
    parent: Option<usize>,
    /// `false` once dominated or invalidated.
    alive: bool,
    /// Labels that extend this one.
    children: Vec<usize>,
}

/// An incremental multi-objective search on a [`DynamicGraph`]: every
/// state keeps the Pareto set of path costs from the start (label
/// correcting), and edge changes repair only what they touch.
///
/// After [`DynamicSearch::set_cost`], [`DynamicSearch::remove_edge`] or
/// [`DynamicSearch::add_edge`], the labels whose paths use the changed
/// edge, and everything built on them, are dropped; every state that lost a
/// label is re-seeded from the live labels of its predecessors, and label
/// correcting runs from there, as LPA* re-expands only the inconsistent
/// part of its search tree. Labels elsewhere stay. [`DynamicSearch::front`]
/// then equals a fresh search's front.
///
/// Goal states end paths. Edge costs must be non-negative.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{DynamicGraph, DynamicSearch};
///
/// let mut graph = DynamicGraph::new(3);
/// let direct = graph.add_edge(0, 2, [10_u32, 1]);
/// graph.add_edge(0, 1, [1, 5]);
/// graph.add_edge(1, 2, [1, 5]);
///
/// let mut search = DynamicSearch::new(graph, 0, |s: &u32| *s == 2);
/// let costs = |s: &DynamicSearch<_, _, 2>| s.front().solutions.iter().map(|x| x.cost).collect::<Vec<_>>();
/// assert_eq!(costs(&search), [[2, 10], [10, 1]]);
///
/// search.set_cost(direct, [1, 1]); // the direct road got better
/// assert_eq!(costs(&search), [[1, 1]]);
/// ```
pub struct DynamicSearch<C, P, const N: usize> {
    /// The current graph.
    graph: DynamicGraph<C, N>,
    /// Goal test.
    is_goal: P,
    /// Every label ever made.
    labels: Vec<Label<C, N>>,
    /// Live labels per state, mutually non-dominated.
    sets: Vec<Vec<usize>>,
    /// Labels per edge identity that extend along it.
    via: Vec<Vec<usize>>,
    /// Effort counters over all repairs.
    stats: Stats,
}

impl<C: Cost, P: Fn(&u32) -> bool, const N: usize> DynamicSearch<C, P, N> {
    /// Searches `graph` from `start`.
    ///
    /// # Panics
    ///
    /// Panics if `start` is not a state of `graph`.
    pub fn new(graph: DynamicGraph<C, N>, start: u32, is_goal: P) -> Self {
        let mut search = Self {
            sets: alloc::vec![Vec::new(); graph.node_count()],
            via: alloc::vec![Vec::new(); graph.edges.len()],
            graph,
            is_goal,
            labels: Vec::new(),
            stats: Stats::default(),
        };

        let root = search
            .insert(start, [C::ZERO; N], None, None)
            .expect("the first label");
        search.propagate(VecDeque::from([root]));
        search
    }

    /// The current graph.
    pub const fn graph(&self) -> &DynamicGraph<C, N> {
        &self.graph
    }

    /// Effort counters over the initial search and every repair.
    pub const fn stats(&self) -> Stats {
        self.stats
    }

    /// Changes the cost of edge `id` and repairs.
    ///
    /// # Panics
    ///
    /// Panics if the edge does not exist.
    pub fn set_cost(&mut self, id: usize, cost: [C; N]) {
        let edge = self.graph.edges[id].as_mut().expect("an existing edge");
        edge.2 = cost;
        let head = edge.1;
        self.repair(id, head);
    }

    /// Removes edge `id` and repairs.
    ///
    /// # Panics
    ///
    /// Panics if the edge does not exist.
    pub fn remove_edge(&mut self, id: usize) {
        let (_, head, _) = self.graph.edges[id].take().expect("an existing edge");
        self.repair(id, head);
    }

    /// Adds an edge, repairs, and returns its identity.
    pub fn add_edge(&mut self, from: u32, to: u32, cost: [C; N]) -> usize {
        let id = self.graph.add_edge(from, to, cost);
        self.via.push(Vec::new());
        self.repair(id, to);
        id
    }

    /// The current Pareto-optimal goal paths, one per cost, sorted.
    ///
    /// # Panics
    ///
    /// Never: states are indexed by `u32`.
    #[must_use]
    pub fn front(&self) -> Output<u32, [C; N]> {
        let mut ends: Vec<usize> = (0..self.sets.len())
            .filter(|&s| (self.is_goal)(&u32::try_from(s).expect("state index")))
            .flat_map(|s| self.sets[s].iter().copied())
            .collect();
        ends.sort_by_key(|&l| self.labels[l].cost);
        let mut front = Front::new();

        Output {
            solutions: ends
                .into_iter()
                .filter(|&l| front.insert(self.labels[l].cost))
                .map(|l| self.solution(l))
                .collect(),
            stats: self.stats,
        }
    }

    /// Drops every label built on edge `id`, re-seeds the states that lost
    /// labels (and `head`, which a cheaper or new edge may improve) from
    /// their predecessors, and propagates.
    fn repair(&mut self, id: usize, head: u32) {
        let mut affected = BTreeSet::from([head]);
        for root in core::mem::take(&mut self.via[id]) {
            self.kill(root, &mut affected);
        }

        let mut queue = VecDeque::new();
        for &target in &affected {
            for &e in &self.graph.into[index(target)].clone() {
                let Some((from, to, w)) = self.graph.edges[e] else {
                    continue;
                };
                if (self.is_goal)(&from) {
                    continue;
                }

                for l in self.sets[index(from)].clone() {
                    let cost = add(self.labels[l].cost, w);
                    if let Some(new) = self.insert(to, cost, Some(l), Some(e)) {
                        queue.push_back(new);
                    }
                }
            }
        }

        self.propagate(queue);
    }

    /// Kills `root` and every label built on it, recording their states.
    fn kill(&mut self, root: usize, affected: &mut BTreeSet<u32>) {
        let mut stack = alloc::vec![root];
        while let Some(l) = stack.pop() {
            let label = &mut self.labels[l];
            if label.alive {
                label.alive = false;
                let state = label.state;
                self.sets[index(state)].retain(|&k| k != l);
                affected.insert(state);
            }

            stack.append(&mut self.labels[l].children);
        }
    }

    /// Adds a label at `state` unless a live one there weakly dominates it,
    /// dropping the live ones it dominates (but not what they built).
    fn insert(
        &mut self,
        state: u32,
        cost: [C; N],
        parent: Option<usize>,
        via: Option<usize>,
    ) -> Option<usize> {
        let set = &self.sets[index(state)];
        if set
            .iter()
            .any(|&k| weakly_dominates(&self.labels[k].cost, &cost))
        {
            return None;
        }

        let labels = &mut self.labels;
        self.sets[index(state)].retain(|&k| {
            let dominated = weakly_dominates(&cost, &labels[k].cost);
            if dominated {
                labels[k].alive = false;
            }

            !dominated
        });

        let id = self.labels.len();
        self.labels.push(Label {
            state,
            cost,
            parent,
            alive: true,
            children: Vec::new(),
        });
        self.sets[index(state)].push(id);
        if let Some(p) = parent {
            self.labels[p].children.push(id);
        }

        if let Some(e) = via {
            self.via[e].push(id);
        }

        self.stats.generated += 1;
        Some(id)
    }

    /// Label correcting from the labels in `queue`.
    fn propagate(&mut self, mut queue: VecDeque<usize>) {
        while let Some(l) = queue.pop_front() {
            let (state, cost) = (self.labels[l].state, self.labels[l].cost);
            if !self.labels[l].alive || (self.is_goal)(&state) {
                continue;
            }

            self.stats.expanded += 1;
            for &e in &self.graph.out[index(state)].clone() {
                let Some((_, to, w)) = self.graph.edges[e] else {
                    continue;
                };
                if let Some(new) = self.insert(to, add(cost, w), Some(l), Some(e)) {
                    queue.push_back(new);
                }
            }
        }
    }

    /// The path of label `l`.
    fn solution(&self, l: usize) -> Solution<u32, [C; N]> {
        let mut path = Vec::new();
        let mut at = Some(l);
        while let Some(i) = at {
            path.push(self.labels[i].state);
            at = self.labels[i].parent;
        }

        path.reverse();
        Solution {
            cost: self.labels[l].cost,
            path,
        }
    }
}
