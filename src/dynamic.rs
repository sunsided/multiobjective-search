//! Dynamic multi-objective search: a graph whose edges change between
//! queries, and a search that repairs its labels instead of starting over.
//!
//! The dynamic MOS setting of the survey by Salzman et al. (arXiv
//! 2510.25504, Section 3.2), with the incremental repair of LPA* and D*
//! Lite (Koenig and Likhachev) as carried to multiple objectives by
//! multi-objective path-based D* Lite (Ren et al. 2022). Like D* Lite, the
//! search is guided by a heuristic: labels are expanded in `f` order and
//! pruned against the goal front, and a repair that worsens the front
//! brings the pruned labels back.

use alloc::collections::{BTreeSet, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::{Cost, Heuristic};

use crate::cost::{add, weakly_dominates};
use crate::graph::{EdgeGraph, Graph, ReverseGraph, ZeroHeuristic};
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
    /// `cost` plus the heuristic estimate at `state`; `None` when the
    /// heuristic proves `state` a dead end.
    f: Option<[C; N]>,
    /// The label it extends; `None` for the start.
    parent: Option<usize>,
    /// `false` once dominated or invalidated.
    alive: bool,
    /// `true` once extended along every out-edge of `state`; repairs then
    /// extend it along changed edges. Stays `false` while pruned.
    expanded: bool,
    /// Labels that extend this one.
    children: Vec<usize>,
}

/// The open list: labels by lexicographically smallest `f`, ties by age.
type Open<C, const N: usize> = BinaryHeap<Reverse<([C; N], usize)>>;

/// An incremental multi-objective search on a [`DynamicGraph`]: every
/// state keeps the Pareto set of path costs from the start, and edge
/// changes repair only what they touch.
///
/// After [`DynamicSearch::set_cost`], [`DynamicSearch::remove_edge`] or
/// [`DynamicSearch::add_edge`], the labels whose paths use the changed
/// edge, and everything built on them, are dropped; every state that lost a
/// label is re-seeded from the expanded labels of its predecessors, and the
/// search runs from there, as LPA* re-expands only the inconsistent part of
/// its search tree. Labels elsewhere stay. [`DynamicSearch::front`] then
/// equals a fresh search's front.
///
/// Labels are expanded in lexicographic order of `f = g + h`. A label whose
/// `f` is weakly dominated by a goal label's cost, or whose state the
/// heuristic proves a dead end, is pruned: it stays in its state's Pareto
/// set but is not expanded. When a repair drops a goal label, the front can
/// only get worse, so every pruned label is checked again and expanded once
/// no goal label dominates it any more.
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
pub struct DynamicSearch<C, P, const N: usize, H = ZeroHeuristic> {
    /// The current graph.
    graph: DynamicGraph<C, N>,
    /// Goal test.
    is_goal: P,
    /// Admissible on every graph the search sees.
    heuristic: H,
    /// The goal states, ascending.
    goals: Vec<usize>,
    /// Every label ever made.
    labels: Vec<Label<C, N>>,
    /// Live labels per state, mutually non-dominated.
    sets: Vec<Vec<usize>>,
    /// Labels per edge identity that extend along it.
    via: Vec<Vec<usize>>,
    /// Labels left unexpanded by pruning; may hold dead ones.
    deferred: Vec<usize>,
    /// Effort counters over all repairs.
    stats: Stats,
}

impl<C: Cost, P: Fn(&u32) -> bool, const N: usize> DynamicSearch<C, P, N> {
    /// Searches `graph` from `start` without a heuristic: labels are
    /// pruned only by goal labels that dominate them.
    ///
    /// # Panics
    ///
    /// Panics if `start` is not a state of `graph`.
    pub fn new(graph: DynamicGraph<C, N>, start: u32, is_goal: P) -> Self {
        Self::with_heuristic(graph, ZeroHeuristic, start, is_goal)
    }
}

impl<C, P, H, const N: usize> DynamicSearch<C, P, N, H>
where
    C: Cost,
    P: Fn(&u32) -> bool,
    H: Heuristic<u32, [C; N]>,
{
    /// Searches `graph` from `start`, guided by `heuristic`.
    ///
    /// `heuristic` must be admissible in every objective on every graph the
    /// search will see, not only the first: an estimate never exceeds the
    /// cheapest cost to a goal after any later edge change, and `None` (a
    /// dead end) stays true for good. It need not be consistent. An
    /// [`IdealPoint`](crate::IdealPoint) built on the first graph stays
    /// admissible under removals and cost increases only; for additions and
    /// decreases, use bounds that hold for every edge cost, such as
    /// geometric distances times the cheapest cost per unit.
    ///
    /// # Panics
    ///
    /// Panics if `start` is not a state of `graph`.
    ///
    /// # Examples
    ///
    /// The detour through 1 is pruned while the direct edge exists, and
    /// expanded once it is removed:
    ///
    /// ```
    /// use multiobjective_search::{DynamicGraph, DynamicSearch, heuristic_fn};
    ///
    /// let mut graph = DynamicGraph::new(3);
    /// let direct = graph.add_edge(0, 2, [1_u32, 1]);
    /// graph.add_edge(0, 1, [2, 2]);
    /// graph.add_edge(1, 2, [2, 2]);
    ///
    /// let h = heuristic_fn(|s: &u32| Some([[1, 1], [2, 2], [0, 0]][*s as usize]));
    /// let mut search = DynamicSearch::with_heuristic(graph, h, 0, |s: &u32| *s == 2);
    /// let costs = |s: &DynamicSearch<_, _, 2, _>| s.front().solutions.iter().map(|x| x.cost).collect::<Vec<_>>();
    /// assert_eq!(costs(&search), [[1, 1]]);
    /// assert_eq!(search.stats().expanded, 1); // only the start
    ///
    /// search.remove_edge(direct);
    /// assert_eq!(costs(&search), [[4, 4]]);
    /// assert_eq!(search.stats().expanded, 2);
    /// ```
    pub fn with_heuristic(graph: DynamicGraph<C, N>, heuristic: H, start: u32, is_goal: P) -> Self {
        let goals = (0..graph.node_count())
            .filter(|&s| is_goal(&u32::try_from(s).expect("state index")))
            .collect();
        let mut search = Self {
            sets: alloc::vec![Vec::new(); graph.node_count()],
            via: alloc::vec![Vec::new(); graph.edges.len()],
            graph,
            is_goal,
            heuristic,
            goals,
            labels: Vec::new(),
            deferred: Vec::new(),
            stats: Stats::default(),
        };

        let root = search
            .insert(start, [C::ZERO; N], None, None)
            .expect("the first label");
        let mut open = Open::new();
        search.enqueue(root, &mut open);
        search.propagate(open);
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
    #[must_use]
    pub fn front(&self) -> Output<u32, [C; N]> {
        let mut ends: Vec<usize> = self
            .goals
            .iter()
            .flat_map(|&s| self.sets[s].iter().copied())
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
    /// their predecessors, re-queues the pruned labels if a goal label was
    /// lost, and propagates.
    fn repair(&mut self, id: usize, head: u32) {
        let mut affected = BTreeSet::from([head]);
        let mut lost_goal = false;
        for root in core::mem::take(&mut self.via[id]) {
            lost_goal |= self.kill(root, &mut affected);
        }

        let mut open = Open::new();
        for &target in &affected {
            self.reseed(target, &mut open);
        }

        // shortcut: re-queues every deferred label, not only those the lost
        // goal labels dominated; index them by dominating goal if repairs get slow.
        if lost_goal {
            for l in core::mem::take(&mut self.deferred) {
                if self.labels[l].alive {
                    self.enqueue(l, &mut open);
                }
            }
        }

        self.propagate(open);
    }

    /// Extends the expanded labels of every predecessor of `target` along
    /// its edge into `target`.
    fn reseed(&mut self, target: u32, open: &mut Open<C, N>) {
        for &e in &self.graph.into[index(target)].clone() {
            let Some((from, to, w)) = self.graph.edges[e] else {
                continue;
            };
            if (self.is_goal)(&from) {
                continue;
            }

            for l in self.sets[index(from)].clone() {
                if !self.labels[l].expanded {
                    continue;
                }

                let cost = add(self.labels[l].cost, w);
                if let Some(new) = self.insert(to, cost, Some(l), Some(e)) {
                    self.enqueue(new, open);
                }
            }
        }
    }

    /// Kills `root` and every label built on it, recording their states;
    /// `true` if one of them was a live goal label.
    fn kill(&mut self, root: usize, affected: &mut BTreeSet<u32>) -> bool {
        let mut lost_goal = false;
        let mut stack = alloc::vec![root];
        while let Some(l) = stack.pop() {
            let label = &mut self.labels[l];
            if label.alive {
                label.alive = false;
                let state = label.state;
                self.sets[index(state)].retain(|&k| k != l);
                affected.insert(state);
                lost_goal |= (self.is_goal)(&state);
            }

            stack.append(&mut self.labels[l].children);
        }

        lost_goal
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
        let f = self.heuristic.estimate(&state).map(|h| add(cost, h));
        self.labels.push(Label {
            state,
            cost,
            f,
            parent,
            alive: true,
            expanded: false,
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

    /// Puts label `l` on `open`, or defers it if its state is a dead end.
    fn enqueue(&mut self, l: usize, open: &mut Open<C, N>) {
        match self.labels[l].f {
            Some(f) => open.push(Reverse((f, l))),
            None => self.deferred.push(l),
        }
    }

    /// `true` if a label with estimate `f` cannot lead to a new goal cost:
    /// a live goal label's cost weakly dominates `f`.
    ///
    /// shortcut: scans every goal label per pop; keep a [`Front`] of goal
    /// costs if fronts grow large.
    fn pruned(&self, f: &[C; N]) -> bool {
        self.goals
            .iter()
            .flat_map(|&s| &self.sets[s])
            .any(|&k| weakly_dominates(&self.labels[k].cost, f))
    }

    /// Expands the labels on `open` in `f` order, deferring pruned ones.
    fn propagate(&mut self, mut open: Open<C, N>) {
        while let Some(Reverse((f, l))) = open.pop() {
            let label = &self.labels[l];
            if !label.alive || label.expanded || (self.is_goal)(&label.state) {
                continue;
            }

            if self.pruned(&f) {
                self.deferred.push(l);
                continue;
            }

            self.expand(l, &mut open);
        }
    }

    /// Extends label `l` along every out-edge of its state.
    fn expand(&mut self, l: usize, open: &mut Open<C, N>) {
        let (state, cost) = (self.labels[l].state, self.labels[l].cost);
        self.labels[l].expanded = true;
        self.stats.expanded += 1;
        for &e in &self.graph.out[index(state)].clone() {
            let Some((_, to, w)) = self.graph.edges[e] else {
                continue;
            };
            if let Some(new) = self.insert(to, add(cost, w), Some(l), Some(e)) {
                self.enqueue(new, open);
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
