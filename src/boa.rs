//! BOA*: bi-objective A* search.
//!
//! Hernández, Yeoh, Baier, Zhang, Suazo, Koenig, Salzman: "A Simple and
//! Fast Bi-Objective Search Algorithm", ICAPS 2020.
//!
//! Labels leave the open list in lexicographic `(f1, f2)` order. With a
//! consistent heuristic, a label is then dominated exactly when its `g2` is
//! no smaller than the smallest `g2` already expanded at its state, or its
//! `f2` is no smaller than the `g2` of the last solution found. Both checks
//! are `O(1)` per state, so BOA* needs no per-state frontier.
//!
//! ```mermaid
//! graph TD
//!     pop[Pop min f1, f2] --> pruned{Dominated?}
//!     pruned -- yes --> pop
//!     pruned -- no --> record[g2_min of state = g2]
//!     record --> goal{Goal?}
//!     goal -- yes --> solution[Add solution] --> pop
//!     goal -- no --> expand[Generate successors, prune, push] --> pop
//! ```

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::mem;

use crate::cost::{Cost, add};
use crate::graph::{Graph, Heuristic};
use crate::output::{Output, Solution, Stats};

/// Finds one path for every Pareto-optimal cost vector from `start` to a
/// state that satisfies `is_goal`.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is consistent in both objectives and zero on every goal.
///
/// With [`ZeroHeuristic`](crate::ZeroHeuristic) the search is a
/// bi-objective Dijkstra. Solutions are sorted by the first objective
/// ascending and so by the second strictly descending.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, boa_star};
///
/// /// Two routes from 0 to 2: fast but costly, or slow but cheap.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         match state {
///             0 => out.extend([(1, [1, 5]), (2, [4, 1])]),
///             1 => out.push((2, [1, 5])),
///             _ => {}
///         }
///     }
/// }
///
/// let out = boa_star(&Routes, &ZeroHeuristic, 0, |s| *s == 2);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[2, 10], [4, 1]]);
/// assert_eq!(out.solutions[0].path, [0, 1, 2]);
/// ```
pub fn boa_star<G, H, C, P>(
    graph: &G,
    heuristic: &H,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; 2]>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut search = Search::new(graph, heuristic, is_goal);
    search.push_start(start);

    while let Some(Reverse((f, index))) = search.open.pop() {
        search.process(f, index);
    }

    Output {
        solutions: search.solutions,
        stats: search.stats,
    }
}

/// A search label: one path to a state, stored in the node arena.
struct Node<S, C> {
    /// The state this path ends in.
    state: S,
    /// Cost vector of the path from the start.
    g: [C; 2],
    /// Arena index of the predecessor label; `None` for the start.
    parent: Option<usize>,
}

/// Mutable state of one BOA* run.
struct Search<'a, G: Graph, H, C, P> {
    /// The graph being searched.
    graph: &'a G,
    /// Per-objective lower bounds on the remaining cost.
    heuristic: &'a H,
    /// Goal test.
    is_goal: P,
    /// Every generated label; open-list entries and parents index into it.
    nodes: Vec<Node<G::State, C>>,
    /// Open list as a min-heap on `(f, arena index)`, `f` compared
    /// lexicographically. The index breaks ties deterministically.
    open: BinaryHeap<Reverse<([C; 2], usize)>>,
    /// Smallest `g2` of any label expanded (or recorded as solution) per
    /// state. Absent means "not reached yet".
    g2_min: BTreeMap<G::State, C>,
    /// `g2` of the most recent solution, which is the smallest so far;
    /// `None` until the first solution.
    best_goal_g2: Option<C>,
    /// Solutions in the order found, first objective ascending.
    solutions: Vec<Solution<G::State, [C; 2]>>,
    /// Reused successor buffer, empty between expansions.
    successors: Vec<(G::State, [C; 2])>,
    /// Effort counters.
    stats: Stats,
}

impl<'a, G, H, C, P> Search<'a, G, H, C, P>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    /// Creates an empty search with no labels.
    fn new(graph: &'a G, heuristic: &'a H, is_goal: P) -> Self {
        Self {
            graph,
            heuristic,
            is_goal,
            nodes: Vec::new(),
            open: BinaryHeap::new(),
            g2_min: BTreeMap::new(),
            best_goal_g2: None,
            solutions: Vec::new(),
            successors: Vec::new(),
            stats: Stats::default(),
        }
    }

    /// Pushes the start label, unless the heuristic reports it as a dead end.
    fn push_start(&mut self, start: G::State) {
        if let Some(h) = self.heuristic.estimate(&start) {
            self.push(start, [C::default(); 2], h, None);
        }
    }

    /// Handles one label taken from the open list: prune, record a
    /// solution, or expand.
    fn process(&mut self, f: [C; 2], index: usize) {
        let node = &self.nodes[index];
        if self.is_dominated(&node.state, node.g[1], f[1]) {
            return;
        }

        let g2 = node.g[1];
        self.g2_min.insert(node.state.clone(), g2);

        if (self.is_goal)(&node.state) {
            self.best_goal_g2 = Some(g2);
            let solution = self.solution(index);
            self.solutions.push(solution);
        } else {
            self.expand(index);
        }
    }

    /// Generates the successors of a label and pushes those that survive
    /// pruning.
    fn expand(&mut self, index: usize) {
        self.stats.expanded += 1;

        let mut successors = mem::take(&mut self.successors);
        let g = self.nodes[index].g;
        self.graph
            .successors(&self.nodes[index].state, &mut successors);

        for (state, weight) in successors.drain(..) {
            let Some(h) = self.heuristic.estimate(&state) else {
                continue;
            };

            let g_next = add(g, weight);
            if !self.is_dominated(&state, g_next[1], g_next[1] + h[1]) {
                self.push(state, g_next, h, Some(index));
            }
        }

        self.successors = successors;
    }

    /// Returns `true` if a label at `state` with second costs `g2` and `f2`
    /// is weakly dominated by an expanded label or by a solution.
    fn is_dominated(&self, state: &G::State, g2: C, f2: C) -> bool {
        self.g2_min.get(state).is_some_and(|&min| g2 >= min)
            || self.best_goal_g2.is_some_and(|best| f2 >= best)
    }

    /// Adds a label to the arena and the open list.
    fn push(&mut self, state: G::State, g: [C; 2], h: [C; 2], parent: Option<usize>) {
        let index = self.nodes.len();
        self.nodes.push(Node { state, g, parent });
        self.open.push(Reverse((add(g, h), index)));
        self.stats.generated += 1;
    }

    /// Rebuilds the path ending in the label at `index`.
    fn solution(&self, index: usize) -> Solution<G::State, [C; 2]> {
        let mut path = Vec::new();
        let mut current = Some(index);

        while let Some(i) = current {
            path.push(self.nodes[i].state.clone());
            current = self.nodes[i].parent;
        }

        path.reverse();

        Solution {
            cost: self.nodes[index].g,
            path,
        }
    }
}
