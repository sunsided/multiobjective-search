//! The label-setting engine shared by NAMOA*, NAMOA*dr and EMOA*.
//!
//! All three pop labels in lexicographic `f` order and keep, per state, an
//! archive of the `g` vectors already expanded ("closed"). They differ in
//! two choices:
//!
//! - **Archive:** NAMOA* compares full vectors ([`Front`]); NAMOA*dr and
//!   EMOA* drop the first objective ([`TruncatedFront`]), which is sound
//!   only with a consistent heuristic.
//! - **Open frontier:** NAMOA* and NAMOA*dr also keep the `g` vectors of the
//!   labels still on the open list per state and prune new labels against
//!   them eagerly. EMOA* skips that bookkeeping and filters lazily when a
//!   label is popped.
//!
//! ```mermaid
//! graph TD
//!     pop[Pop lexicographically smallest f] --> stale{Removed, or dominated<br/>by closed or solutions?}
//!     stale -- yes --> pop
//!     stale -- no --> close[Archive g as closed]
//!     close --> goal{Goal?}
//!     goal -- yes --> solution[Add solution] --> pop
//!     goal -- no --> expand[Generate successors] --> admit{Dominated by solutions,<br/>closed or open?}
//!     admit -- yes --> pop
//!     admit -- no --> push[Remove open labels it dominates, push] --> pop
//! ```

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::mem;

use crate::cost::{Cost, add, weakly_dominates};
use crate::graph::{Graph, Heuristic};
use crate::output::{Output, Solution, Stats};
use crate::pareto::{Front, TruncatedFront};

/// A set of cost vectors that answers "is this vector weakly dominated?".
///
/// Implemented with full dominance by [`Front`] and with dominance over
/// objectives `1..N` by [`TruncatedFront`].
pub(crate) trait Archive<C, const N: usize> {
    /// Creates an empty archive.
    fn empty() -> Self;

    /// Returns `true` if a stored vector weakly dominates `point` in the
    /// sense of this archive.
    fn weakly_dominates(&self, point: &[C; N]) -> bool;

    /// Adds `point`, dropping stored vectors it makes redundant.
    fn add(&mut self, point: [C; N]);
}

impl<C: Cost, const N: usize> Archive<C, N> for Front<C, N> {
    fn empty() -> Self {
        Self::new()
    }

    fn weakly_dominates(&self, point: &[C; N]) -> bool {
        Front::weakly_dominates(self, point)
    }

    fn add(&mut self, point: [C; N]) {
        self.insert(point);
    }
}

impl<C: Cost, const N: usize> Archive<C, N> for TruncatedFront<C, N> {
    fn empty() -> Self {
        Self::new()
    }

    fn weakly_dominates(&self, point: &[C; N]) -> bool {
        TruncatedFront::weakly_dominates(self, point)
    }

    fn add(&mut self, point: [C; N]) {
        self.insert(point);
    }
}

/// Runs the engine from `start` to the goal set.
///
/// `A` picks full or truncated dominance; `track_open` turns the per-state
/// open frontier on (NAMOA*, NAMOA*dr) or off (EMOA*). The caller states
/// the heuristic requirements of the chosen combination.
pub(crate) fn run<G, H, C, P, A, const N: usize>(
    graph: &G,
    heuristic: &H,
    start: G::State,
    is_goal: P,
    track_open: bool,
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
    A: Archive<C, N>,
{
    let mut search = Search::<G, H, C, P, A, N>::new(graph, heuristic, is_goal, track_open);
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
struct Node<S, C, const N: usize> {
    /// The state this path ends in.
    state: S,
    /// Cost vector of the path from the start.
    g: [C; N],
    /// Arena index of the predecessor label; `None` for the start.
    parent: Option<usize>,
    /// `false` once an open label was found dominated at generation time
    /// of another label; its open-list entry is then skipped.
    alive: bool,
}

/// Per-state bookkeeping.
struct StateLabels<A> {
    /// Arena indices of this state's labels still on the open list. Only
    /// maintained when the open frontier is tracked; mutually
    /// non-dominated.
    open: Vec<usize>,
    /// `g` vectors of the labels expanded (or recorded as solutions) at
    /// this state.
    closed: A,
}

/// Mutable state of one engine run.
struct Search<'a, G: Graph, H, C, P, A, const N: usize> {
    /// The graph being searched.
    graph: &'a G,
    /// Per-objective lower bounds on the remaining cost.
    heuristic: &'a H,
    /// Goal test.
    is_goal: P,
    /// Whether [`StateLabels::open`] is maintained and checked.
    track_open: bool,
    /// Every generated label; open-list entries and parents index into it.
    nodes: Vec<Node<G::State, C, N>>,
    /// Open list as a min-heap on `(f, arena index)`, `f` compared
    /// lexicographically. The index breaks ties deterministically.
    open: BinaryHeap<Reverse<([C; N], usize)>>,
    /// Per-state open and closed labels. Absent means "not reached yet".
    labels: BTreeMap<G::State, StateLabels<A>>,
    /// Costs of the solutions found so far, for filtering by `f`.
    costs: A,
    /// Solutions in the order found, lexicographically ascending.
    solutions: Vec<Solution<G::State, [C; N]>>,
    /// Reused successor buffer, empty between expansions.
    successors: Vec<(G::State, [C; N])>,
    /// Effort counters.
    stats: Stats,
}

impl<'a, G, H, C, P, A, const N: usize> Search<'a, G, H, C, P, A, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
    A: Archive<C, N>,
{
    /// Creates an empty search with no labels.
    fn new(graph: &'a G, heuristic: &'a H, is_goal: P, track_open: bool) -> Self {
        Self {
            graph,
            heuristic,
            is_goal,
            track_open,
            nodes: Vec::new(),
            open: BinaryHeap::new(),
            labels: BTreeMap::new(),
            costs: A::empty(),
            solutions: Vec::new(),
            successors: Vec::new(),
            stats: Stats::default(),
        }
    }

    /// Pushes the start label, unless the heuristic reports it as a dead end.
    fn push_start(&mut self, start: G::State) {
        if let Some(h) = self.heuristic.estimate(&start) {
            self.push(start, [C::default(); N], h, None);
        }
    }

    /// Handles one label taken from the open list: drop it if stale, else
    /// close it and record a solution or expand.
    fn process(&mut self, f: [C; N], index: usize) {
        if !self.nodes[index].alive {
            return;
        }

        self.leave_open(index);
        if self.is_stale(f, index) {
            return;
        }

        let node = &self.nodes[index];
        let g = node.g;
        let goal = (self.is_goal)(&node.state);
        self.labels_mut(index).closed.add(g);

        if goal {
            self.costs.add(g);
            let solution = self.solution(index);
            self.solutions.push(solution);
        } else {
            self.expand(index);
        }
    }

    /// Returns `true` if the popped label at `index` is weakly dominated by
    /// a solution (by `f`) or by a closed label at its state (by `g`).
    fn is_stale(&self, f: [C; N], index: usize) -> bool {
        let node = &self.nodes[index];

        self.costs.weakly_dominates(&f)
            || self
                .labels
                .get(&node.state)
                .is_some_and(|labels| labels.closed.weakly_dominates(&node.g))
    }

    /// Removes the label at `index` from its state's open frontier.
    fn leave_open(&mut self, index: usize) {
        if self.track_open {
            self.labels_mut(index).open.retain(|&i| i != index);
        }
    }

    /// The bookkeeping of the state of the label at `index`, created on
    /// first use.
    fn labels_mut(&mut self, index: usize) -> &mut StateLabels<A> {
        self.labels
            .entry(self.nodes[index].state.clone())
            .or_insert_with(|| StateLabels {
                open: Vec::new(),
                closed: A::empty(),
            })
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
            if self.admits(&state, g_next, add(g_next, h)) {
                self.push(state, g_next, h, Some(index));
            }
        }

        self.successors = successors;
    }

    /// Returns `true` if a new label at `state` with costs `g` and `f` is
    /// not weakly dominated by a solution, a closed label or an open label.
    /// Open labels that the new one dominates are removed as a side effect.
    fn admits(&mut self, state: &G::State, g: [C; N], f: [C; N]) -> bool {
        if self.costs.weakly_dominates(&f) {
            return false;
        }

        let Some(labels) = self.labels.get_mut(state) else {
            return true;
        };

        if labels.closed.weakly_dominates(&g) {
            return false;
        }

        if !self.track_open {
            return true;
        }

        let nodes = &mut self.nodes;
        if labels
            .open
            .iter()
            .any(|&i| weakly_dominates(&nodes[i].g, &g))
        {
            return false;
        }

        labels.open.retain(|&i| {
            let dominated = weakly_dominates(&g, &nodes[i].g);
            if dominated {
                nodes[i].alive = false;
            }

            !dominated
        });

        true
    }

    /// Adds a label to the arena, the open list and its state's open
    /// frontier.
    fn push(&mut self, state: G::State, g: [C; N], h: [C; N], parent: Option<usize>) {
        let index = self.nodes.len();
        self.nodes.push(Node {
            state,
            g,
            parent,
            alive: true,
        });
        self.open.push(Reverse((add(g, h), index)));
        self.stats.generated += 1;

        if self.track_open {
            self.labels_mut(index).open.push(index);
        }
    }

    /// Rebuilds the path ending in the label at `index`.
    fn solution(&self, index: usize) -> Solution<G::State, [C; N]> {
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
