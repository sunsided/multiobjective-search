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

use search_core::{Cost, Heuristic};

use crate::cost::add;
use crate::epsilon::Epsilon;
use crate::graph::Graph;
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
/// let out = boa_star(&Routes, ZeroHeuristic, 0, |s| *s == 2);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[2, 10], [4, 1]]);
/// assert_eq!(out.solutions[0].path, [0, 1, 2]);
/// ```
pub fn boa_star<G, H, C, P>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; 2]>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    run(graph, heuristic, start, is_goal, Tolerance::EXACT)
}

/// [`boa_star`] with an additive tolerance `slack` on the second
/// objective: returns a subset of the Pareto-optimal solutions that covers
/// every Pareto-optimal cost vector within `slack`.
///
/// Guarantees, under the requirements of [`boa_star`]:
///
/// - every returned solution is Pareto-optimal, and they are sorted as in
///   [`boa_star`];
/// - for every Pareto-optimal cost vector `s` some returned solution `r`
///   has `r[0] <= s[0]` and `r[1] <= s[1] + slack`.
///
/// `slack = 0` is exactly [`boa_star`]. The only change is the pruning by
/// solutions: a label is dropped once `f2 + slack` reaches the second cost
/// of the last solution, so larger tolerances expand fewer labels. The
/// first objective is never relaxed.
///
/// This is an additive variant of the multiplicative
/// [`boa_star_eps`] (BOA*ε); [`pp_a_star`](crate::pp_a_star) also relaxes the
/// first objective.
///
/// `f2 + slack` must not overflow `C`: a slack near `C::MAX` panics in
/// debug builds and wraps in release builds.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, boa_star_approx};
///
/// /// Three routes from 0 to 1 whose second costs differ by one each.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [1, 3]), (1, [2, 2]), (1, [3, 1])]);
///         }
///     }
/// }
///
/// let out = boa_star_approx(&Routes, ZeroHeuristic, 0, |s| *s == 1, 1);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[1, 3], [3, 1]]); // [2, 2] is within 1 of [1, 3]
/// ```
pub fn boa_star_approx<G, H, C, P>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    slack: C,
) -> Output<G::State, [C; 2]>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    run(graph, heuristic, start, is_goal, Tolerance::Additive(slack))
}

/// BOA*ε: [`boa_star`] with a multiplicative tolerance `1 + eps` on the
/// second objective.
///
/// Guarantees, under the requirements of [`boa_star`]:
///
/// - every returned solution is Pareto-optimal, and they are sorted as in
///   [`boa_star`];
/// - for every Pareto-optimal cost vector `s` some returned solution `r`
///   has `r[0] <= s[0]` and `r[1] <= (1 + eps) · s[1]`.
///
/// As in Goldin and Salzman's BOA*ε (ICAPS 2021, eq. 2), the only change to
/// [`boa_star`] is the pruning by solutions: a label is dropped once
/// `(1 + eps) · f2` reaches the second cost of the last solution. The
/// per-state pruning stays exact, so the first objective is never relaxed;
/// [`pp_a_star`](crate::pp_a_star) relaxes both. `eps` = [`Epsilon::ZERO`]
/// is exactly [`boa_star`].
///
/// # Why the guarantee holds
///
/// Labels leave the open list in lexicographic `f` order, so a solution
/// `r` found before a label `x` has `r[0] <= f1(x)`. A label dropped because
/// `r[1] <= (1 + eps) · f2(x)` only leads to solutions `s >= f(x)`, which
/// `r` covers. A label dropped by the exact per-state test is dominated by
/// an expanded label at the same state, whose completions dominate its own.
/// A returned solution `r'` passed the test against every earlier `r`, so
/// `(1 + eps) · r'[1] < r[1]`; a pruned path `s` with `r[1] <= (1 + eps) ·
/// s[1]` then has `s[1] > r'[1]` and cannot dominate `r'`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Epsilon, Graph, ZeroHeuristic, boa_star_eps};
///
/// /// Three routes from 0 to 1.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [1, 110]), (1, [2, 100]), (1, [3, 50])]);
///         }
///     }
/// }
///
/// let out = boa_star_eps(&Routes, ZeroHeuristic, 0, |s| *s == 1, Epsilon::new(1, 10));
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[1, 110], [3, 50]]); // 110 <= 1.1 * 100
/// ```
pub fn boa_star_eps<G, H, C, P>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    eps: Epsilon,
) -> Output<G::State, [C; 2]>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    run(graph, heuristic, start, is_goal, Tolerance::Relative(eps))
}

/// Runs one unidirectional BOA* search with the given solution-pruning
/// tolerance.
fn run<G, H, C, P>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    tolerance: Tolerance<C>,
) -> Output<G::State, [C; 2]>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut search = Search::new(graph, heuristic, is_goal, tolerance);
    search.push_start(start);
    while search.step() {}

    search.into_output()
}

/// How much worse in the second objective a pruned label may be than the
/// last solution that prunes it.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Tolerance<C> {
    /// Prune when `f2 + slack >= best`.
    Additive(C),
    /// Prune when `best <= (1 + ε) · f2`.
    Relative(Epsilon),
}

impl<C: Cost> Tolerance<C> {
    /// No tolerance: exact BOA* pruning, `f2 >= best`.
    pub(crate) const EXACT: Self = Self::Additive(C::ZERO);

    /// Returns `true` if a solution with second cost `best` covers every
    /// completion of a label with second estimate `f2`.
    fn covers(self, best: C, f2: C) -> bool {
        match self {
            Self::Additive(slack) => f2 + slack >= best,
            Self::Relative(eps) => eps.within(best, f2),
        }
    }
}

/// A known completion of a state to the target, for BOBA*'s early
/// solution update: the cost of the completion that is optimal in the
/// primary objective (ties broken by the secondary one).
pub(crate) struct Completion<C> {
    /// Primary cost of the completion, the exact primary distance.
    pub(crate) primary: C,
    /// Secondary cost of the same completion.
    pub(crate) secondary: C,
    /// `true` if the completion is also optimal in the secondary
    /// objective, so it dominates every other completion and the state
    /// need not be expanded.
    pub(crate) terminal: bool,
}

/// Completions of states to the target, known before the search runs.
pub(crate) trait Completions<S, C> {
    /// The primary-optimal completion of `state`, or `None` if unknown.
    fn completion(&self, state: &S) -> Option<Completion<C>>;

    /// Appends the states of that completion after `state`, target
    /// included.
    fn suffix(&self, state: &S, out: &mut Vec<S>);
}

/// No known completions: plain BOA*.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NoCompletions;

impl<S, C> Completions<S, C> for NoCompletions {
    fn completion(&self, _state: &S) -> Option<Completion<C>> {
        None
    }

    fn suffix(&self, _state: &S, _out: &mut Vec<S>) {}
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
///
/// Also one direction of BOBA*, which drives two of these step by step and
/// exchanges their bounds; see [`Search::set_primary_bound`].
pub(crate) struct Search<'a, G: Graph, H, C, P, K = NoCompletions> {
    /// The graph being searched.
    graph: &'a G,
    /// Per-objective lower bounds on the remaining cost.
    heuristic: H,
    /// Goal test.
    is_goal: P,
    /// Tolerance on the second objective when pruning by solutions.
    /// Additive zero for exact search.
    tolerance: Tolerance<C>,
    /// Labels with `f1` at or above this bound are pruned, and the search
    /// stops once the open list reaches it. `None` (no bound) except in
    /// BOBA*, where it is the first cost of the other direction's last
    /// solution.
    primary_bound: Option<C>,
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
    /// Known completions for the early solution update; [`NoCompletions`]
    /// outside BOBA*'s own heuristics.
    completions: K,
    /// Raised lower bounds on the secondary heuristic per state, from the
    /// other direction's first expansions (BOBA*'s heuristic tuning).
    secondary_floor: BTreeMap<G::State, C>,
    /// The state and primary `g` of the latest first expansion of a state,
    /// until [`Search::take_first_expansion`] collects it.
    first_expansion: Option<(G::State, C)>,
}

impl<'a, G, H, C, P> Search<'a, G, H, C, P>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    /// Creates an empty plain BOA* search with no labels.
    pub(crate) fn new(graph: &'a G, heuristic: H, is_goal: P, tolerance: Tolerance<C>) -> Self {
        Search::with_completions(graph, heuristic, is_goal, tolerance, NoCompletions)
    }
}

impl<'a, G, H, C, P, K> Search<'a, G, H, C, P, K>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
    K: Completions<G::State, C>,
{
    /// Creates an empty search with no labels that uses `completions` for
    /// the early solution update.
    pub(crate) fn with_completions(
        graph: &'a G,
        heuristic: H,
        is_goal: P,
        tolerance: Tolerance<C>,
        completions: K,
    ) -> Self {
        Self {
            graph,
            heuristic,
            is_goal,
            tolerance,
            primary_bound: None,
            nodes: Vec::new(),
            open: BinaryHeap::new(),
            g2_min: BTreeMap::new(),
            best_goal_g2: None,
            solutions: Vec::new(),
            successors: Vec::new(),
            stats: Stats::default(),
            completions,
            secondary_floor: BTreeMap::new(),
            first_expansion: None,
        }
    }

    /// Raises the secondary heuristic of `state` to at least `floor`, from
    /// the other direction's first expansion of `state`.
    pub(crate) fn raise_secondary(&mut self, state: G::State, floor: C) {
        let entry = self.secondary_floor.entry(state).or_insert(floor);
        *entry = (*entry).max(floor);
    }

    /// The state and primary `g` of the latest first expansion of a state
    /// since the last call, if any.
    pub(crate) fn take_first_expansion(&mut self) -> Option<(G::State, C)> {
        self.first_expansion.take()
    }

    /// Processes the next label from the open list.
    ///
    /// Returns `false`, without processing anything, once the search is
    /// done: the open list is empty or its best label has `f1` at or above
    /// the primary bound.
    pub(crate) fn step(&mut self) -> bool {
        let Some(Reverse((f, index))) = self.open.pop() else {
            return false;
        };

        if self.primary_bound.is_some_and(|bound| f[0] >= bound) {
            return false;
        }

        self.process(f, index);

        true
    }

    /// The second cost of the last solution found, which is the smallest
    /// so far; `None` before the first one.
    pub(crate) fn secondary_bound(&self) -> Option<C> {
        self.best_goal_g2
    }

    /// Sets the bound on `f1` beyond which labels cannot lead to new
    /// solutions. Only ever lowered, by the BOBA* driver.
    pub(crate) fn set_primary_bound(&mut self, bound: Option<C>) {
        self.primary_bound = bound;
    }

    /// Consumes the search and returns its solutions and counters.
    pub(crate) fn into_output(self) -> Output<G::State, [C; 2]> {
        Output {
            solutions: self.solutions,
            stats: self.stats,
        }
    }

    /// Pushes the start label, unless the heuristic reports it as a dead end.
    pub(crate) fn push_start(&mut self, start: G::State) {
        if let Some(h) = self.estimate(&start) {
            self.push(start, [C::ZERO; 2], h, None);
        }
    }

    /// The heuristic estimate of `state`, with the secondary objective
    /// raised to its tuned floor.
    fn estimate(&mut self, state: &G::State) -> Option<[C; 2]> {
        let mut h = self.heuristic.estimate(state)?;
        if let Some(&floor) = self.secondary_floor.get(state) {
            h[1] = h[1].max(floor);
        }

        Some(h)
    }

    /// Handles one label taken from the open list: prune, record a
    /// solution (early, through a known completion, or at a goal), or
    /// expand.
    fn process(&mut self, f: [C; 2], index: usize) {
        let node = &self.nodes[index];
        if self.is_dominated(&node.state, node.g[1], f) {
            return;
        }

        let (state, g) = (node.state.clone(), node.g);
        if self.g2_min.insert(state.clone(), g[1]).is_none() {
            self.first_expansion = Some((state.clone(), g[0]));
        }

        if let Some(completion) = self.completions.completion(&state) {
            let cost = [g[0] + completion.primary, g[1] + completion.secondary];
            if self.best_goal_g2.is_none_or(|best| cost[1] < best) {
                let solution = self.early_solution(index, cost);
                self.add_solution(solution);
            }

            if completion.terminal {
                return;
            }
        }

        if (self.is_goal)(&state) {
            let solution = self.solution(index);
            self.add_solution(solution);
        } else {
            self.expand(index);
        }
    }

    /// Records a solution; it replaces the last one if both have the same
    /// first cost (the new one then has the smaller second cost).
    fn add_solution(&mut self, solution: Solution<G::State, [C; 2]>) {
        self.best_goal_g2 = Some(solution.cost[1]);
        if self
            .solutions
            .last()
            .is_some_and(|last| last.cost[0] == solution.cost[0])
        {
            self.solutions.pop();
        }

        self.solutions.push(solution);
    }

    /// The path of the label at `index` joined with the known completion of
    /// its state, at total cost `cost`.
    fn early_solution(&self, index: usize, cost: [C; 2]) -> Solution<G::State, [C; 2]> {
        let mut solution = self.solution(index);
        self.completions
            .suffix(&self.nodes[index].state, &mut solution.path);

        solution.cost = cost;
        solution
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
            let Some(h) = self.estimate(&state) else {
                continue;
            };

            let g_next = add(g, weight);
            if !self.is_dominated(&state, g_next[1], add(g_next, h)) {
                self.push(state, g_next, h, Some(index));
            }
        }

        self.successors = successors;
    }

    /// Returns `true` if a label at `state` with second cost `g2` and
    /// estimate `f` is weakly dominated by an expanded label or by a
    /// solution (within the tolerance), or is beyond the primary bound.
    fn is_dominated(&self, state: &G::State, g2: C, f: [C; 2]) -> bool {
        self.g2_min.get(state).is_some_and(|&min| g2 >= min)
            || self
                .best_goal_g2
                .is_some_and(|best| self.tolerance.covers(best, f[1]))
            || self.primary_bound.is_some_and(|bound| f[0] >= bound)
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
