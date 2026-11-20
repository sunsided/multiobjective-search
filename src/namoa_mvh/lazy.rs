//! L-NAMOA*dr-mvh: lazy dimensionality reduction with an admissible
//! multi-valued heuristic (Algorithm 2 of the paper).

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::mem;

use search_core::Cost;

use crate::cost::add;
use crate::graph::Graph;
use crate::mvh::MvHeuristic;
use crate::output::{Output, Solution, Stats};
use crate::pareto::{Front, TruncatedFront};

/// L-NAMOA*dr-mvh: NAMOA*dr with an admissible multi-valued heuristic and
/// lazy dimensionality reduction.
///
/// One label per path, bound to the lexicographically smallest estimate of
/// its state whose `f = g + h` no solution weakly dominates (`ChooseH`). A
/// popped label whose `f` became dominated is re-bound to the next such
/// estimate and pushed again instead of being dropped. Against the closed
/// labels of a state, the check first drops the first objective, as
/// NAMOA*dr does, but trusts that answer only if no closed label there has
/// a larger `g_1`; otherwise it falls back to a full dominance check
/// (`LocalDomCheck`). The answer is the exact Pareto-optimal set
/// (Theorem 2), without requiring consistency.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is an admissible [`MvHeuristic`]: for every path from a
///   state to a goal, some estimate weakly dominates its cost (so every
///   goal has the zero estimate).
///
/// Each state's estimates are fetched once, when it is first reached, and
/// kept sorted. Solutions are sorted lexicographically, one per
/// Pareto-optimal cost vector.
///
/// # Differences from the paper
///
/// A goal predicate replaces the single goal. The local fallback compares
/// against the largest `g_1` ever closed at the state rather than the
/// largest one still in the truncated set, which can only fall back more
/// often.
///
/// The check against solutions drops the first objective, as in the paper,
/// although labels can leave the open list out of `f_1` order. It stays
/// sound because Pareto-optimal solutions are still found in lexicographic
/// order: every label on such a path is bound to an estimate no greater,
/// lexicographically, than the one that bounds the path, so it is popped
/// before any lexicographically larger solution.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, mv_heuristic_fn, namoa_star_dr_mvh_lazy};
///
/// /// Two routes from 0 to the goal 2 through 1.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         match state {
///             0 => out.push((1, [1, 1])),
///             1 => out.extend([(2, [1, 4]), (2, [3, 1])]),
///             _ => {}
///         }
///     }
/// }
///
/// // Admissible, but not consistent: state 1 knows less than state 0.
/// let h = mv_heuristic_fn(|s: &u8, out: &mut Vec<[u32; 2]>| match s {
///     0 => out.extend([[2, 5], [4, 2]]),
///     _ => out.push([0, 0]),
/// });
///
/// let out = namoa_star_dr_mvh_lazy(&Routes, h, 0, |s| *s == 2);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[2, 5], [4, 2]]);
/// ```
pub fn namoa_star_dr_mvh_lazy<G, M, C, P, const N: usize>(
    graph: &G,
    heuristic: M,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]>,
    M: MvHeuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut search = Search::new(graph, heuristic, is_goal);
    search.push_start(start);

    while let Some(Reverse((f, index))) = search.open.pop() {
        search.process(f, index);
    }

    search.into_output()
}

/// One label: a path, stored in the node arena.
struct Node<S, C, const N: usize> {
    /// The state this path ends in.
    state: S,
    /// Cost vector of the path from the start.
    g: [C; N],
    /// Arena index of the predecessor label; `None` for the start.
    parent: Option<usize>,
}

/// Per-state bookkeeping.
struct StateInfo<C, const N: usize> {
    /// The state's estimates, sorted lexicographically ascending. Empty for
    /// a dead end.
    estimates: Vec<[C; N]>,
    /// `g` of the labels closed here, compared on objectives `1..N`.
    closed_tail: TruncatedFront<C, N>,
    /// `g` of the labels closed here, compared in full: the fallback.
    closed: Front<C, N>,
    /// The largest `g_1` ever closed here; `None` before the first.
    max_g1: Option<C>,
}

/// Mutable state of one run.
struct Search<'a, G: Graph, M, C, P, const N: usize> {
    /// The graph being searched.
    graph: &'a G,
    /// The multi-valued heuristic.
    heuristic: M,
    /// Goal test.
    is_goal: P,
    /// Every label; open-list entries and parents index into it.
    nodes: Vec<Node<G::State, C, N>>,
    /// Open list as a min-heap on `(f, arena index)`, compared
    /// lexicographically.
    open: BinaryHeap<Reverse<([C; N], usize)>>,
    /// Per-state bookkeeping, created when a state is first reached.
    states: BTreeMap<G::State, StateInfo<C, N>>,
    /// Costs of the solutions found so far, compared on objectives
    /// `1..N` (the paper's `G_cl^Tr(s_goal)`).
    costs: TruncatedFront<C, N>,
    /// Arena indices of the solutions, in the order found.
    solutions: Vec<usize>,
    /// Reused successor buffer, empty between expansions.
    successors: Vec<(G::State, [C; N])>,
    /// When `false`, the local check trusts the truncated answer without
    /// checking the `g_1` order: the naive combination the paper's Lemma 1
    /// refutes. Only tests turn it off.
    validate_order: bool,
    /// Effort counters.
    stats: Stats,
}

impl<'a, G, M, C, P, const N: usize> Search<'a, G, M, C, P, N>
where
    G: Graph<Weight = [C; N]>,
    M: MvHeuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    /// Creates an empty search.
    fn new(graph: &'a G, heuristic: M, is_goal: P) -> Self {
        Self {
            graph,
            heuristic,
            is_goal,
            nodes: Vec::new(),
            open: BinaryHeap::new(),
            states: BTreeMap::new(),
            costs: TruncatedFront::new(),
            solutions: Vec::new(),
            successors: Vec::new(),
            validate_order: true,
            stats: Stats::default(),
        }
    }

    /// Pushes the start label with the lexicographically smallest start
    /// estimate (line 5), unless the start is a dead end.
    fn push_start(&mut self, start: G::State) {
        self.reach(&start);
        if let Some(h) = self.choose_h(&start, &[C::ZERO; N]) {
            self.push(start, [C::ZERO; N], h, None);
        }
    }

    /// Handles one popped label (lines 8 to 32).
    fn process(&mut self, f: [C; N], index: usize) {
        let state = self.nodes[index].state.clone();
        let g = self.nodes[index].g;

        if self.costs.weakly_dominates(&f) {
            if let Some(h) = self.choose_h(&state, &g) {
                self.open.push(Reverse((add(g, h), index)));
            }

            return;
        }

        if self.is_locally_dominated(&state, &g) {
            return;
        }

        self.close(&state, g);
        if (self.is_goal)(&state) {
            self.costs.insert(g);
            self.solutions.push(index);
        } else {
            self.expand(index);
        }
    }

    /// Generates the successors of the label at `index` and pushes those
    /// that survive both checks (lines 23 to 32).
    fn expand(&mut self, index: usize) {
        self.stats.expanded += 1;

        let mut successors = mem::take(&mut self.successors);
        let g = self.nodes[index].g;
        self.graph
            .successors(&self.nodes[index].state, &mut successors);

        for (state, weight) in successors.drain(..) {
            let g_next = add(g, weight);
            self.reach(&state);

            let Some(h) = self.choose_h(&state, &g_next) else {
                continue;
            };

            if !self.is_locally_dominated(&state, &g_next) {
                self.push(state, g_next, h, Some(index));
            }
        }

        self.successors = successors;
    }

    /// `ChooseH` (lines 34 to 40): the lexicographically smallest estimate
    /// of `state` whose `g + h` no solution weakly dominates.
    fn choose_h(&self, state: &G::State, g: &[C; N]) -> Option<[C; N]> {
        self.states[state]
            .estimates
            .iter()
            .copied()
            .find(|&h| !self.costs.weakly_dominates(&add(*g, h)))
    }

    /// `LocalDomCheck` (lines 41 to 47): returns `true` if a closed label
    /// at `state` weakly dominates `g`.
    ///
    /// The truncated check alone is trusted only when no closed label has
    /// a larger `g_1` than `g`; then the dominating one is no worse in the
    /// first objective either. Otherwise the full archive decides.
    fn is_locally_dominated(&self, state: &G::State, g: &[C; N]) -> bool {
        let Some(info) = self.states.get(state) else {
            return false;
        };

        if !info.closed_tail.weakly_dominates(g) {
            return false;
        }

        let in_order = info.max_g1.is_none_or(|max| max <= g[0]);
        if in_order || !self.validate_order {
            return true;
        }

        info.closed.weakly_dominates(g)
    }

    /// Records `g` as closed at `state` (lines 18 and 19).
    fn close(&mut self, state: &G::State, g: [C; N]) {
        let info = self.states.get_mut(state).expect("reached before closed");
        info.closed_tail.insert(g);
        info.closed.insert(g);
        info.max_g1 = Some(info.max_g1.map_or(g[0], |max| max.max(g[0])));
    }

    /// Fetches the estimates of `state` on first contact.
    fn reach(&mut self, state: &G::State) {
        if self.states.contains_key(state) {
            return;
        }

        let mut estimates = Vec::new();
        self.heuristic.estimates(state, &mut estimates);
        estimates.sort_unstable();
        estimates.dedup();

        self.states.insert(
            state.clone(),
            StateInfo {
                estimates,
                closed_tail: TruncatedFront::new(),
                closed: Front::new(),
                max_g1: None,
            },
        );
    }

    /// Adds a label to the arena and the open list.
    fn push(&mut self, state: G::State, g: [C; N], h: [C; N], parent: Option<usize>) {
        let index = self.nodes.len();
        self.nodes.push(Node { state, g, parent });
        self.open.push(Reverse((add(g, h), index)));
        self.stats.generated += 1;
    }

    /// Consumes the search and returns the solutions sorted by cost.
    fn into_output(self) -> Output<G::State, [C; N]> {
        let mut solutions: Vec<_> = self.solutions.iter().map(|&i| self.solution(i)).collect();
        solutions.sort_by_key(|s| s.cost);

        Output {
            solutions,
            stats: self.stats,
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

#[cfg(test)]
mod tests {
    //! The paper's counterexample to the naive combination of an MVH with
    //! dimensionality reduction (Lemma 1, Figure 2, Table 1).

    use alloc::vec;
    use alloc::vec::Vec;
    use core::cmp::Reverse;

    use super::Search;
    use crate::graph::Graph;
    use crate::mvh::mv_heuristic_fn;

    /// States of Figure 2.
    const START: u8 = 0;
    /// State `s1`.
    const S1: u8 = 1;
    /// State `s2`.
    const S2: u8 = 2;
    /// State `s3`.
    const S3: u8 = 3;
    /// The goal state.
    const GOAL: u8 = 4;

    /// The search graph of Figure 2.
    struct Figure2;

    impl Graph for Figure2 {
        type State = u8;
        type Weight = [u32; 2];

        fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
            match *state {
                START => out.extend([(GOAL, [3, 15]), (S1, [7, 9]), (S3, [10, 6])]),
                S1 => out.push((S3, [1, 1])),
                S3 => out.extend([(GOAL, [2, 7]), (S2, [7, 1])]),
                S2 => out.push((GOAL, [1, 1])),
                _ => {}
            }
        }
    }

    /// Runs the lazy search on Figure 2 with the MVH of Table 1, checking
    /// the `g_1` order or not, and returns the solution costs.
    fn run(validate_order: bool) -> Vec<[u32; 2]> {
        let table = mv_heuristic_fn(|s: &u8, out: &mut Vec<[u32; 2]>| match *s {
            START => out.extend([[2, 14], [11, 12], [15, 11], [17, 7]]),
            S1 => out.extend([[2, 7], [8, 2]]),
            S2 => out.push([1, 1]),
            S3 => out.extend([[1, 6], [7, 1]]),
            _ => out.push([0, 0]),
        });

        let mut search = Search::new(&Figure2, table, |s: &u8| *s == GOAL);
        search.validate_order = validate_order;
        search.push_start(START);
        while let Some(Reverse((f, index))) = search.open.pop() {
            search.process(f, index);
        }

        search
            .into_output()
            .solutions
            .iter()
            .map(|s| s.cost)
            .collect()
    }

    /// The naive combination prunes `(8, 10)` at `s3` against the closed
    /// `(10, 6)` by the second objective alone and loses `(16, 12)`
    /// (iteration 5 of Table 2); the order check keeps it.
    #[test]
    fn naive_dimensionality_reduction_loses_a_solution() {
        assert_eq!(run(false), vec![[3, 15], [12, 13], [18, 8]]);
        assert_eq!(run(true), vec![[3, 15], [12, 13], [16, 12], [18, 8]]);
    }
}
