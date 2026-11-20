//! A*pex: approximate multi-objective search over apex-path pairs.
//!
//! Zhang, Salzman, Kumar, Felner, Hernández, Koenig: "A*pex: Efficient
//! Approximate Multi-Objective Search on Graphs", ICAPS 2022.
//!
//! An *apex-path pair* at a state is an apex, a cost vector component-wise
//! no worse than every path the pair stands for, and one real
//! *representative* path to the state. The pair is *ε-bounded* when the
//! representative's `f` is within the tolerances of the apex's `f` in every
//! objective: `g_i(rep) + h_i <= (1 + ε_i) · (apex_i + h_i)` (the paper's
//! Contribution 3). With a consistent heuristic, extending both along an
//! edge keeps the pair bounded, and at a goal (`h = 0`) the representative
//! is within the tolerances of every path the pair stands for. The search
//! is NAMOA*dr over apexes (Algorithm 2 of the paper), with three
//! additions:
//!
//! - a new pair merges into an open pair at the same state when the merged
//!   pair (component-wise minimum apex, the representative with the
//!   lexicographically smaller reverse cost) is ε-bounded;
//! - a goal pair merges into a solution pair the same way;
//! - a pair is dropped when some solution's representative is within the
//!   tolerances of its `f` in objectives `1..N`. That solution's apex then
//!   drops to the component-wise minimum of the two, so that later merges
//!   of the solution keep covering the dropped pair.
//!
//! ```mermaid
//! graph TD
//!     pop[Pop min apex f] --> pruned{"a solution within (1+ε) of f, or<br/>f dominated by an expanded apex? (objectives 1..N)"}
//!     pruned -- yes --> pop
//!     pruned -- no --> close[Archive apex]
//!     close --> goal{Goal?}
//!     goal -- yes --> sol[Merge into a solution if bounded,<br/>else add as solution] --> pop
//!     goal -- no --> expand[Extend apex and representative, prune] --> merge[Merge into an open pair<br/>at the state if bounded, else push] --> pop
//! ```
//!
//! Differences from the paper, which keep its guarantee: a goal predicate
//! replaces the single goal, so solution pairs at different goal states
//! may merge (they are never extended, and `h = 0` at every goal); the
//! representative is chosen by the paper's L method rather than its G
//! (greedy slack) method, because [`Cost`] has no division or cost-by-cost
//! product to compare the slack ratios exactly; and dominated or duplicate
//! representatives are dropped from the output.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::mem;

use search_core::{Cost, Heuristic};

use crate::cost::add;
use crate::epsilon::Epsilon;
use crate::graph::Graph;
use crate::output::{Output, Solution, Stats};
use crate::pareto::{Front, TruncatedFront};

/// A*pex: finds a set of solutions that covers the Pareto-optimal front
/// within a factor `1 + eps[i]` in each objective `i`, for any number of
/// objectives `N`.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is consistent in every objective and zero on every goal.
///
/// Guarantees:
///
/// - for every Pareto-optimal cost vector `s` some returned solution `r`
///   has `r[i] <= (1 + eps[i]) · s[i]` for every `i`;
/// - the returned solutions are real paths with their exact costs, mutually
///   non-dominated and sorted lexicographically ascending; they need not be
///   Pareto-optimal themselves;
/// - when two pairs merge, the representative is the one whose cost is
///   lexicographically smaller read from the last objective to the first
///   (the paper's L method), and the pairs stay apart if that one is not
///   ε-bounded by the merged apex;
/// - `eps = [Epsilon::ZERO; N]` returns exactly the cost vectors of
///   [`emoa_star`](crate::emoa_star).
///
/// # Why the guarantee holds
///
/// This is Lemma 4 and Theorem 1 of the paper. Pop order keeps the first
/// component of `f` non-decreasing (Lemma 1; a merged apex is the minimum
/// of two apexes whose first components are both at least the last popped
/// one), which is all that the truncated checks against expanded apexes
/// and solutions need. Each pruning or merging step covers a pair relative
/// to its own apex, and apexes only go down along these steps, so the
/// tolerances never compound: an expanded apex at or below a pruned one
/// covers it, a merged apex covers both pairs (Lemma 3), a solution whose
/// apex was lowered to the pruned pair's `f` keeps covering it through
/// later merges, and a bounded goal pair's representative covers its apex.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Epsilon, Graph, ZeroHeuristic, apex_star};
///
/// /// Three routes from 0 to 1 with three objectives.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 3];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 3])>) {
///         if *state == 0 {
///             out.extend([(1, [100, 100, 50]), (1, [104, 98, 51]), (1, [50, 200, 100])]);
///         }
///     }
/// }
///
/// let eps = [Epsilon::new(1, 10); 3];
/// let out = apex_star(&Routes, ZeroHeuristic, 0, |s| *s == 1, eps);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// // [100, 100, 50] has the lexicographically smaller reverse cost, so it
/// // represents the pair merged with [104, 98, 51] and covers it.
/// assert_eq!(costs, [[50, 200, 100], [100, 100, 50]]);
/// ```
pub fn apex_star<G, H, C, P, const N: usize>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    eps: [Epsilon; N],
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut search = Search {
        graph,
        heuristic,
        is_goal,
        eps,
        paths: Vec::new(),
        pairs: Vec::new(),
        open: BinaryHeap::new(),
        states: BTreeMap::new(),
        solutions: Vec::new(),
        successors: Vec::new(),
        stats: Stats::default(),
    };

    search.push_start(start);
    while let Some(Reverse((f, index))) = search.open.pop() {
        search.process(f, index);
    }

    search.into_output()
}

/// One path in the path arena: its last state, its cost and its parent.
struct PathNode<S, C, const N: usize> {
    /// The state this path ends in.
    state: S,
    /// Cost vector of the path from the start.
    g: [C; N],
    /// Arena index of the path without its last edge; `None` for the start.
    parent: Option<usize>,
}

/// An apex-path pair.
#[derive(Clone, Copy)]
struct Pair<C, const N: usize> {
    /// The apex: component-wise no worse than every path the pair stands
    /// for. Not itself the cost of a path in general.
    apex: [C; N],
    /// Path arena index of the representative path.
    rep: usize,
    /// Heuristic estimate at the pair's state.
    h: [C; N],
    /// `false` once the pair was merged into another; its open-list entry
    /// is then skipped.
    alive: bool,
}

/// Per-state bookkeeping.
struct StateInfo<C, const N: usize> {
    /// Apexes of the pairs expanded (or recorded as solutions) here,
    /// compared on objectives `1..N`.
    closed: TruncatedFront<C, N>,
    /// Pair arena indices of this state's pairs on the open list.
    open: Vec<usize>,
}

/// Mutable state of one A*pex run.
struct Search<'a, G: Graph, H, C, P, const N: usize> {
    /// The graph being searched.
    graph: &'a G,
    /// Per-objective lower bounds on the remaining cost.
    heuristic: H,
    /// Goal test.
    is_goal: P,
    /// Tolerances per objective.
    eps: [Epsilon; N],
    /// Every representative path; pairs and parents index into it.
    paths: Vec<PathNode<G::State, C, N>>,
    /// Every pair ever pushed; open-list entries index into it.
    pairs: Vec<Pair<C, N>>,
    /// Open list as a min-heap on `(apex f, pair index)`, compared
    /// lexicographically.
    open: BinaryHeap<Reverse<([C; N], usize)>>,
    /// Per-state bookkeeping. Absent means "not reached yet".
    states: BTreeMap<G::State, StateInfo<C, N>>,
    /// The solution pairs. Their apexes only go down, by merges and by
    /// pruning updates.
    solutions: Vec<Pair<C, N>>,
    /// Reused successor buffer, empty between expansions.
    successors: Vec<(G::State, [C; N])>,
    /// Effort counters.
    stats: Stats,
}

impl<G, H, C, P, const N: usize> Search<'_, G, H, C, P, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    /// Pushes the start pair, unless the heuristic reports a dead end.
    fn push_start(&mut self, start: G::State) {
        if let Some(h) = self.heuristic.estimate(&start) {
            let rep = self.new_path(start, [C::ZERO; N], None);
            self.push(Pair {
                apex: [C::ZERO; N],
                rep,
                h,
                alive: true,
            });
        }
    }

    /// Handles one pair taken from the open list: skip it if merged away,
    /// prune it, record a solution, or expand it.
    fn process(&mut self, f: [C; N], index: usize) {
        let pair = self.pairs[index];
        if !pair.alive {
            return;
        }

        let state = self.paths[pair.rep].state.clone();
        self.info_mut(&state).open.retain(|&i| i != index);
        if self.is_pruned(&state, &pair.apex, &f) {
            return;
        }

        self.info_mut(&state).closed.insert(pair.apex);

        if (self.is_goal)(&state) {
            self.insert_solution(pair);
        } else {
            self.expand(pair);
        }
    }

    /// Returns `true` if a pair at `state` with apex `apex` and apex
    /// estimate `f` is covered by a solution or by an expanded pair there
    /// (the paper's `is_dominated`, lines 19 to 26).
    fn is_pruned(&mut self, state: &G::State, apex: &[C; N], f: &[C; N]) -> bool {
        self.is_covered_by_solution(f)
            || self
                .states
                .get(state)
                .is_some_and(|info| info.closed.weakly_dominates(apex))
    }

    /// Returns `true` if some solution's representative is within the
    /// tolerances of `f` in objectives `1..N`, and lowers that solution's
    /// apex to the component-wise minimum with `f` (lines 20 to 22).
    ///
    /// Objective 0 needs no check: the solution's apex is no larger there
    /// than `f` (Lemma 1), and its bounded representative is within the
    /// tolerance of that apex.
    fn is_covered_by_solution(&mut self, f: &[C; N]) -> bool {
        let paths = &self.paths;
        let eps = &self.eps;
        let Some(solution) = self.solutions.iter_mut().find(|s| {
            let g = &paths[s.rep].g;
            (1..N).all(|i| eps[i].within(g[i], f[i]))
        }) else {
            return false;
        };

        solution.apex = core::array::from_fn(|i| solution.apex[i].min(f[i]));
        true
    }

    /// Returns `true` if `cost[i] <= (1 + eps[i]) · bound[i]` for every `i`.
    fn within(&self, cost: &[C; N], bound: &[C; N]) -> bool {
        (0..N).all(|i| self.eps[i].within(cost[i], bound[i]))
    }

    /// Extends the apex and the representative of `pair` along every
    /// outgoing edge and inserts the surviving pairs.
    fn expand(&mut self, pair: Pair<C, N>) {
        self.stats.expanded += 1;

        let mut successors = mem::take(&mut self.successors);
        let from = self.paths[pair.rep].state.clone();
        self.graph.successors(&from, &mut successors);

        for (state, weight) in successors.drain(..) {
            let Some(h) = self.heuristic.estimate(&state) else {
                continue;
            };

            let apex = add(pair.apex, weight);
            if self.is_pruned(&state, &apex, &add(apex, h)) {
                continue;
            }

            let g = add(self.paths[pair.rep].g, weight);
            let rep = self.new_path(state, g, Some(pair.rep));
            self.insert(Pair {
                apex,
                rep,
                h,
                alive: true,
            });
        }

        self.successors = successors;
    }

    /// Merges the goal pair `pair` into the first solution pair it merges
    /// with, or adds it as a new solution (line 11 and lines 27 to 35).
    fn insert_solution(&mut self, pair: Pair<C, N>) {
        let found = self
            .solutions
            .iter()
            .enumerate()
            .find_map(|(k, &other)| self.merge(pair, other).map(|merged| (k, merged)));

        match found {
            Some((k, merged)) => self.solutions[k] = merged,
            None => self.solutions.push(pair),
        }
    }

    /// Merges `pair` into the first open pair at its state that it merges
    /// with, or pushes it as is (line 17 and lines 27 to 35).
    fn insert(&mut self, pair: Pair<C, N>) {
        let state = self.paths[pair.rep].state.clone();
        let open = self
            .states
            .get(&state)
            .map_or(&[][..], |info| &info.open[..]);
        let found = open.iter().enumerate().find_map(|(position, &other)| {
            self.merge(pair, self.pairs[other])
                .map(|merged| (position, other, merged))
        });

        if let Some((position, other, merged)) = found {
            self.pairs[other].alive = false;
            self.info_mut(&state).open.swap_remove(position);
            self.push(merged);
            return;
        }

        self.push(pair);
    }

    /// The pair with the component-wise minimum apex of `a` and `b` and the
    /// representative chosen by the paper's L method, or `None` if that
    /// pair is not ε-bounded.
    ///
    /// The L method takes the representative whose cost is lexicographically
    /// smaller read from the last objective to the first (`a`'s on a tie)
    /// and does not fall back to the other one. `a` and `b` share `h`: they
    /// are at the same state, or both at goals, where `h` is zero.
    fn merge(&self, a: Pair<C, N>, b: Pair<C, N>) -> Option<Pair<C, N>> {
        let apex = core::array::from_fn(|i| a.apex[i].min(b.apex[i]));
        let (ga, gb) = (&self.paths[a.rep].g, &self.paths[b.rep].g);
        let rep = if gb.iter().rev().lt(ga.iter().rev()) {
            b.rep
        } else {
            a.rep
        };

        let merged = Pair {
            apex,
            rep,
            h: a.h,
            alive: true,
        };

        self.is_bounded(&merged).then_some(merged)
    }

    /// Returns `true` if the representative's `f` is within the tolerances
    /// of the apex's `f` in every objective.
    fn is_bounded(&self, pair: &Pair<C, N>) -> bool {
        let f_rep = add(self.paths[pair.rep].g, pair.h);
        self.within(&f_rep, &add(pair.apex, pair.h))
    }

    /// Adds a pair to the pair arena, the open list and its state's open
    /// pairs.
    fn push(&mut self, pair: Pair<C, N>) {
        let index = self.pairs.len();
        let state = self.paths[pair.rep].state.clone();

        self.pairs.push(pair);
        self.open.push(Reverse((add(pair.apex, pair.h), index)));
        self.info_mut(&state).open.push(index);
        self.stats.generated += 1;
    }

    /// Adds a path to the path arena and returns its index.
    fn new_path(&mut self, state: G::State, g: [C; N], parent: Option<usize>) -> usize {
        self.paths.push(PathNode { state, g, parent });
        self.paths.len() - 1
    }

    /// The bookkeeping of `state`, created on first use.
    fn info_mut(&mut self, state: &G::State) -> &mut StateInfo<C, N> {
        self.states
            .entry(state.clone())
            .or_insert_with(|| StateInfo {
                closed: TruncatedFront::new(),
                open: Vec::new(),
            })
    }

    /// Consumes the search and returns the mutually non-dominated
    /// representatives, sorted, with the counters.
    fn into_output(self) -> Output<G::State, [C; N]> {
        let mut reps: Vec<_> = self
            .solutions
            .iter()
            .map(|s| (self.paths[s.rep].g, s.rep))
            .collect();
        reps.sort_unstable();

        // In lexicographic order a later vector cannot dominate an earlier
        // one, so one pass with a frontier drops every dominated or equal one.
        let mut front = Front::new();
        let solutions = reps
            .into_iter()
            .filter(|&(cost, _)| front.insert(cost))
            .map(|(cost, r)| self.solution(r, cost))
            .collect();

        Output {
            solutions,
            stats: self.stats,
        }
    }

    /// Rebuilds the path ending in path node `index`.
    fn solution(&self, index: usize, cost: [C; N]) -> Solution<G::State, [C; N]> {
        let mut path = Vec::new();
        let mut current = Some(index);

        while let Some(i) = current {
            path.push(self.paths[i].state.clone());
            current = self.paths[i].parent;
        }

        path.reverse();

        Solution { cost, path }
    }
}
