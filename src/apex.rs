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
//!   pair (component-wise minimum apex, a representative chosen by a
//!   [`MergeRule`]) is ε-bounded;
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
//! may merge (they are never extended, and `h = 0` at every goal); and
//! dominated or duplicate representatives are dropped from the output.

mod anytime;

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::mem;

use search_core::{Cost, Heuristic, WideMul};

use crate::cost::add;
use crate::epsilon::Epsilon;
use crate::graph::{Graph, ZeroHeuristic};
use crate::output::{Output, Solution, Stats};
use crate::pareto::{Front, TruncatedFront};

pub use anytime::apex_star_anytime;

/// How [`apex_star_with`] picks the representative path when it merges two
/// apex-path pairs. Both are the paper's; the choice never affects the
/// guarantee, only how many pairs merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MergeRule {
    /// The paper's G method, the one its experiments favour: of the two representatives that
    /// keep the merged pair ε-bounded, take the one with the larger slack
    /// `min_i (1 + ε_i - f_i(π) / f_i(A)) / ε_i`, which leaves the most room
    /// for later merges. The first pair's representative wins a tie.
    /// Compared exactly by cross-multiplication with [`WideMul`].
    #[default]
    Greedy,
    /// The paper's L method: take the representative whose cost is
    /// lexicographically smaller read from the last objective to the first
    /// (the first pair's on a tie), and do not merge if it is not
    /// ε-bounded. For two objectives this is PP-A*'s bottom-right path.
    ReverseLex,
}

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
/// - when two pairs merge, the representative is chosen by the paper's
///   greedy G method ([`MergeRule::Greedy`]); [`apex_star_with`] selects
///   another rule;
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
/// // [100, 100, 50] leaves the merged apex [100, 98, 50] more slack than
/// // [104, 98, 51], so it represents the merged pair and covers both.
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
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    apex_star_with(graph, heuristic, start, is_goal, eps, MergeRule::Greedy)
}

/// [`apex_star`] with a chosen [`MergeRule`] for the representative of
/// merged pairs.
///
/// Requirements and guarantees are those of [`apex_star`], for every rule.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Epsilon, Graph, MergeRule, ZeroHeuristic, apex_star_with};
///
/// /// Two routes from 0 to 1 that merge within 50 %.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [10, 13]), (1, [14, 10])]);
///         }
///     }
/// }
///
/// let eps = [Epsilon::new(1, 2); 2];
/// let run = |rule| apex_star_with(&Routes, ZeroHeuristic, 0, |s| *s == 1, eps, rule);
/// // Against the merged apex [10, 10], [10, 13] is 30 % over in one
/// // objective and [14, 10] is 40 % over: the greedy rule keeps the first.
/// assert_eq!(run(MergeRule::Greedy).solutions[0].cost, [10, 13]);
/// // The reverse-lexicographic rule prefers the smaller last objective.
/// assert_eq!(run(MergeRule::ReverseLex).solutions[0].cost, [14, 10]);
/// ```
pub fn apex_star_with<G, H, C, P, const N: usize>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    eps: [Epsilon; N],
    rule: MergeRule,
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let mut search = Search::new(graph, heuristic, is_goal, eps, rule, false);
    search.push_start(start);
    search.run();

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
    /// Full apexes of the pairs expanded here; only filled when the search
    /// records them.
    expanded: Front<C, N>,
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
    /// How merges pick the representative.
    rule: MergeRule,
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
    /// Costs of solutions known from earlier runs (A-A*pex): pairs within
    /// the tolerances of one are pruned, checked in every objective.
    seeds: Vec<[C; N]>,
    /// Whether [`StateInfo::expanded`] is filled, for
    /// [`expanded_apexes`].
    record: bool,
    /// Effort counters.
    stats: Stats,
}

/// Runs A*pex from every state in `starts` with no goal and the zero
/// heuristic, and returns, per reached state, the non-dominated apexes of
/// the pairs expanded there.
///
/// Every path from a start to a state `s` has a cost weakly dominated by
/// one of `s`'s apexes (case 1 of the paper's Lemma 4; with no goal the
/// solution case never applies), and the apexes are within the tolerances
/// of real path costs. This is how Wolff et al. (arXiv 2606.20644,
/// Section 6) build a multi-valued heuristic, on the reversed graph.
pub(crate) fn expanded_apexes<G, C, const N: usize>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    eps: [Epsilon; N],
) -> BTreeMap<G::State, Front<C, N>>
where
    G: Graph<Weight = [C; N]>,
    C: Cost + WideMul,
{
    let never = |_: &G::State| false;
    let mut search = Search::new(graph, ZeroHeuristic, never, eps, MergeRule::Greedy, true);
    for start in starts {
        search.push_start(start);
    }

    search.run();
    search
        .states
        .into_iter()
        .filter(|(_, info)| !info.expanded.is_empty())
        .map(|(state, info)| (state, info.expanded))
        .collect()
}

impl<'a, G, H, C, P, const N: usize> Search<'a, G, H, C, P, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    /// Creates an empty search; `record` fills [`StateInfo::expanded`].
    fn new(
        graph: &'a G,
        heuristic: H,
        is_goal: P,
        eps: [Epsilon; N],
        rule: MergeRule,
        record: bool,
    ) -> Self {
        Self {
            graph,
            heuristic,
            is_goal,
            eps,
            rule,
            paths: Vec::new(),
            pairs: Vec::new(),
            open: BinaryHeap::new(),
            states: BTreeMap::new(),
            solutions: Vec::new(),
            successors: Vec::new(),
            seeds: Vec::new(),
            record,
            stats: Stats::default(),
        }
    }

    /// Processes pairs until the open list is empty.
    fn run(&mut self) {
        while let Some(Reverse((f, index))) = self.open.pop() {
            self.process(f, index);
        }
    }

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

        let record = self.record;
        let info = self.info_mut(&state);
        info.closed.insert(pair.apex);
        if record {
            info.expanded.insert(pair.apex);
        }

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
        // Seeds come from earlier runs, so they get the full test.
        if self.seeds.iter().any(|s| self.within(s, f)) {
            return true;
        }

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
    /// representative chosen by [`Search::rule`], or `None` if no choice
    /// the rule allows is ε-bounded (lines 29 and 30).
    ///
    /// `a` and `b` share `h`: they are at the same state, or both at goals,
    /// where `h` is zero.
    fn merge(&self, a: Pair<C, N>, b: Pair<C, N>) -> Option<Pair<C, N>> {
        let with_rep = |rep| Pair {
            apex: core::array::from_fn(|i| a.apex[i].min(b.apex[i])),
            rep,
            h: a.h,
            alive: true,
        };
        let (for_a, for_b) = (with_rep(a.rep), with_rep(b.rep));

        match self.rule {
            MergeRule::Greedy => {
                let excess_a = self.excess(&for_a);
                let excess_b = self.excess(&for_b);
                match (excess_a, excess_b) {
                    (Some(x), Some(y)) if x.is_greater(y) => Some(for_b),
                    (Some(_), _) => Some(for_a),
                    (None, Some(_)) => Some(for_b),
                    (None, None) => None,
                }
            }
            MergeRule::ReverseLex => {
                let (ga, gb) = (&self.paths[a.rep].g, &self.paths[b.rep].g);
                let merged = if gb.iter().rev().lt(ga.iter().rev()) {
                    for_b
                } else {
                    for_a
                };

                self.is_bounded(&merged).then_some(merged)
            }
        }
    }

    /// Returns `true` if the representative's `f` is within the tolerances
    /// of the apex's `f` in every objective.
    fn is_bounded(&self, pair: &Pair<C, N>) -> bool {
        let f_rep = add(self.paths[pair.rep].g, pair.h);
        self.within(&f_rep, &add(pair.apex, pair.h))
    }

    /// The largest relative excess `max_i (f_i(π) - f_i(A)) / (ε_i f_i(A))`
    /// of the representative over the apex, or `None` if the pair is not
    /// ε-bounded.
    ///
    /// The paper's slack is one minus this, so the greedy rule's larger
    /// slack is the smaller excess. Both terms of an [`Excess::Ratio`] are
    /// exact: a bounded pair has `den_i · f_i(π)` and
    /// `(den_i + num_i) · f_i(A)` without saturation (see
    /// [`Epsilon::within`]), and both terms are at most those.
    fn excess(&self, pair: &Pair<C, N>) -> Option<Excess<C>> {
        if !self.is_bounded(pair) {
            return None;
        }

        let f_rep = add(self.paths[pair.rep].g, pair.h);
        let f_apex = add(pair.apex, pair.h);
        let ratios = (0..N).filter(|&i| f_rep[i] != f_apex[i]).map(|i| {
            let eps = self.eps[i];
            let over = f_rep[i].abs_diff(f_apex[i]).saturating_times(eps.den());
            Excess::Ratio {
                over,
                of: f_apex[i].saturating_times(eps.num()),
            }
        });

        Some(ratios.fold(Excess::Zero, |x, y| if y.is_greater(x) { y } else { x }))
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
                expanded: Front::new(),
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

/// How far a representative's `f` exceeds its apex's `f`, relative to the
/// tolerance, in the worst objective.
#[derive(Clone, Copy)]
enum Excess<C> {
    /// The representative's `f` equals the apex's in every objective.
    Zero,
    /// The positive ratio `over / of`: `over` is `den_i · (f_i(π) - f_i(A))`
    /// and `of` is `num_i · f_i(A)`, both non-zero.
    Ratio {
        /// The scaled excess of the representative.
        over: C,
        /// The scaled tolerance it is measured against.
        of: C,
    },
}

impl<C: WideMul> Excess<C> {
    /// Returns `true` if `self` is larger than `other`, compared exactly by
    /// cross-multiplication.
    fn is_greater(self, other: Self) -> bool {
        match (self, other) {
            (Self::Ratio { over, of }, Self::Ratio { over: o, of: f }) => {
                over.wide_mul(f) > o.wide_mul(of)
            }
            (Self::Ratio { .. }, Self::Zero) => true,
            (Self::Zero, _) => false,
        }
    }
}
