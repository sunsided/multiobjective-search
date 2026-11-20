//! A-BOA*ε: anytime approximate bi-objective search.
//!
//! "Anytime Approximate Bi-Objective Search" (anonymous preprint,
//! crl.cs.technion.ac.il).
//!
//! The search keeps the Pareto-optimal solutions found so far as the
//! corners of a list of *intervals*. An interval holds two consecutive
//! solutions, top-left `tl` and bottom-right `br`, and the *to-expand*
//! paths whose `f` lies between them and that may still lead to a new
//! Pareto-optimal solution there. Its approximation factor `ε̂` is the
//! worst, over its to-expand paths, of how well `tl` or `br` covers the
//! path. Each iteration takes the interval with the largest `ε̂` and runs
//! BOA*ε on its to-expand paths with `ε = ε̂ / d` (Algorithm 3), which
//! splits it at every solution found into intervals of factor at most `ε`.
//! The paths BOA*ε prunes only because of the tolerance are kept as the
//! new intervals' to-expand paths, so no work is repeated.
//!
//! ```mermaid
//! graph TD
//!     seed["tl, br: lexicographic optima (c1, c2) and (c2, c1)"] --> list["ILIST = [(tl, br, {start})]"]
//!     list --> pick{"some interval with ε̂ > 0?"}
//!     pick -- no --> done[exact Pareto-optimal front]
//!     pick -- yes --> search["BOA*ε with ε = ε̂ / d on the interval<br/>with the largest ε̂"]
//!     search --> split["replace it by the intervals between<br/>the solutions found"] --> report["on_front if the overall ε̂ fell"] --> pick
//! ```

mod factor;
mod lex;

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::mem;

use search_core::{Budget, Cost, Heuristic, WideMul};

use crate::cost::add;
use crate::graph::Graph;
use crate::output::{Solution, Stats};

pub use factor::ApproxFactor;

/// What an anytime bi-objective search returns: the solutions found, the
/// approximation factor they are known to achieve, and whether the search
/// ran to completion.
#[derive(Debug, Clone)]
pub struct AnytimeFront<S, C> {
    /// Pareto-optimal solutions, one per cost vector, sorted by cost
    /// lexicographically ascending (first objective strictly ascending,
    /// second strictly descending). Empty when no goal is reachable.
    pub solutions: Vec<Solution<S, [C; 2]>>,
    /// An `ε` such that every Pareto-optimal cost is within `1 + ε` of
    /// some solution in both objectives. Zero when
    /// [`AnytimeFront::complete`] is set.
    pub factor: ApproxFactor<C>,
    /// `true` if the search finished on its own rather than being stopped
    /// by its budget: then [`AnytimeFront::solutions`] is the whole
    /// Pareto-optimal front.
    pub complete: bool,
    /// Counters collected over all iterations, the two seeding searches
    /// included.
    pub stats: Stats,
}

/// The weighted-sum heuristic of the paper's Section 5 for
/// [`boa_star_anytime_ws`]: a scalar lower bound `heuristic(s)` on
/// `c1 + weight · c2` of every path from `s` to a goal.
///
/// The exact one is a single-objective Dijkstra from the goal with edge
/// cost `c1 + weight · c2`; any admissible bound is sound. A state for
/// which `heuristic` returns `None` is treated as a dead end.
#[derive(Debug, Clone, Copy)]
pub struct WeightedSum<W> {
    /// The weight `w > 0` of the second objective.
    pub weight: u32,
    /// The lower bound on `c1 + weight · c2`.
    pub heuristic: W,
}

/// A-BOA*ε: finds the Pareto-optimal front from `start` to a state that
/// satisfies `is_goal`, reporting ever better approximations of it.
///
/// After seeding with the two lexicographically optimal solutions and
/// after every iteration that lowers the overall approximation factor,
/// `on_front` gets the solutions found so far and an `ε` they achieve:
/// every Pareto-optimal cost `s` has a reported `r` with
/// `r_i <= (1 + ε) · s_i` (the paper's Theorem 1). The factors strictly
/// decrease and the last call of a complete run has `ε = 0`, the exact
/// front (Theorem 2). The search stops early when `budget` is exhausted;
/// the result then carries the last known factor.
///
/// `d` is the paper's divisor: each iteration targets `ε̂ / d` for the
/// interval it refines (the paper uses 4).
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is consistent in both objectives and zero on every goal.
///
/// # Panics
///
/// Panics if `d < 2`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, boa_star_anytime};
///
/// /// Four routes from 0 to 1.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [10, 100]), (1, [40, 45]), (1, [42, 41]), (1, [100, 10])]);
///         }
///     }
/// }
///
/// let mut sizes = Vec::new();
/// let result = boa_star_anytime(
///     &Routes,
///     ZeroHeuristic,
///     0,
///     |s| *s == 1,
///     4,
///     |front, _factor| sizes.push(front.len()),
///     |_: &_| false,
/// );
///
/// assert!(result.complete && result.factor.is_zero());
/// assert_eq!(result.solutions.len(), 4);
/// // The two lexicographic optima first, then the improved fronts.
/// assert_eq!(sizes.first(), Some(&2));
/// assert_eq!(sizes.last(), Some(&4));
/// ```
pub fn boa_star_anytime<G, H, C, P>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    d: u32,
    on_front: impl FnMut(&[Solution<G::State, [C; 2]>], ApproxFactor<C>),
    budget: impl Budget<Stats>,
) -> AnytimeFront<G::State, C>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let variant = Variant::<NoSum> {
        reuse: true,
        weighted: None,
    };

    run(
        graph, heuristic, start, is_goal, d, variant, on_front, budget,
    )
}

/// A-BOA*ε-w: [`boa_star_anytime`] with the paper's weighted-sum pruning.
///
/// Inside an interval, with `π'` the last solution found and `π''` the
/// interval's bottom-right solution, a path `π` is pruned when
/// `g1(π) + w · g2(π) + h_w(s(π)) >= f1(π'') + w · f2(π')`: every solution
/// through it lies on or above that line, where neither `π'` nor `π''`
/// leaves room for a new non-dominated one (Property 1). Same guarantees
/// and requirements as [`boa_star_anytime`], plus an admissible
/// `weighted.heuristic`.
///
/// # Panics
///
/// Panics if `d < 2`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{
///     Graph, WeightedSum, ZeroHeuristic, boa_star_anytime_ws, heuristic_fn,
/// };
///
/// /// Three routes from 0 to the goal 2 through 1.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         match state {
///             0 => out.push((1, [1, 1])),
///             1 => out.extend([(2, [1, 9]), (2, [5, 5]), (2, [9, 1])]),
///             _ => {}
///         }
///     }
/// }
///
/// // Exact c1 + c2 to the goal: 11 from 0, 10 from 1.
/// let sum = heuristic_fn(|s: &u8| Some([11_u32, 10, 0][usize::from(*s)]));
/// let weighted = WeightedSum { weight: 1, heuristic: sum };
/// let result = boa_star_anytime_ws(
///     &Routes,
///     ZeroHeuristic,
///     0,
///     |s| *s == 2,
///     4,
///     weighted,
///     |_, _| {},
///     |_: &_| false,
/// );
///
/// let costs: Vec<_> = result.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[2, 10], [6, 6], [10, 2]]);
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "the paper's inputs plus the anytime callback and budget, as in boa_star_anytime"
)]
pub fn boa_star_anytime_ws<G, H, W, C, P>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    d: u32,
    weighted: WeightedSum<W>,
    on_front: impl FnMut(&[Solution<G::State, [C; 2]>], ApproxFactor<C>),
    budget: impl Budget<Stats>,
) -> AnytimeFront<G::State, C>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    W: Heuristic<G::State, C>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let variant = Variant {
        reuse: true,
        weighted: Some(weighted),
    };

    run(
        graph, heuristic, start, is_goal, d, variant, on_front, budget,
    )
}

/// Basic-A-BOA*ε, the paper's baseline: [`boa_star_anytime`] without reuse.
///
/// Every iteration runs BOA*ε from scratch over the whole front with `ε`
/// the current overall factor divided by `d`, and keeps its result. Same
/// reports and guarantees as [`boa_star_anytime`]; slower, since every
/// iteration repeats the work of the last. Mainly for comparison.
///
/// # Panics
///
/// Panics if `d < 2`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, boa_star_anytime_basic};
///
/// /// Two routes from 0 to 1.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [1, 2]), (1, [2, 1])]);
///         }
///     }
/// }
///
/// let result =
///     boa_star_anytime_basic(&Routes, ZeroHeuristic, 0, |s| *s == 1, 4, |_, _| {}, |_: &_| false);
/// assert_eq!(result.solutions.len(), 2);
/// ```
pub fn boa_star_anytime_basic<G, H, C, P>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    d: u32,
    on_front: impl FnMut(&[Solution<G::State, [C; 2]>], ApproxFactor<C>),
    budget: impl Budget<Stats>,
) -> AnytimeFront<G::State, C>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let variant = Variant::<NoSum> {
        reuse: false,
        weighted: None,
    };

    run(
        graph, heuristic, start, is_goal, d, variant, on_front, budget,
    )
}

/// Which A-BOA*ε variant runs.
struct Variant<W> {
    /// `true`: refine one interval per iteration and keep the others
    /// (A-BOA*ε); `false`: rerun over the whole front (Basic-A-BOA*ε).
    reuse: bool,
    /// The weighted-sum pruning of A-BOA*ε-w, if enabled.
    weighted: Option<WeightedSum<W>>,
}

/// The weighted-sum heuristic type when the pruning is off; never queried.
struct NoSum;

impl<S, C: Cost> Heuristic<S, C> for NoSum {
    fn estimate(&mut self, _state: &S) -> Option<C> {
        Some(C::ZERO)
    }
}

/// Seeds the search and runs the iterations (Algorithm 2).
#[expect(
    clippy::too_many_arguments,
    reason = "the public entry points' arguments plus the variant"
)]
fn run<G, H, W, C, P>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
    d: u32,
    variant: Variant<W>,
    mut on_front: impl FnMut(&[Solution<G::State, [C; 2]>], ApproxFactor<C>),
    mut budget: impl Budget<Stats>,
) -> AnytimeFront<G::State, C>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    W: Heuristic<G::State, C>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    assert!(d >= 2, "the divisor d must be at least 2");

    let mut stats = Stats::default();
    let Some(tl) = lex::lex_first(graph, &mut heuristic, &start, &is_goal, 0, &mut stats) else {
        return AnytimeFront {
            solutions: Vec::new(),
            factor: ApproxFactor::Zero,
            complete: true,
            stats,
        };
    };
    let br = lex::lex_first(graph, &mut heuristic, &start, &is_goal, 1, &mut stats)
        .expect("a goal reachable in one order is reachable in the other");

    if tl.cost == br.cost {
        let solutions = alloc::vec![tl];
        on_front(&solutions, ApproxFactor::Zero);
        return AnytimeFront {
            solutions,
            factor: ApproxFactor::Zero,
            complete: true,
            stats,
        };
    }

    let mut search = Anytime {
        graph,
        heuristic,
        is_goal,
        weighted: variant.weighted,
        start,
        nodes: Vec::new(),
        solutions: alloc::vec![tl, br],
        intervals: Vec::new(),
        successors: Vec::new(),
        stats,
    };
    search.intervals = alloc::vec![search.seed_interval()];

    let complete = search.iterate(d, variant.reuse, &mut on_front, &mut budget);
    search.into_front(complete)
}

/// One path of the search, stored in the node arena.
struct Node<S, C> {
    /// The state the path ends in.
    state: S,
    /// Cost of the path.
    g: [C; 2],
    /// `g + h`.
    f: [C; 2],
    /// The weighted-sum heuristic at `state`; `None` if not in use or a
    /// dead end.
    sum: Option<C>,
    /// Arena index of the path without its last edge; `None` for the start.
    parent: Option<usize>,
}

/// An interval `⟨π_tl, π_br, Π⟩` (Definition 1) with its factor
/// (Definition 2).
struct Interval<C> {
    /// Index of the top-left solution in [`Anytime::solutions`].
    tl: usize,
    /// Index of the bottom-right solution in [`Anytime::solutions`].
    br: usize,
    /// Arena indices of the to-expand paths.
    paths: Vec<usize>,
    /// `ε̂`: zero when `paths` is empty.
    factor: ApproxFactor<C>,
}

/// Mutable state of one A-BOA*ε run.
struct Anytime<'a, G: Graph, H, W, C, P> {
    /// The graph being searched.
    graph: &'a G,
    /// Per-objective lower bounds on the remaining cost.
    heuristic: H,
    /// Goal test.
    is_goal: P,
    /// The weighted-sum pruning, if enabled.
    weighted: Option<WeightedSum<W>>,
    /// The start state, for Basic-A-BOA*ε's restarts.
    start: G::State,
    /// Every path generated; intervals and parents index into it.
    nodes: Vec<Node<G::State, C>>,
    /// Every solution found, the two seeds first. Intervals index into it.
    solutions: Vec<Solution<G::State, [C; 2]>>,
    /// `ILIST`, sorted by the first cost of the top-left solution; each
    /// interval's `br` is the next one's `tl` (Property 2).
    intervals: Vec<Interval<C>>,
    /// Reused successor buffer, empty between expansions.
    successors: Vec<(G::State, [C; 2])>,
    /// Effort counters.
    stats: Stats,
}

/// The per-call bookkeeping of Algorithm 3.
struct Refine<S, C> {
    /// `ε` of this call.
    eps: ApproxFactor<C>,
    /// Cost of the input interval's top-left solution.
    tl: [C; 2],
    /// Cost of the input interval's bottom-right solution.
    br: [C; 2],
    /// Cost of the last solution found, `π_tmp`; starts at `tl`.
    last: [C; 2],
    /// `g2_min` per state, reset for every call.
    g2_min: BTreeMap<S, C>,
    /// Paths pruned only by the tolerance since the last solution: `Π_tmp`.
    kept: Vec<usize>,
}

impl<G, H, W, C, P> Anytime<'_, G, H, W, C, P>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    W: Heuristic<G::State, C>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    /// The interval `⟨π_tl, π_br, {[s_start]}⟩` over the whole front, with
    /// a fresh start path.
    fn seed_interval(&mut self) -> Interval<C> {
        let start = self.start.clone();
        let paths: Vec<usize> = self
            .new_node(start, [C::ZERO; 2], None)
            .into_iter()
            .collect();
        self.stats.generated += paths.len();

        self.interval(0, 1, paths)
    }

    /// Runs iterations until every interval has factor zero (returns
    /// `true`) or the budget is exhausted (returns `false`).
    fn iterate(
        &mut self,
        d: u32,
        reuse: bool,
        on_front: &mut impl FnMut(&[Solution<G::State, [C; 2]>], ApproxFactor<C>),
        budget: &mut impl Budget<Stats>,
    ) -> bool {
        let mut reported = self.factor();
        on_front(&self.front(), reported);

        loop {
            let (index, factor) = self.widest();
            if factor.is_zero() {
                return true;
            }

            let eps = self.target(factor, d);
            let finished = if reuse {
                let interval = self.intervals.remove(index);
                let (parts, finished) = self.refine(&interval, eps, budget);
                self.intervals.splice(index..index, parts);
                finished
            } else {
                let seed = self.seed_interval();
                let (parts, finished) = self.refine(&seed, eps, budget);
                self.intervals = parts;
                finished
            };

            let now = self.factor();
            if now < reported {
                reported = now;
                on_front(&self.front(), reported);
            }

            if !finished {
                return false;
            }
        }
    }

    /// The `ε` to refine an interval of factor `factor` with: `factor / d`,
    /// or `1` for an infinite factor (see `docs/design.md`).
    fn target(&self, factor: ApproxFactor<C>, d: u32) -> ApproxFactor<C> {
        match factor {
            ApproxFactor::Infinite => {
                // Any positive cost over itself: ε = 1. The seeds' first costs
                // differ, so the bottom-right one is positive.
                let one = self.solutions[1].cost[0];
                ApproxFactor::Ratio { num: one, den: one }
            }
            finite => finite.divided(d),
        }
    }

    /// Algorithm 3: BOA*ε with `eps` on the to-expand paths of `interval`,
    /// returning the intervals between the solutions it finds, and `false`
    /// if the budget stopped it (then the unexplored paths stay in the last
    /// interval).
    fn refine(
        &mut self,
        interval: &Interval<C>,
        eps: ApproxFactor<C>,
        budget: &mut impl Budget<Stats>,
    ) -> (Vec<Interval<C>>, bool) {
        let tl = self.solutions[interval.tl].cost;
        let mut refine = Refine {
            eps,
            tl,
            br: self.solutions[interval.br].cost,
            last: tl,
            g2_min: BTreeMap::new(),
            kept: Vec::new(),
        };
        let mut open: BinaryHeap<_> = interval
            .paths
            .iter()
            .map(|&i| Reverse((self.nodes[i].f, i)))
            .collect();
        let mut parts = Vec::new();
        let mut top_left = interval.tl;
        let mut best: Option<C> = None;

        while let Some(Reverse((_, index))) = open.pop() {
            if budget.exhausted(&self.stats) {
                refine.kept.push(index);
                refine.kept.extend(open.drain().map(|Reverse((_, i))| i));
                parts.push(self.interval(top_left, interval.br, refine.kept));
                return (parts, false);
            }

            if self.is_dominated(index, best, &mut refine) {
                continue;
            }

            let node = &self.nodes[index];
            refine.g2_min.insert(node.state.clone(), node.g[1]);
            if (self.is_goal)(&self.nodes[index].state) {
                let solution = self.record(index);
                parts.push(self.interval(top_left, solution, mem::take(&mut refine.kept)));
                top_left = solution;
                refine.last = self.nodes[index].g;
                best = Some(self.nodes[index].g[1]);
                continue;
            }

            self.expand(index, best, &mut refine, &mut open);
        }

        parts.push(self.interval(top_left, interval.br, refine.kept));
        (parts, true)
    }

    /// Generates the children of the path at `index` and pushes those that
    /// are not dominated (Algorithm 3, lines 13 to 17).
    fn expand(
        &mut self,
        index: usize,
        best: Option<C>,
        refine: &mut Refine<G::State, C>,
        open: &mut BinaryHeap<Reverse<([C; 2], usize)>>,
    ) {
        self.stats.expanded += 1;

        let mut successors = mem::take(&mut self.successors);
        let g = self.nodes[index].g;
        self.graph
            .successors(&self.nodes[index].state, &mut successors);

        for (state, weight) in successors.drain(..) {
            let Some(child) = self.new_node(state, add(g, weight), Some(index)) else {
                continue;
            };

            if !self.is_dominated(child, best, refine) {
                open.push(Reverse((self.nodes[child].f, child)));
                self.stats.generated += 1;
            }
        }

        self.successors = successors;
    }

    /// `is_dominated` of Algorithm 3 (lines 20 to 27). Paths pruned only by
    /// the tolerance, not by BOA*'s own tests, are kept in
    /// [`Refine::kept`].
    fn is_dominated(
        &self,
        index: usize,
        best: Option<C>,
        refine: &mut Refine<G::State, C>,
    ) -> bool {
        let node = &self.nodes[index];
        let (g, f) = (node.g, node.f);

        if f[0] >= refine.br[0] || f[1] >= refine.tl[1] || self.below_line(node, refine) {
            return true;
        }

        let exact = refine.g2_min.get(&node.state).is_some_and(|&m| g[1] >= m)
            || best.is_some_and(|b| f[1] >= b);
        let tolerated = best.is_some_and(|b| refine.eps.covers(b, f[1]));
        if exact || tolerated {
            if !exact {
                refine.kept.push(index);
            }

            return true;
        }

        false
    }

    /// The weighted-sum test of Section 5: `g1 + w · g2 + h_w >= f1(π_br) +
    /// w · f2(π_tmp)`. Off when the pruning is disabled or the right-hand
    /// side saturates.
    fn below_line(&self, node: &Node<G::State, C>, refine: &Refine<G::State, C>) -> bool {
        let Some(weighted) = &self.weighted else {
            return false;
        };

        let Some(sum) = node.sum else {
            return true;
        };

        let w = weighted.weight;
        let line = refine.br[0].saturating_add(refine.last[1].saturating_times(w));
        if line.saturating_add(line) == line && line != C::ZERO {
            return false;
        }

        let reach = node.g[0]
            .saturating_add(node.g[1].saturating_times(w))
            .saturating_add(sum);

        reach >= line
    }

    /// Adds a path to the arena unless its state is a dead end, and
    /// returns its index.
    fn new_node(&mut self, state: G::State, g: [C; 2], parent: Option<usize>) -> Option<usize> {
        let h = self.heuristic.estimate(&state)?;
        let sum = match &mut self.weighted {
            Some(weighted) => weighted.heuristic.estimate(&state),
            None => None,
        };

        self.nodes.push(Node {
            state,
            g,
            f: add(g, h),
            sum,
            parent,
        });

        Some(self.nodes.len() - 1)
    }

    /// Records the goal path at `index` as a solution and returns its index.
    fn record(&mut self, index: usize) -> usize {
        let mut path = Vec::new();
        let mut current = Some(index);

        while let Some(i) = current {
            path.push(self.nodes[i].state.clone());
            current = self.nodes[i].parent;
        }

        path.reverse();
        self.solutions.push(Solution {
            cost: self.nodes[index].g,
            path,
        });

        self.solutions.len() - 1
    }

    /// The interval between solutions `tl` and `br` with to-expand paths
    /// `paths`, and its factor `max_π min(DF(tl, π), DF(br, π))`.
    fn interval(&self, tl: usize, br: usize, paths: Vec<usize>) -> Interval<C> {
        let (tl_cost, br_cost) = (self.solutions[tl].cost, self.solutions[br].cost);
        let factor = paths
            .iter()
            .map(|&i| {
                let f = self.nodes[i].f;
                ApproxFactor::domination(tl_cost, f).min(ApproxFactor::domination(br_cost, f))
            })
            .max()
            .unwrap_or(ApproxFactor::Zero);

        Interval {
            tl,
            br,
            paths,
            factor,
        }
    }

    /// The index and factor of the interval with the largest factor; the
    /// first on a tie.
    fn widest(&self) -> (usize, ApproxFactor<C>) {
        self.intervals
            .iter()
            .enumerate()
            .fold((0, ApproxFactor::Zero), |acc, (i, interval)| {
                if interval.factor > acc.1 {
                    (i, interval.factor)
                } else {
                    acc
                }
            })
    }

    /// The overall factor: the largest over all intervals (Theorem 1).
    fn factor(&self) -> ApproxFactor<C> {
        self.widest().1
    }

    /// The solutions at the intervals' corners, sorted by cost.
    fn front(&self) -> Vec<Solution<G::State, [C; 2]>> {
        let first = self.intervals.first().map(|interval| interval.tl);
        first
            .into_iter()
            .chain(self.intervals.iter().map(|interval| interval.br))
            .map(|i| self.solutions[i].clone())
            .collect()
    }

    /// Consumes the search and returns the result.
    fn into_front(self, complete: bool) -> AnytimeFront<G::State, C> {
        AnytimeFront {
            solutions: self.front(),
            factor: self.factor(),
            complete,
            stats: self.stats,
        }
    }
}
