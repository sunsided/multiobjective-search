//! PP-A*: approximate bi-objective search over path pairs.
//!
//! Goldin and Salzman: "Approximate Bi-Criteria Search by Efficient
//! Representation of Subsets of the Pareto-Optimal Frontier", ICAPS 2021
//! (arXiv 2006.10302).
//!
//! A *path pair* at a state holds two paths to it: the top-left path `tl`
//! with the smallest first cost and the bottom-right path `br` with the
//! smallest second cost. Its *apex* `(c1(tl), c2(br))` is component-wise no
//! worse than every path the pair stands for. A pair is *bounded* when
//! `c1(br) <= (1 + ε1) · c1(tl)`: then `br`, extended by any suffix, is
//! within the tolerance of every path whose cost is at or above the apex.
//! The search is BOA* over path pairs, ordered by the apex `f`, with two
//! additions:
//!
//! - a new pair merges into an open pair at the same state when the merged
//!   pair is still bounded, so near-equal paths share one open entry;
//! - a goal pair merges into the last solution when the result is bounded.
//!
//! ```mermaid
//! graph TD
//!     pop[Pop min apex f] --> pruned{"(1+ε2) f2(br) >= last solution c2,<br/>or g2(br) >= g2_min(state)?"}
//!     pruned -- yes --> pop
//!     pruned -- no --> record[g2_min of state = g2 of br]
//!     record --> goal{Goal?}
//!     goal -- yes --> sol[Merge into last solution if bounded, else add] --> pop
//!     goal -- no --> expand[Extend tl and br, prune] --> merge[Merge into an open pair<br/>at the state if bounded, else push] --> pop
//! ```
//!
//! Differences from the paper, all of which keep its guarantee:
//!
//! - The bounded test is one-sided, on the first objective only, and the
//!   search returns `br` (the paper bounds both objectives and returns
//!   `tl`). The second objective is relaxed only by the solution pruning, so
//!   each tolerance is spent once: a pruned or merged path is covered by a
//!   returned one relative to its own apex, never through a chain of
//!   approximations. See [`pp_a_star`] for the argument.
//! - Several goal states are allowed, through a predicate.
//! - Merged pairs re-enter the open list as new entries; the old entry is
//!   skipped when popped.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::mem;

use search_core::{Cost, Heuristic};

use crate::cost::add;
use crate::epsilon::Epsilon;
use crate::graph::Graph;
use crate::output::{Output, Solution, Stats};

/// PP-A*: finds a set of solutions that covers the Pareto-optimal front
/// within a factor `1 + eps[i]` in each objective `i`.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is consistent in both objectives and zero on every goal.
///
/// Guarantees:
///
/// - for every Pareto-optimal cost vector `s` some returned solution `r`
///   has `r[0] <= (1 + eps[0]) · s[0]` and `r[1] <= (1 + eps[1]) · s[1]`;
/// - the returned solutions are real paths with their exact costs, mutually
///   non-dominated, sorted by the first objective strictly ascending and so
///   by the second strictly descending. Unlike [`boa_star_eps`](crate::boa_star_eps),
///   a returned solution need not be Pareto-optimal itself;
/// - `eps = [Epsilon::ZERO; 2]` returns exactly the cost vectors of
///   [`boa_star`](crate::boa_star).
///
/// # Why the guarantee holds
///
/// Call a pair *covered* when, for every path `p` at its state whose cost is
/// at or above its apex `a` and every suffix to a goal, some returned
/// solution is within the tolerances of the completed path. Pop order keeps
/// apex `f1` non-decreasing (merging only lowers the apex to that of one of
/// the two pairs, both still open). Then:
///
/// - a bounded pair that reaches a goal is covered by its `br`: `c1(br) <=
///   (1 + ε1) · a1` and `c2(br) = a2`; extending both by a suffix keeps the
///   bound, since `b + c <= (1 + ε1) · a + c <= (1 + ε1) · (a + c)`;
/// - merging lowers the apex component-wise, so the merged pair covers
///   both;
/// - a pair dropped by the solution test is covered by the last solution
///   `r`: `r[0] <= (1 + ε1) · ` its apex `f1`, which is at most the dropped
///   pair's `f1`, and `r[1] <= (1 + ε2) · f2`;
/// - a pair dropped by the per-state test has its apex at or above that of
///   a pair already expanded there.
///
/// Every bound is relative to an apex, which only goes down along these
/// steps, so the tolerances never compound.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Epsilon, Graph, ZeroHeuristic, boa_star, pp_a_star};
///
/// /// Four routes from 0 to 1 on a convex front.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [100, 40]), (1, [105, 30]), (1, [150, 20]), (1, [200, 19])]);
///         }
///     }
/// }
///
/// let eps = [Epsilon::new(1, 10), Epsilon::new(1, 10)];
/// let out = pp_a_star(&Routes, ZeroHeuristic, 0, |s| *s == 1, eps);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// // [105, 30] covers [100, 40]; [150, 20] covers [200, 19].
/// assert_eq!(costs, [[105, 30], [150, 20]]);
///
/// let exact = pp_a_star(&Routes, ZeroHeuristic, 0, |s| *s == 1, [Epsilon::ZERO; 2]);
/// assert_eq!(exact.solutions.len(), boa_star(&Routes, ZeroHeuristic, 0, |s| *s == 1).solutions.len());
/// ```
pub fn pp_a_star<G, H, C, P>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    eps: [Epsilon; 2],
) -> Output<G::State, [C; 2]>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
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
    while let Some(Reverse((_, index))) = search.open.pop() {
        search.process(index);
    }

    search.into_output()
}

/// One path in the path arena: its last state, its cost and its parent.
struct PathNode<S, C> {
    /// The state this path ends in.
    state: S,
    /// Cost vector of the path from the start.
    g: [C; 2],
    /// Arena index of the path without its last edge; `None` for the start.
    parent: Option<usize>,
}

/// A path pair: two paths to the same state.
#[derive(Clone, Copy)]
struct Pair<C> {
    /// Path arena index of the top-left path, which has the smallest first
    /// cost of the pair.
    tl: usize,
    /// Path arena index of the bottom-right path, which has the smallest
    /// second cost. Equal to `tl` for a pair of one path.
    br: usize,
    /// Heuristic estimate at the pair's state.
    h: [C; 2],
    /// `false` once the pair was merged into another; its open-list entry
    /// is then skipped.
    alive: bool,
}

/// Per-state bookkeeping.
struct StateInfo<C> {
    /// Second cost of the `br` path of the last pair expanded here, the
    /// smallest so far; `None` before the first.
    g2_min: Option<C>,
    /// Pair arena indices of this state's pairs on the open list.
    open: Vec<usize>,
}

/// Mutable state of one PP-A* run.
struct Search<'a, G: Graph, H, C, P> {
    /// The graph being searched.
    graph: &'a G,
    /// Per-objective lower bounds on the remaining cost.
    heuristic: H,
    /// Goal test.
    is_goal: P,
    /// Tolerances per objective.
    eps: [Epsilon; 2],
    /// Every path that is part of some pair; pairs and parents index into
    /// it.
    paths: Vec<PathNode<G::State, C>>,
    /// Every pair ever pushed; open-list entries index into it.
    pairs: Vec<Pair<C>>,
    /// Open list as a min-heap on `(apex f, pair index)`, compared
    /// lexicographically.
    open: BinaryHeap<Reverse<([C; 2], usize)>>,
    /// Per-state bookkeeping. Absent means "not reached yet".
    states: BTreeMap<G::State, StateInfo<C>>,
    /// Solution pairs as `(tl, br)` path indices, in the order found; their
    /// `br` second costs strictly descend.
    solutions: Vec<(usize, usize)>,
    /// Reused successor buffer, empty between expansions.
    successors: Vec<(G::State, [C; 2])>,
    /// Effort counters.
    stats: Stats,
}

impl<G, H, C, P> Search<'_, G, H, C, P>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    /// Pushes the start pair, unless the heuristic reports a dead end.
    fn push_start(&mut self, start: G::State) {
        if let Some(h) = self.heuristic.estimate(&start) {
            let path = self.new_path(start, [C::ZERO; 2], None);
            self.push(Pair {
                tl: path,
                br: path,
                h,
                alive: true,
            });
        }
    }

    /// Handles one pair taken from the open list: skip it if merged away,
    /// prune it, record a solution, or expand it.
    fn process(&mut self, index: usize) {
        let pair = self.pairs[index];
        if !pair.alive {
            return;
        }

        let state = self.paths[pair.br].state.clone();
        let info = self.info_mut(&state);
        info.open.retain(|&i| i != index);

        let g2 = self.paths[pair.br].g[1];
        if self.is_dominated(&state, g2, g2 + pair.h[1]) {
            return;
        }

        self.info_mut(&state).g2_min = Some(g2);

        if (self.is_goal)(&state) {
            self.add_solution(pair);
        } else {
            self.expand(pair);
        }
    }

    /// Returns `true` if a pair at `state` whose `br` has second cost `g2`
    /// and second estimate `f2` is covered by the last solution or by a
    /// pair already expanded at `state`.
    fn is_dominated(&self, state: &G::State, g2: C, f2: C) -> bool {
        self.best_g2()
            .is_some_and(|best| self.eps[1].within(best, f2))
            || self
                .states
                .get(state)
                .and_then(|info| info.g2_min)
                .is_some_and(|min| g2 >= min)
    }

    /// Second cost of the last solution, the smallest so far.
    fn best_g2(&self) -> Option<C> {
        self.solutions.last().map(|&(_, br)| self.paths[br].g[1])
    }

    /// Records a goal pair: merges it into the last solution if the merged
    /// pair is bounded, else adds it as a new solution.
    fn add_solution(&mut self, pair: Pair<C>) {
        if let Some(last) = self.solutions.last_mut() {
            // The last solution's tl has the smaller first cost (pop order),
            // the new br the smaller second cost (it passed the solution test).
            if self.eps[0].within(self.paths[pair.br].g[0], self.paths[last.0].g[0]) {
                last.1 = pair.br;
                return;
            }
        }

        self.solutions.push((pair.tl, pair.br));
    }

    /// Extends both paths of `pair` along every outgoing edge and inserts
    /// the surviving pairs.
    fn expand(&mut self, pair: Pair<C>) {
        self.stats.expanded += 1;

        let mut successors = mem::take(&mut self.successors);
        let from = self.paths[pair.br].state.clone();
        self.graph.successors(&from, &mut successors);

        for (state, weight) in successors.drain(..) {
            let Some(h) = self.heuristic.estimate(&state) else {
                continue;
            };

            let g_br = add(self.paths[pair.br].g, weight);
            if self.is_dominated(&state, g_br[1], g_br[1] + h[1]) {
                continue;
            }

            let br = self.new_path(state.clone(), g_br, Some(pair.br));
            let tl = if pair.tl == pair.br {
                br
            } else {
                let g_tl = add(self.paths[pair.tl].g, weight);
                self.new_path(state, g_tl, Some(pair.tl))
            };

            self.insert(Pair {
                tl,
                br,
                h,
                alive: true,
            });
        }

        self.successors = successors;
    }

    /// Merges `pair` into the first open pair at its state for which the
    /// merged pair is bounded, or pushes it as is.
    fn insert(&mut self, pair: Pair<C>) {
        let state = self.paths[pair.br].state.clone();
        let open = self
            .states
            .get(&state)
            .map_or(&[][..], |info| &info.open[..]);
        let found = open.iter().enumerate().find_map(|(position, &other)| {
            let merged = self.merge(pair, self.pairs[other]);
            self.is_bounded(merged).then_some((position, other, merged))
        });

        if let Some((position, other, merged)) = found {
            self.pairs[other].alive = false;
            self.info_mut(&state).open.swap_remove(position);
            self.push(merged);
            return;
        }

        self.push(pair);
    }

    /// The pair made of the smaller-first-cost `tl` and the
    /// smaller-second-cost `br` of `a` and `b`; ties go to `a`.
    fn merge(&self, a: Pair<C>, b: Pair<C>) -> Pair<C> {
        let first = |i: usize| self.paths[i].g[0];
        let second = |i: usize| self.paths[i].g[1];

        Pair {
            tl: if first(b.tl) < first(a.tl) {
                b.tl
            } else {
                a.tl
            },
            br: if second(b.br) < second(a.br) {
                b.br
            } else {
                a.br
            },
            h: a.h,
            alive: true,
        }
    }

    /// Returns `true` if `c1(br) <= (1 + ε1) · c1(tl)`.
    fn is_bounded(&self, pair: Pair<C>) -> bool {
        self.eps[0].within(self.paths[pair.br].g[0], self.paths[pair.tl].g[0])
    }

    /// Adds a pair to the pair arena, the open list and its state's open
    /// pairs.
    fn push(&mut self, pair: Pair<C>) {
        let index = self.pairs.len();
        let apex = [
            self.paths[pair.tl].g[0] + pair.h[0],
            self.paths[pair.br].g[1] + pair.h[1],
        ];
        let state = self.paths[pair.br].state.clone();

        self.pairs.push(pair);
        self.open.push(Reverse((apex, index)));
        self.info_mut(&state).open.push(index);
        self.stats.generated += 1;
    }

    /// Adds a path to the path arena and returns its index.
    fn new_path(&mut self, state: G::State, g: [C; 2], parent: Option<usize>) -> usize {
        self.paths.push(PathNode { state, g, parent });
        self.paths.len() - 1
    }

    /// The bookkeeping of `state`, created on first use.
    fn info_mut(&mut self, state: &G::State) -> &mut StateInfo<C> {
        self.states
            .entry(state.clone())
            .or_insert_with(|| StateInfo {
                g2_min: None,
                open: Vec::new(),
            })
    }

    /// Consumes the search and returns the non-dominated `br` paths of the
    /// solution pairs, sorted, with the counters.
    fn into_output(self) -> Output<G::State, [C; 2]> {
        let mut costs: Vec<_> = self
            .solutions
            .iter()
            .map(|&(_, br)| (self.paths[br].g, br))
            .collect();
        costs.sort_unstable();

        let mut solutions: Vec<Solution<G::State, [C; 2]>> = Vec::new();
        for (cost, br) in costs {
            // Sorted by first cost: keep a point only if it strictly lowers
            // the second cost, which drops every dominated or equal one.
            if solutions.last().is_none_or(|last| cost[1] < last.cost[1]) {
                solutions.push(self.solution(br, cost));
            }
        }

        Output {
            solutions,
            stats: self.stats,
        }
    }

    /// Rebuilds the path ending in path node `index`.
    fn solution(&self, index: usize, cost: [C; 2]) -> Solution<G::State, [C; 2]> {
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
