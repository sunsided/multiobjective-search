//! A-A*pex: anytime A*pex that reuses its search effort across tolerance
//! iterations (Zhang, Salzman, Felner, Hernández Ulloa, Koenig, "A-A*pex:
//! Efficient Anytime Approximate Multi-Objective Search", `SoCS` 2024,
//! Algorithm 2).

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::{Budget, Cost, Heuristic, WideMul};

use super::Excess;
use crate::anytime_boa::{AnytimeFront, ApproxFactor};
use crate::cost::{add, weakly_dominates};
use crate::epsilon::Epsilon;
use crate::graph::Graph;
use crate::output::{Solution, Stats};
use crate::pareto::Front;
use crate::pp_a::iterative::as_factor;

/// How A-A*pex starts each iteration (the paper's Section 4 variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnytimeApex {
    /// A-A*pex-reuse (Algorithm 2): start from the representative paths
    /// earlier iterations pruned or discarded while merging.
    #[default]
    Reuse,
    /// A-A*pex-reuse-enh: as [`AnytimeApex::Reuse`], and also prune a pair
    /// whose apex a representative expanded in an earlier iteration at its
    /// state weakly dominates.
    ReuseEnhanced,
    /// A-A*pex-restart: start every iteration from the start state, keeping
    /// only the solutions.
    Restart,
    /// A-A*pex-hybrid: restart until an iteration expands more than
    /// `threshold` times as many pairs as it prunes paths (the paper uses
    /// 5), then reuse.
    Hybrid {
        /// The expansions-to-pruned-paths ratio that switches to reuse.
        threshold: u32,
    },
}

/// A-A*pex: anytime approximate search for any number of objectives,
/// [`AnytimeApex::Reuse`] with [`apex_star_anytime_with`].
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Epsilon, Graph, ZeroHeuristic, apex_star_anytime};
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
/// let schedule = [[Epsilon::new(1, 10); 3], [Epsilon::ZERO; 3]];
/// let mut sizes = Vec::new();
/// let result = apex_star_anytime(&Routes, ZeroHeuristic, 0, |s| *s == 1, schedule, |front, _| sizes.push(front.len()), |_: &_| false);
/// assert!(result.complete && result.factor.is_zero());
/// assert_eq!(sizes, [2, 3]);
/// ```
pub fn apex_star_anytime<G, H, C, P, const N: usize>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    schedule: impl IntoIterator<Item = [Epsilon; N]>,
    on_front: impl FnMut(&[Solution<G::State, [C; N]>], ApproxFactor<C>),
    budget: impl Budget<Stats>,
) -> AnytimeFront<G::State, C, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    apex_star_anytime_with(
        graph,
        heuristic,
        start,
        is_goal,
        schedule,
        AnytimeApex::Reuse,
        on_front,
        budget,
    )
}

/// A-A*pex with a chosen variant.
///
/// Each iteration takes the next tolerance vector `ε` from `schedule` (the
/// paper's `getNextEps`; it uses 0.1 divided by `η` each time) and runs
/// A*pex's `findApproxPF` with `ε` (Algorithm 1 as the paper restates it:
/// solutions are plain paths, a pair is pruned when a solution is within
/// `ε` of its `f` in every objective or a pair expanded at its state in this
/// iteration weakly dominates it in objectives `1..N`, and new pairs merge
/// greedily into an open pair at their state when the result is
/// ε-bounded). The solutions persist across iterations. Whatever a pruning
/// or a merge drops but might still lead to a Pareto-optimal solution goes
/// to the list `Pruned` (lines 53-54, 57-59, 68-70), and the next iteration
/// starts from those paths as pairs `⟨c(π), π⟩` instead of from the start
/// (lines 42-45), which recovers the paths a merge represented. After an
/// iteration the solutions are an `ε`-approximate front (Theorem 1); once an
/// iteration ends with `Pruned` empty they are the exact front (Theorem 2)
/// and the search stops, possibly before the schedule ends. A schedule that
/// ends with zero tolerances always gets there.
///
/// `on_front` gets the solutions after every iteration that lowers the
/// factor (the iteration's largest tolerance, zero once exact). `budget` is
/// checked between iterations.
///
/// Requirements as for [`apex_star`](crate::apex_star): non-negative costs,
/// a consistent heuristic zero on goals.
///
/// # Panics
///
/// Panics if a solution path is malformed, which cannot happen.
#[expect(
    clippy::too_many_arguments,
    reason = "the anytime search inputs plus the variant"
)]
pub fn apex_star_anytime_with<G, H, C, P, const N: usize>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    schedule: impl IntoIterator<Item = [Epsilon; N]>,
    variant: AnytimeApex,
    mut on_front: impl FnMut(&[Solution<G::State, [C; N]>], ApproxFactor<C>),
    mut budget: impl Budget<Stats>,
) -> AnytimeFront<G::State, C, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let mut search = Anytime::new(graph, heuristic, is_goal, variant, start);
    let mut reported: Option<ApproxFactor<C>> = None;
    let mut complete = false;
    let mut reuse = matches!(variant, AnytimeApex::Reuse | AnytimeApex::ReuseEnhanced);

    for eps in schedule {
        if reported.is_some() && budget.exhausted(&search.stats) {
            break;
        }

        let (expanded, pruned) = search.iterate(eps, reuse);
        if let AnytimeApex::Hybrid { threshold } = variant {
            reuse = reuse || expanded > pruned.saturating_mul(threshold as usize);
        }

        complete = search.pruned.is_empty();
        let front = search.front();
        let factor = if complete {
            ApproxFactor::Zero
        } else {
            as_factor(widest(eps), &front)
        };
        if reported.is_none_or(|r| factor < r) {
            reported = Some(factor);
            on_front(&front, factor);
        }

        if complete {
            break;
        }
    }

    AnytimeFront {
        solutions: search.front(),
        factor: reported.unwrap_or(ApproxFactor::Infinite),
        complete,
        stats: search.stats,
    }
}

/// The largest tolerance of `eps`.
fn widest<const N: usize>(eps: [Epsilon; N]) -> Epsilon {
    eps.into_iter()
        .reduce(|a, b| {
            let larger =
                u64::from(b.num()) * u64::from(a.den()) > u64::from(a.num()) * u64::from(b.den());
            if larger { b } else { a }
        })
        .unwrap_or(Epsilon::ZERO)
}

/// An apex-path pair.
#[derive(Clone, Copy)]
struct Pair<C, const N: usize> {
    /// The apex.
    apex: [C; N],
    /// The representative path, an index into the path arena.
    rep: usize,
    /// The heuristic at the pair's state.
    h: [C; N],
    /// `false` once merged away.
    alive: bool,
}

/// The state of one A-A*pex run across its iterations.
struct Anytime<'a, G: Graph, H, C, P, const N: usize> {
    /// The graph.
    graph: &'a G,
    /// The heuristic.
    heuristic: H,
    /// Goal test.
    is_goal: P,
    /// Which variant runs.
    variant: AnytimeApex,
    /// The start state.
    start: G::State,
    /// Every path: state, cost, parent.
    paths: Vec<(G::State, [C; N], Option<usize>)>,
    /// Pairs of the current iteration.
    pairs: Vec<Pair<C, N>>,
    /// Open list on `(f, pair index)`.
    open: BinaryHeap<Reverse<([C; N], usize)>>,
    /// Open pairs per state, for merging.
    open_at: BTreeMap<G::State, Vec<usize>>,
    /// `G^Tr_cl` of this iteration: per state, the apexes expanded (compared
    /// on objectives `1..N`) with their representatives.
    closed: BTreeMap<G::State, Vec<([C; N], usize)>>,
    /// `C(s)` of the enhanced checks: per state, the costs of the
    /// representatives expanded in any iteration.
    expanded: BTreeMap<G::State, Front<C, N>>,
    /// Solution paths, mutually non-dominated.
    solutions: Vec<usize>,
    /// `Pruned`: paths that may still lead to Pareto-optimal solutions.
    pruned: Vec<usize>,
    /// The tolerances of the current iteration.
    eps: [Epsilon; N],
    /// Effort counters over all iterations.
    stats: Stats,
}

impl<'a, G, H, C, P, const N: usize> Anytime<'a, G, H, C, P, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    /// A run with `Pruned = {[s_start]}` (line 38).
    fn new(graph: &'a G, heuristic: H, is_goal: P, variant: AnytimeApex, start: G::State) -> Self {
        let mut search = Self {
            graph,
            heuristic,
            is_goal,
            variant,
            start: start.clone(),
            paths: Vec::new(),
            pairs: Vec::new(),
            open: BinaryHeap::new(),
            open_at: BTreeMap::new(),
            closed: BTreeMap::new(),
            expanded: BTreeMap::new(),
            solutions: Vec::new(),
            pruned: Vec::new(),
            eps: [Epsilon::ZERO; N],
            stats: Stats::default(),
        };
        search.paths.push((start, [C::ZERO; N], None));
        search.pruned.push(0);
        search
    }

    /// One iteration (lines 41-46); returns its expansions and the number
    /// of paths it pruned.
    fn iterate(&mut self, eps: [Epsilon; N], reuse: bool) -> (usize, usize) {
        self.eps = eps;
        self.pairs.clear();
        self.open.clear();
        self.open_at.clear();
        self.closed.clear();
        let before = self.stats.expanded;

        let seeds = core::mem::take(&mut self.pruned);
        let seeds = if reuse {
            seeds
        } else {
            self.paths.push((self.start.clone(), [C::ZERO; N], None));
            alloc::vec![self.paths.len() - 1]
        };

        for path in seeds {
            let (state, cost) = (self.paths[path].0.clone(), self.paths[path].1);
            if let Some(h) = self.heuristic.estimate(&state) {
                self.add_to_open(Pair {
                    apex: cost,
                    rep: path,
                    h,
                    alive: true,
                });
            }
        }

        self.find_approx_front();
        (self.stats.expanded - before, self.pruned.len())
    }

    /// `findApproxPF` (lines 8-21).
    fn find_approx_front(&mut self) {
        while let Some(Reverse((_, index))) = self.open.pop() {
            let pair = self.pairs[index];
            if !pair.alive {
                continue;
            }

            let state = self.paths[pair.rep].0.clone();
            if let Some(list) = self.open_at.get_mut(&state) {
                list.retain(|&i| i != index);
            }

            if self.is_dominated(&pair) {
                continue;
            }

            self.closed
                .entry(state.clone())
                .or_default()
                .push((pair.apex, pair.rep));
            if self.variant == AnytimeApex::ReuseEnhanced {
                let cost = self.paths[pair.rep].1;
                self.expanded.entry(state.clone()).or_default().insert(cost);
            }

            if (self.is_goal)(&state) {
                let cost = self.paths[pair.rep].1;
                let paths = &self.paths;
                self.solutions
                    .retain(|&s| !weakly_dominates(&cost, &paths[s].1));
                self.solutions.push(pair.rep);
                continue;
            }

            self.expand(pair);
        }
    }

    /// Lines 17-21: the children of `pair`.
    fn expand(&mut self, pair: Pair<C, N>) {
        self.stats.expanded += 1;

        let state = self.paths[pair.rep].0.clone();
        let mut edges = Vec::new();
        self.graph.successors(&state, &mut edges);
        for (next, weight) in edges {
            let Some(h) = self.heuristic.estimate(&next) else {
                continue;
            };

            let cost = add(self.paths[pair.rep].1, weight);
            self.paths.push((next, cost, Some(pair.rep)));
            let child = Pair {
                apex: add(pair.apex, weight),
                rep: self.paths.len() - 1,
                h,
                alive: true,
            };
            if !self.is_dominated(&child) {
                self.add_to_open(child);
                self.stats.generated += 1;
            }
        }
    }

    /// `isDominated` (lines 51-61), adding to `Pruned` what may still lead
    /// to a Pareto-optimal solution.
    fn is_dominated(&mut self, pair: &Pair<C, N>) -> bool {
        let (state, rep_cost) = (self.paths[pair.rep].0.clone(), self.paths[pair.rep].1);
        let f = add(pair.apex, pair.h);
        let rep_f = add(rep_cost, pair.h);

        let covering = self
            .solutions
            .iter()
            .map(|&s| self.paths[s].1)
            .find(|c| (0..N).all(|i| self.eps[i].within(c[i], f[i])));
        if let Some(c) = covering {
            if !weakly_dominates(&c, &rep_f) {
                self.pruned.push(pair.rep);
            }

            return true;
        }

        let dominator = self.closed.get(&state).and_then(|list| {
            list.iter()
                .find(|(x, _)| (1..N).all(|i| x[i] <= pair.apex[i]))
                .map(|&(_, r)| r)
        });
        if let Some(r) = dominator {
            if !weakly_dominates(&self.paths[r].1, &rep_cost) {
                self.pruned.push(pair.rep);
            }

            return true;
        }

        self.variant == AnytimeApex::ReuseEnhanced
            && self
                .expanded
                .get(&state)
                .is_some_and(|c| c.weakly_dominates(&pair.apex))
    }

    /// `addToOpen` (lines 62-73): merges into the first open pair at the
    /// state that gives an ε-bounded pair, sending the dropped
    /// representative to `Pruned` unless the kept one dominates it.
    fn add_to_open(&mut self, pair: Pair<C, N>) {
        let state = self.paths[pair.rep].0.clone();
        let candidates = self.open_at.get(&state).cloned().unwrap_or_default();
        for other in candidates {
            let Some(merged) = self.merge(pair, self.pairs[other]) else {
                continue;
            };

            self.pairs[other].alive = false;
            if let Some(list) = self.open_at.get_mut(&state) {
                list.retain(|&i| i != other);
            }

            let dropped = if merged.rep == pair.rep {
                self.pairs[other].rep
            } else {
                pair.rep
            };
            if !weakly_dominates(&self.paths[merged.rep].1, &self.paths[dropped].1) {
                self.pruned.push(dropped);
            }

            self.push(merged);
            return;
        }

        self.push(pair);
    }

    /// Queues a pair.
    fn push(&mut self, pair: Pair<C, N>) {
        let state = self.paths[pair.rep].0.clone();
        self.pairs.push(pair);
        let index = self.pairs.len() - 1;
        self.open.push(Reverse((add(pair.apex, pair.h), index)));
        self.open_at.entry(state).or_default().push(index);
    }

    /// The greedy merge of A*pex: the component-wise minimum apex and the
    /// representative with the smaller worst excess, if one is ε-bounded.
    fn merge(&self, a: Pair<C, N>, b: Pair<C, N>) -> Option<Pair<C, N>> {
        let apex = core::array::from_fn(|i| a.apex[i].min(b.apex[i]));
        let for_a = Pair {
            apex,
            rep: a.rep,
            h: a.h,
            alive: true,
        };
        let for_b = Pair {
            apex,
            rep: b.rep,
            h: a.h,
            alive: true,
        };

        match (self.excess(&for_a), self.excess(&for_b)) {
            (Some(x), Some(y)) if x.is_greater(y) => Some(for_b),
            (Some(_), _) => Some(for_a),
            (None, Some(_)) => Some(for_b),
            (None, None) => None,
        }
    }

    /// The worst relative excess of the representative's `f` over the
    /// apex's, or `None` if the pair is not ε-bounded.
    fn excess(&self, pair: &Pair<C, N>) -> Option<Excess<C>> {
        let f_rep = add(self.paths[pair.rep].1, pair.h);
        let f_apex = add(pair.apex, pair.h);
        if !(0..N).all(|i| self.eps[i].within(f_rep[i], f_apex[i])) {
            return None;
        }

        let ratios = (0..N)
            .filter(|&i| f_rep[i] != f_apex[i])
            .map(|i| Excess::Ratio {
                over: f_rep[i]
                    .abs_diff(f_apex[i])
                    .saturating_times(self.eps[i].den()),
                of: f_apex[i].saturating_times(self.eps[i].num()),
            });

        Some(ratios.fold(Excess::Zero, |x, y| if y.is_greater(x) { y } else { x }))
    }

    /// The solutions so far, sorted by cost.
    fn front(&self) -> Vec<Solution<G::State, [C; N]>> {
        let mut out: Vec<_> = self.solutions.iter().map(|&s| self.solution(s)).collect();
        out.sort_by_key(|s| s.cost);
        out
    }

    /// The solution for path `index`.
    fn solution(&self, index: usize) -> Solution<G::State, [C; N]> {
        let mut path = Vec::new();
        let mut at = Some(index);
        while let Some(i) = at {
            path.push(self.paths[i].0.clone());
            at = self.paths[i].2;
        }

        path.reverse();
        Solution {
            cost: self.paths[index].1,
            path,
        }
    }
}
