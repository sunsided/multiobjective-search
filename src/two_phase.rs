//! The two-phase method for bi-objective shortest paths: the supported
//! solutions by weighted sums first, then the non-supported ones between
//! them.
//!
//! Ulungu and Teghem 1995; Raith and Ehrgott 2009 ("A comparison of
//! solution strategies for biobjective shortest path problems"), as
//! surveyed by Craveirinha et al., Algorithms 17(6):222, 2024, Section
//! 2.2.1.

use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};

use search_core::{Cost, Heuristic, WideMul};

use crate::anytime_boa::lex::lex_first;
use crate::constrained::rebuild;
use crate::cost::add;
use crate::graph::Graph;
use crate::output::{Output, Solution, Stats};

/// Phase 1: the extreme supported solutions, those whose costs are the
/// vertices of the convex hull of the Pareto-optimal front, sorted by
/// cost.
///
/// Dichotomic search (Aneja and Nair 1979): starting from the two
/// lexicographic optima, for two neighbours `a` and `b` minimize the
/// weighted sum `(a2 - b2) · c1 + (b1 - a1) · c2`, whose level line runs
/// through `a` and `b`, and keep the minimizer if it lies strictly below
/// that line. Weighted sums are compared exactly with [`WideMul`], ties
/// broken lexicographically, so a minimizer is always a hull vertex.
///
/// Requirements, not checked at run time: non-negative costs and a
/// `heuristic` consistent in both objectives and zero on every goal.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, supported_solutions};
///
/// /// Four routes from 0 to 1; (5, 5) lies above the hull of the others.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [1, 9]), (1, [3, 3]), (1, [5, 5]), (1, [9, 1])]);
///         }
///     }
/// }
///
/// let out = supported_solutions(&Routes, ZeroHeuristic, 0, |s| *s == 1);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[1, 9], [3, 3], [9, 1]]);
/// ```
#[expect(
    clippy::needless_pass_by_value,
    reason = "takes the start by value like every other search in the crate"
)]
pub fn supported_solutions<G, H, C, P>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; 2]>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let mut stats = Stats::default();
    let solutions = phase_one(graph, &mut heuristic, &start, &is_goal, &mut stats);

    Output { solutions, stats }
}

/// The two-phase method: every Pareto-optimal cost, one path each, sorted
/// by cost.
///
/// Phase 1 is [`supported_solutions`]. Phase 2 searches each triangle
/// between two neighbouring supported solutions `a` and `b`: every other
/// Pareto-optimal cost lies strictly inside the box `a1 < c1 < b1`,
/// `b2 < c2 < a2`, found by BOA*'s label setting restricted to that box,
/// with `a`'s second cost as the initial bound.
///
/// Same requirements as [`supported_solutions`]; the result equals
/// [`boa_star`](crate::boa_star)'s.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, boa_star, two_phase};
///
/// /// Four routes from 0 to 1; (5, 5) is non-supported.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [1, 9]), (1, [2, 8]), (1, [5, 5]), (1, [9, 1])]);
///         }
///     }
/// }
///
/// let both = two_phase(&Routes, ZeroHeuristic, 0, |s| *s == 1);
/// let exact = boa_star(&Routes, ZeroHeuristic, 0, |s| *s == 1);
/// assert_eq!(both.solutions, exact.solutions);
/// ```
#[expect(
    clippy::needless_pass_by_value,
    reason = "takes the start by value like every other search in the crate"
)]
pub fn two_phase<G, H, C, P>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; 2]>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let mut stats = Stats::default();
    let supported = phase_one(graph, &mut heuristic, &start, &is_goal, &mut stats);

    let mut solutions = Vec::new();
    for pair in supported.windows(2) {
        let (a, b) = (pair[0].cost, pair[1].cost);
        solutions.extend(in_box(
            graph,
            &mut heuristic,
            &start,
            &is_goal,
            a,
            b,
            &mut stats,
        ));
    }

    solutions.extend(supported);
    solutions.sort_by_key(|s| s.cost);

    Output { solutions, stats }
}

/// The dichotomic search of phase 1.
fn phase_one<G, H, C, P>(
    graph: &G,
    heuristic: &mut H,
    start: &G::State,
    is_goal: &P,
    stats: &mut Stats,
) -> Vec<Solution<G::State, [C; 2]>>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let Some(left) = lex_first(graph, heuristic, start, is_goal, 0, stats) else {
        return Vec::new();
    };
    let right = lex_first(graph, heuristic, start, is_goal, 1, stats)
        .expect("reachable in one order, reachable in the other");
    if left.cost == right.cost {
        return alloc::vec![left];
    }

    let mut found = alloc::vec![left];
    let mut pending = alloc::vec![right];
    while let Some(b) = pending.pop() {
        let a = found.last().expect("starts with the left optimum").cost;
        let weights = [a[1].abs_diff(b.cost[1]), b.cost[0].abs_diff(a[0])];
        let best = weighted_best(graph, heuristic, start, is_goal, weights, stats)
            .expect("a solution exists");

        if weighted_cmp(weights, best.cost, a) == Ordering::Less {
            pending.push(b);
            pending.push(best);
        } else {
            found.push(b);
        }
    }

    found
}

/// Compares `w1 · x1 + w2 · x2` with `w1 · y1 + w2 · y2` exactly.
fn weighted_cmp<C: Cost + WideMul>(w: [C; 2], x: [C; 2], y: [C; 2]) -> Ordering {
    let part = |i: usize| (x[i].cmp(&y[i]), w[i].wide_mul(x[i].abs_diff(y[i])));
    let ((s1, m1), (s2, m2)) = (part(0), part(1));

    match (s1, s2) {
        (Ordering::Equal, s) | (s, Ordering::Equal) => s,
        (left, right) if left == right => left,
        // Opposite signs: the larger magnitude decides.
        (Ordering::Greater, _) => m1.cmp(&m2),
        (Ordering::Less, _) => m2.cmp(&m1),
    }
}

/// A* on the weighted sum `w · f`, ties broken lexicographically on `f`:
/// the goal path with the smallest weighted cost.
fn weighted_best<G, H, C, P>(
    graph: &G,
    heuristic: &mut H,
    start: &G::State,
    is_goal: &P,
    weights: [C; 2],
    stats: &mut Stats,
) -> Option<Solution<G::State, [C; 2]>>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let mut nodes: Vec<(G::State, [C; 2], Option<usize>)> = Vec::new();
    let mut open = BinaryHeap::new();
    let mut closed = BTreeSet::new();
    let mut successors = Vec::new();

    let h = heuristic.estimate(start)?;
    nodes.push((start.clone(), [C::ZERO; 2], None));
    open.push(Reverse(Weighted {
        weights,
        f: h,
        index: 0,
    }));
    stats.generated += 1;

    while let Some(Reverse(Weighted { index, .. })) = open.pop() {
        let (here, g, _) = nodes[index].clone();
        if !closed.insert(here.clone()) {
            continue;
        }

        if is_goal(&here) {
            return Some(rebuild(&nodes, index));
        }

        stats.expanded += 1;
        graph.successors(&here, &mut successors);
        for (next, weight) in successors.drain(..) {
            if closed.contains(&next) {
                continue;
            }

            let Some(h) = heuristic.estimate(&next) else {
                continue;
            };

            let g_next = add(g, weight);
            nodes.push((next, g_next, Some(index)));
            open.push(Reverse(Weighted {
                weights,
                f: add(g_next, h),
                index: nodes.len() - 1,
            }));
            stats.generated += 1;
        }
    }

    None
}

/// An open-list entry ordered by its weighted `f`, then by `f`, then by
/// arena index.
struct Weighted<C> {
    /// The scalarization weights, the same for every entry.
    weights: [C; 2],
    /// `g + h`.
    f: [C; 2],
    /// Arena index.
    index: usize,
}

impl<C: Cost + WideMul> PartialEq for Weighted<C> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<C: Cost + WideMul> Eq for Weighted<C> {}

impl<C: Cost + WideMul> PartialOrd for Weighted<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Cost + WideMul> Ord for Weighted<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        weighted_cmp(self.weights, self.f, other.f)
            .then(self.f.cmp(&other.f))
            .then(self.index.cmp(&other.index))
    }
}

/// Phase 2 for one triangle: the Pareto-optimal solutions strictly inside
/// the box between neighbouring supported solutions `a` and `b`, by BOA*'s
/// pruning with `f1 < b1` and `a2` as the initial solution bound.
fn in_box<G, H, C, P>(
    graph: &G,
    heuristic: &mut H,
    start: &G::State,
    is_goal: &P,
    left: [C; 2],
    right: [C; 2],
    stats: &mut Stats,
) -> Vec<Solution<G::State, [C; 2]>>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut nodes: Vec<(G::State, [C; 2], Option<usize>)> = Vec::new();
    let mut open = BinaryHeap::new();
    let mut g2_min: BTreeMap<G::State, C> = BTreeMap::new();
    let mut bound = left[1];
    let mut found = Vec::new();
    let mut successors = Vec::new();
    let outside = |f: [C; 2], bound: C| f[0] >= right[0] || f[1] >= bound;

    let Some(h) = heuristic.estimate(start) else {
        return found;
    };
    nodes.push((start.clone(), [C::ZERO; 2], None));
    open.push(Reverse((h, 0)));
    stats.generated += 1;

    while let Some(Reverse((f, index))) = open.pop() {
        let (here, g, _) = nodes[index].clone();
        if outside(f, bound) || g2_min.get(&here).is_some_and(|&m| g[1] >= m) {
            continue;
        }

        g2_min.insert(here.clone(), g[1]);
        if is_goal(&here) {
            bound = g[1];
            found.push(rebuild(&nodes, index));
            continue;
        }

        stats.expanded += 1;
        graph.successors(&here, &mut successors);
        for (next, weight) in successors.drain(..) {
            let g_next = add(g, weight);
            if g2_min.get(&next).is_some_and(|&m| g_next[1] >= m) {
                continue;
            }

            let Some(h) = heuristic.estimate(&next) else {
                continue;
            };

            let f = add(g_next, h);
            if !outside(f, bound) {
                nodes.push((next, g_next, Some(index)));
                open.push(Reverse((f, nodes.len() - 1)));
                stats.generated += 1;
            }
        }
    }

    // Pareto-optimality of `a` and `b` already keeps every solution found
    // strictly inside; the filter only documents the box.
    found.retain(|s| s.cost[0] > left[0] && s.cost[1] > right[1]);
    found
}
