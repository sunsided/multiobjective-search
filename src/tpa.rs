//! TPA: an exact anytime method for multi-objective combinatorial problems
//! that splits the objective space into boxes and solves one
//! augmented-Tchebycheff subproblem per box.
//!
//! Domínguez-Ríos, Chicano, Alba, "Effective anytime algorithm for
//! multiobjective combinatorial optimization problems", arXiv 2403.08807
//! (Algorithms 2 to 4).

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::{BigUint, Budget, Cost, ExactArith, ExactNumber, Heuristic};

use crate::anytime_boa::{AnytimeFront, ApproxFactor};
use crate::cost::{add, weakly_dominates};
use crate::graph::Graph;
use crate::output::{Solution, Stats};
use crate::pareto::Front;

/// One augmented-Tchebycheff subproblem `P(u, ω)` (Algorithm 2, line 7) in
/// exact integer form.
///
/// With `ω_i = 1 / d_i` and `ϵ = 1 / K`, the objective
/// `max_i ω_i (f_i - z^I_i) + ϵ Σ_i ω_i (f_i - z^I_i)` times `K · Π d_i`
/// is the integer [`Scalarization::score`]
/// `K · max_i W_i (f_i - z^I_i) + Σ_i W_i (f_i - z^I_i)` with
/// `W_i = Π_j d_j / d_i`, and the paper's test `obj < 1` is
/// `score < `[`Scalarization::limit`].
#[derive(Debug, Clone)]
pub struct Scalarization<C, const N: usize> {
    /// The ideal point `z^I`.
    ideal: [C; N],
    /// `W_i`.
    weights: [BigUint; N],
    /// `K = 1 / ϵ`.
    k: BigUint,
    /// `K · Π d_i`.
    limit: BigUint,
}

impl<C: ExactArith<Exact = BigUint>, const N: usize> Scalarization<C, N> {
    /// The integer objective of a cost vector; components below the ideal
    /// point count as zero, which keeps it a lower bound for partial costs.
    #[must_use]
    pub fn score(&self, cost: &[C; N]) -> BigUint {
        let gaps: [BigUint; N] = core::array::from_fn(|i| {
            if cost[i] > self.ideal[i] {
                self.weights[i].clone() * cost[i].exact().abs_diff(&self.ideal[i].exact())
            } else {
                BigUint::zero()
            }
        });
        let max = gaps.iter().max().cloned().unwrap_or_else(BigUint::zero);
        let sum = gaps.into_iter().fold(BigUint::zero(), |s, g| s + g);

        self.k.clone() * max + sum
    }

    /// The score below which a point lies in the box (the paper's `obj < 1`).
    #[must_use]
    pub const fn limit(&self) -> &BigUint {
        &self.limit
    }
}

/// A single-objective solver for TPA's subproblems: the solution with the
/// smallest [`Scalarization::score`] over the whole feasible set, with its
/// cost vector, or `None` if there is none.
pub trait TchebycheffSolver<C, const N: usize> {
    /// The solution type.
    type Solution;

    /// Minimizes `scalarization.score` over all solutions.
    fn minimize(&mut self, scalarization: &Scalarization<C, N>)
    -> Option<(Self::Solution, [C; N])>;
}

/// What [`tpa`] returns: the solutions with their points, and whether the
/// front is complete.
pub type TpaResult<X, C, const N: usize> = (Vec<(X, [C; N])>, bool);

/// Adjacency lists of an explored graph.
type Adjacency<S, W> = BTreeMap<S, Vec<(S, W)>>;

/// A search zone: the half-open box `[lower, upper)` with a direction and
/// a priority.
#[derive(Debug, Clone)]
struct Zone<C, const N: usize> {
    /// Lower bound.
    lower: [C; N],
    /// Upper bound, exclusive.
    upper: [C; N],
    /// The numerator of the paper's (reduced) scaled priority; the
    /// denominator `Π (z^N_i - z^I_i)` is the same for every zone.
    priority: BigUint,
}

/// TPA (Algorithm 2) over any solver: the exact Pareto front of the problem
/// whose ideal point is `ideal` and whose costs all lie below `nadir`
/// (strictly, in every objective), found box by box.
///
/// The objective space starts as one box `[ideal, nadir)`. Each step
/// selects a box from the next direction's queue in turn (Algorithm 3), by
/// highest priority, and solves the augmented-Tchebycheff subproblem with
/// weights `ω_i = 1 / max(1, u_i - z^I_i)` and `ϵ = 1 / (2p(r - 1))`
/// (`r` the widest range). A score below the limit means a new
/// non-dominated point inside the box (Holzmann and Smith's Theorem 5); it
/// is reported, and every box whose upper bound lies above it is split by
/// the p-partition (Definition 1, Algorithm 4), boxes with comparable
/// upper bounds are joined (Definition 3), and the new boxes get the
/// reduced scaled priority (Definition 4). Otherwise the box is dropped.
/// Every point found is efficient, none twice, and the front is complete
/// when no box is left (Theorem 1).
///
/// `on_point` gets each new solution and point; `budget` sees the solver
/// and is checked before each subproblem.
///
/// Returns the solutions found and whether the front is complete.
///
/// # Panics
///
/// Panics if `ideal` is not below `nadir` in every objective.
pub fn tpa<C, S, const N: usize>(
    solver: &mut S,
    ideal: [C; N],
    nadir: [C; N],
    mut on_point: impl FnMut(&S::Solution, [C; N]),
    mut budget: impl FnMut(&S) -> bool,
) -> TpaResult<S::Solution, C, N>
where
    C: Cost + ExactArith<Exact = BigUint>,
    S: TchebycheffSolver<C, N>,
{
    assert!(
        (0..N).all(|i| ideal[i] < nadir[i]),
        "the ideal point must lie below the nadir bound"
    );
    let e = |c: C| c.exact();
    let range = |i: usize| e(nadir[i]).abs_diff(&e(ideal[i]));
    let widest = (0..N).map(range).max().unwrap_or_else(BigUint::one);
    let p = BigUint::from(u128::try_from(2 * N).expect("small"));
    // K = 2p(r - 1), at least 1.
    let k = (p * widest.abs_diff(&BigUint::one())).max(BigUint::one());

    let mut queues: Vec<Vec<Zone<C, N>>> = (0..N).map(|_| Vec::new()).collect();
    let whole = Zone {
        lower: ideal,
        upper: nadir,
        priority: volume(&ideal, &nadir),
    };
    queues[0].push(whole);
    let mut direction = N - 1;
    let mut found = Vec::new();

    while queues.iter().any(|q| !q.is_empty()) {
        if budget(solver) {
            return (found, false);
        }

        // Algorithm 3: the next non-empty direction, its highest priority.
        direction = (direction + 1) % N;
        while queues[direction].is_empty() {
            direction = (direction + 1) % N;
        }

        let pick = (0..queues[direction].len())
            .max_by(|&a, &b| {
                queues[direction][a]
                    .priority
                    .cmp(&queues[direction][b].priority)
            })
            .expect("non-empty");
        let upper = queues[direction][pick].upper;
        let scalarization = scalarize(&ideal, &upper, &k);

        let inside = (0..N).all(|i| ideal[i] < upper[i]);
        match solver.minimize(&scalarization) {
            Some((solution, point))
                if inside && scalarization.score(&point) < scalarization.limit =>
            {
                on_point(&solution, point);
                update(&mut queues, &point);
                found.push((solution, point));
            }
            _ => {
                queues[direction].swap_remove(pick);
            }
        }
    }

    (found, true)
}

/// The subproblem for a box with upper bound `upper`.
fn scalarize<C: Cost + ExactArith<Exact = BigUint>, const N: usize>(
    ideal: &[C; N],
    upper: &[C; N],
    k: &BigUint,
) -> Scalarization<C, N> {
    let d: [BigUint; N] = core::array::from_fn(|i| {
        upper[i]
            .exact()
            .abs_diff(&ideal[i].exact())
            .max(BigUint::one())
    });
    let product = d.iter().fold(BigUint::one(), |p, x| p * x.clone());
    let weights = core::array::from_fn(|i| {
        d.iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .fold(BigUint::one(), |p, (_, x)| p * x.clone())
    });

    Scalarization {
        ideal: *ideal,
        weights,
        k: k.clone(),
        limit: k.clone() * product,
    }
}

/// `Π (upper_i - lower_i)`, zero for an empty box.
fn volume<C: Cost + ExactArith<Exact = BigUint>, const N: usize>(
    lower: &[C; N],
    upper: &[C; N],
) -> BigUint {
    if (0..N).any(|i| lower[i] >= upper[i]) {
        return BigUint::zero();
    }

    (0..N).fold(BigUint::one(), |p, i| {
        p * upper[i].exact().abs_diff(&lower[i].exact())
    })
}

/// Algorithm 4: p-partition of every box whose upper bound lies above `z`,
/// then joins of boxes with comparable upper bounds per direction.
fn update<C: Cost + ExactArith<Exact = BigUint>, const N: usize>(
    queues: &mut [Vec<Zone<C, N>>],
    z: &[C; N],
) {
    let boxes: Vec<Zone<C, N>> = queues
        .iter_mut()
        .flat_map(|q| {
            let (split, kept): (Vec<_>, Vec<_>) = core::mem::take(q)
                .into_iter()
                .partition(|b| (0..N).all(|i| z[i] < b.upper[i]));
            *q = kept;
            split
        })
        .collect();

    for b in boxes {
        let hat: [C; N] = core::array::from_fn(|i| z[i].max(b.lower[i]));
        for (i, queue) in queues.iter_mut().enumerate() {
            // B_i(z) = [(l_1..l_i, ẑ_{i+1}..ẑ_p), (u_1..u_{i-1}, ẑ_i, u_{i+1}..u_p)].
            let lower = core::array::from_fn(|j| if j <= i { b.lower[j] } else { hat[j] });
            let upper = core::array::from_fn(|j| if j == i { hat[j] } else { b.upper[j] });
            let mut priority = volume(&lower, &upper);
            if priority == BigUint::zero() {
                continue;
            }

            if i == N - 1 {
                // The part [l, z] cannot hold a non-dominated point.
                priority = priority.abs_diff(&volume(&b.lower, z));
            }

            queue.push(Zone {
                lower,
                upper,
                priority,
            });
        }
    }

    for queue in queues.iter_mut() {
        join(queue);
    }
}

/// Joins boxes with comparable upper bounds (Definition 3) until none are.
fn join<C: Cost + ExactArith<Exact = BigUint>, const N: usize>(queue: &mut Vec<Zone<C, N>>) {
    'outer: loop {
        for a in 0..queue.len() {
            for b in 0..queue.len() {
                if a != b && weakly_dominates(&queue[a].upper, &queue[b].upper) {
                    let (small, large) = (queue[a].clone(), queue[b].clone());
                    let lower = core::array::from_fn(|i| small.lower[i].min(large.lower[i]));
                    let priority = volume(&lower, &large.upper);
                    let (hi, lo) = (a.max(b), a.min(b));
                    queue.swap_remove(hi);
                    queue.swap_remove(lo);
                    queue.push(Zone {
                        lower,
                        upper: large.upper,
                        priority,
                    });
                    continue 'outer;
                }
            }
        }

        return;
    }
}

/// The TPA subproblem on a graph: the path to a goal that minimizes the
/// scalarized score, by best-first search on the score of `g + h` with
/// per-state dominance pruning.
pub struct PathTchebycheff<'a, G: Graph, H, P> {
    /// The graph.
    graph: &'a G,
    /// An admissible heuristic.
    heuristic: H,
    /// The start state.
    start: <G as Graph>::State,
    /// Goal test.
    is_goal: P,
    /// Effort counters over all subproblems.
    stats: Stats,
}

impl<G: Graph, H, P> core::fmt::Debug for PathTchebycheff<'_, G, H, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PathTchebycheff")
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl<G, H, C, P, const N: usize> TchebycheffSolver<C, N> for PathTchebycheff<'_, G, H, P>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost + ExactArith<Exact = BigUint>,
    P: Fn(&G::State) -> bool,
{
    type Solution = Vec<G::State>;

    fn minimize(&mut self, scalarization: &Scalarization<C, N>) -> Option<(Vec<G::State>, [C; N])> {
        let mut nodes: Vec<(G::State, [C; N], Option<usize>)> = Vec::new();
        let mut open = BinaryHeap::new();
        let mut closed: BTreeMap<G::State, Front<C, N>> = BTreeMap::new();
        let mut successors = Vec::new();

        let h = self.heuristic.estimate(&self.start)?;
        nodes.push((self.start.clone(), [C::ZERO; N], None));
        open.push(Reverse((scalarization.score(&h), h, 0)));

        while let Some(Reverse((_, _, index))) = open.pop() {
            let (state, g) = (nodes[index].0.clone(), nodes[index].1);
            if closed.get(&state).is_some_and(|c| c.weakly_dominates(&g)) {
                continue;
            }

            closed.entry(state.clone()).or_default().insert(g);
            if (self.is_goal)(&state) {
                let mut path = Vec::new();
                let mut at = Some(index);
                while let Some(i) = at {
                    path.push(nodes[i].0.clone());
                    at = nodes[i].2;
                }

                path.reverse();
                return Some((path, g));
            }

            self.stats.expanded += 1;
            self.graph.successors(&state, &mut successors);
            for (next, w) in successors.drain(..) {
                let g_next = add(g, w);
                if closed
                    .get(&next)
                    .is_some_and(|c| c.weakly_dominates(&g_next))
                {
                    continue;
                }

                let Some(h) = self.heuristic.estimate(&next) else {
                    continue;
                };
                let f = add(g_next, h);
                nodes.push((next, g_next, Some(index)));
                open.push(Reverse((scalarization.score(&f), f, nodes.len() - 1)));
                self.stats.generated += 1;
            }
        }

        None
    }
}

/// TPA for shortest paths: the exact Pareto front from `start` to a state
/// that satisfies `is_goal`, found well spread over the objective space
/// first and completed if the budget allows.
///
/// The ideal point and a strict upper bound of the nadir point (the sum of
/// all reachable edge costs plus one, since Pareto-optimal paths are
/// simple) come from one exploration of the finite reachable graph and one
/// Dijkstra per objective; each subproblem is a [`PathTchebycheff`] search
/// with `heuristic`. `on_front` gets the solutions after every new point,
/// with an infinite factor until the front is complete (TPA has no
/// approximation guarantee before that) and zero when it is. `budget` sees
/// the expansions so far and is checked before each subproblem.
///
/// Requirements: non-negative integer-like costs (`ExactArith` into
/// `BigUint`), a finite reachable graph, an admissible `heuristic` zero on
/// goals.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, boa_star, tpa_paths};
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
///             out.extend([(1, [1, 9]), (1, [3, 5]), (1, [5, 3]), (1, [9, 1])]);
///         }
///     }
/// }
///
/// let result = tpa_paths(&Routes, ZeroHeuristic, 0, |s| *s == 1, |_, _| {}, |_: &_| false);
/// assert!(result.complete);
/// assert_eq!(result.solutions, boa_star(&Routes, ZeroHeuristic, 0, |s| *s == 1).solutions);
/// ```
pub fn tpa_paths<G, H, C, P, const N: usize>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    mut on_front: impl FnMut(&[Solution<G::State, [C; N]>], ApproxFactor<C>),
    mut budget: impl Budget<Stats>,
) -> AnytimeFront<G::State, C, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost + ExactArith<Exact = BigUint>,
    P: Fn(&G::State) -> bool,
{
    let Some((ideal, nadir)) = bounds(graph, &start, &is_goal) else {
        return AnytimeFront {
            solutions: Vec::new(),
            factor: ApproxFactor::Zero,
            complete: true,
            stats: Stats::default(),
        };
    };

    let mut solver = PathTchebycheff {
        graph,
        heuristic,
        start,
        is_goal,
        stats: Stats::default(),
    };
    let mut front: Vec<Solution<G::State, [C; N]>> = Vec::new();

    // Without any positive edge cost every path costs zero: one subproblem
    // finds the only point.
    if (0..N).any(|i| nadir[i] <= ideal[i]) {
        let only = solver.minimize(&scalarize(&ideal, &ideal, &BigUint::one()));
        front.extend(only.map(|(path, cost)| Solution { cost, path }));
        on_front(&front, ApproxFactor::Zero);
        return AnytimeFront {
            solutions: front,
            factor: ApproxFactor::Zero,
            complete: true,
            stats: solver.stats,
        };
    }

    let (_, complete) = tpa(
        &mut solver,
        ideal,
        nadir,
        |path, cost| {
            front.push(Solution {
                cost,
                path: path.clone(),
            });
            front.sort_by_key(|s| s.cost);
            on_front(&front, ApproxFactor::Infinite);
        },
        |solver| budget.exhausted(&solver.stats),
    );

    let factor = if complete {
        ApproxFactor::Zero
    } else {
        ApproxFactor::Infinite
    };
    if complete {
        on_front(&front, factor);
    }

    AnytimeFront {
        solutions: front,
        factor,
        complete,
        stats: solver.stats,
    }
}

/// The ideal point and a strict nadir bound of the paths from `start` to a
/// goal; `None` if no goal is reachable.
fn bounds<G, C, P, const N: usize>(
    graph: &G,
    start: &G::State,
    is_goal: &P,
) -> Option<([C; N], [C; N])>
where
    G: Graph<Weight = [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    // Explore, summing every edge's costs.
    let mut seen = alloc::collections::BTreeSet::from([start.clone()]);
    let mut stack = alloc::vec![start.clone()];
    let mut sum = [C::ZERO; N];
    let mut edges = Vec::new();
    let mut adjacency: Adjacency<G::State, [C; N]> = BTreeMap::new();
    while let Some(s) = stack.pop() {
        if is_goal(&s) {
            continue;
        }

        graph.successors(&s, &mut edges);
        for (t, w) in &edges {
            sum = add(sum, *w);
            if seen.insert(t.clone()) {
                stack.push(t.clone());
            }
        }

        adjacency.insert(s, core::mem::take(&mut edges));
    }

    let ideal: Option<[C; N]> = (0..N)
        .map(|i| {
            let mut dist: BTreeMap<G::State, C> = BTreeMap::new();
            let mut heap = BinaryHeap::from([Reverse((C::ZERO, start.clone()))]);
            while let Some(Reverse((d, s))) = heap.pop() {
                if dist.contains_key(&s) {
                    continue;
                }

                dist.insert(s.clone(), d);
                if is_goal(&s) {
                    return Some(d);
                }

                for (t, w) in adjacency.get(&s).into_iter().flatten() {
                    heap.push(Reverse((d + w[i], t.clone())));
                }
            }

            None
        })
        .collect::<Option<Vec<C>>>()
        .map(|v| core::array::from_fn(|i| v[i]));

    // Pareto-optimal paths are simple, so no cost exceeds the edge sums;
    // `2 · sum + unit` is strictly above them, with `unit` the smallest
    // positive edge cost (Cost has no one).
    let unit = adjacency
        .values()
        .flatten()
        .flat_map(|(_, w)| *w)
        .filter(|&c| c > C::ZERO)
        .min();
    ideal.map(|ideal| {
        let unit = unit.unwrap_or(C::ZERO);
        (ideal, sum.map(|c| c.saturating_add(c).saturating_add(unit)))
    })
}
