//! Ratio objectives that do not lie on an additive Pareto front: the
//! linear fractional path and the cost/reliability path.
//!
//! Surveyed by Craveirinha et al., Algorithms 17(6):222, 2024, Section
//! 2.2.3(b) and (c): Ahuja 1988 (minimum cost–reliability ratio) and
//! Soroush 2008 (linear fractional path problem).

mod big;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::{AddAssign, SubAssign};

use search_core::{BigInt, BigUint, Cost};

use crate::graph::Graph;
use crate::output::Solution;

use big::Big;

/// The path from `start` to a goal that minimizes `Σ c1 / Σ c2` among
/// those with `Σ c2 > 0`; `None` if there is no such path. Goal states end
/// paths, and paths are simple.
///
/// Dinkelbach's parametric method: with `λ = num / den` the ratio of the
/// best path so far, find the simple path minimizing
/// `den · c1 - num · c2`; if that is negative, its ratio is smaller and
/// becomes the next `λ`, else `λ` is optimal. The first round uses `-c2`
/// alone to find some path with a positive denominator. The weights are
/// signed. A round sums them in `i128` when a bound on every sum it can
/// form fits, and as search-core [`BigInt`]s otherwise, so the result is
/// exact without a width limit.
/// Each round is a Bellman-Ford pass over the reachable graph when the
/// weights have no negative cycle, and an exhaustive search over simple
/// paths when they do (the inner problem is then NP-hard, as Soroush
/// notes). `λ` strictly falls, so the rounds are finitely many.
///
/// Requirements: non-negative costs that convert into `u64`, and a finite
/// reachable graph.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, min_ratio_path};
///
/// /// Two routes from 0 to 1: (cost, benefit).
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [2, 3]), (1, [5, 10])]);
///         }
///     }
/// }
///
/// let best = min_ratio_path(&Routes, 0, |s| *s == 1).unwrap();
/// assert_eq!(best.cost, [5, 10]); // 1/2 < 2/3
/// ```
pub fn min_ratio_path<G, C, P>(
    graph: &G,
    start: G::State,
    is_goal: P,
) -> Option<Solution<G::State, [C; 2]>>
where
    G: Graph<Weight = [C; 2]>,
    C: Cost + Into<u64>,
    P: Fn(&G::State) -> bool,
{
    let net = Explicit::explore(graph, start, &is_goal);
    dinkelbach(&net, false).map(|path| net.solution(&path))
}

/// Dinkelbach's rounds over `net`: the best path, as edge indices. With
/// `exact_only`, every round uses [`BigInt`] weights (for tests).
fn dinkelbach<S: Clone + Ord, C: Cost + Into<u64>>(
    net: &Explicit<S, C>,
    exact_only: bool,
) -> Option<Vec<usize>> {
    // λ = num / den; the first round, den = 0, maximizes c2. Path totals
    // sum fewer than 2^64 costs below 2^64, so they fit `u128`.
    let (mut num, mut den) = (1_u128, 0_u128);
    let mut best: Option<Vec<usize>> = None;

    loop {
        let small = if exact_only {
            None
        } else {
            net.small_weights(num, den)
        };
        let found = match small {
            Some(weights) => net.minimum(&weights).map(|(v, path)| (v < 0, path)),
            None => net
                .minimum(&net.exact_weights(num, den))
                .map(|(v, path)| (v.is_negative(), path)),
        };
        let (negative, path) = found?;
        if !negative {
            break;
        }

        let [c1, c2] = net.totals(&path);
        if c2 == 0 {
            break;
        }

        (num, den) = (c1, c2);
        best = Some(path);
    }

    best
}

/// A signed weight of the inner problem: `i128` when no sum can
/// overflow it, else [`BigInt`]. `Default` is zero.
trait Weight:
    Clone + Ord + Default + for<'a> AddAssign<&'a Self> + for<'a> SubAssign<&'a Self>
{
}

impl<W> Weight for W where
    W: Clone + Ord + Default + for<'a> AddAssign<&'a W> + for<'a> SubAssign<&'a W>
{
}

/// The outcome of a Bellman-Ford pass.
enum Relaxed<W> {
    /// No negative cycle: the best goal path, if any.
    Done(Option<(W, Vec<usize>)>),
    /// A negative cycle is reachable.
    NegativeCycle,
}

/// The explicit reachable graph, with dense indices (the start is 0).
struct Explicit<S, C> {
    /// States by index.
    states: Vec<S>,
    /// Goal flags.
    goal: Vec<bool>,
    /// Edges `(from, to, [c1, c2])`.
    edges: Vec<(usize, usize, [C; 2])>,
}

impl<S: Clone + Ord, C: Cost + Into<u64>> Explicit<S, C> {
    /// Explores everything reachable from `start`; goals are not expanded.
    fn explore<G: Graph<State = S, Weight = [C; 2]>>(
        graph: &G,
        start: S,
        is_goal: &impl Fn(&S) -> bool,
    ) -> Self {
        let mut index = BTreeMap::from([(start.clone(), 0)]);
        let mut net = Self {
            states: alloc::vec![start],
            goal: Vec::new(),
            edges: Vec::new(),
        };
        let mut buffer = Vec::new();
        let mut i = 0;

        while i < net.states.len() {
            let here = net.states[i].clone();
            let goal = is_goal(&here);
            net.goal.push(goal);
            if !goal {
                graph.successors(&here, &mut buffer);
                for (to, w) in buffer.drain(..) {
                    let j = *index.entry(to.clone()).or_insert_with(|| {
                        net.states.push(to);
                        net.states.len() - 1
                    });
                    net.edges.push((i, j, w));
                }
            }

            i += 1;
        }

        net
    }

    /// The edge weights `den · c1 - num · c2` in `i128`, or `None` if a
    /// weight or a sum the searches below can form might not fit.
    ///
    /// A Bellman-Ford distance or candidate is a walk of at most
    /// `(n + 2) · |edges|` edges (one per relaxation), and an exhaustive
    /// sum a subset of the edges, so `widest · |edges| · (n + 2)` bounds
    /// every sum.
    fn small_weights(&self, num: u128, den: u128) -> Option<Vec<i128>> {
        let term = |a: u128, c: C| {
            a.checked_mul(u128::from(c.into()))
                .and_then(|t| i128::try_from(t).ok())
        };
        let weights = self
            .edges
            .iter()
            .map(|e| term(den, e.2[0])?.checked_sub(term(num, e.2[1])?))
            .collect::<Option<Vec<i128>>>()?;
        let widest = weights.iter().map(|w| w.unsigned_abs()).max().unwrap_or(0);
        let steps = u128::try_from(self.edges.len())
            .ok()?
            .checked_mul(u128::try_from(self.states.len()).ok()? + 2)?;
        let bound = widest.checked_mul(steps)?;
        i128::try_from(bound).is_ok().then_some(weights)
    }

    /// The edge weights `den · c1 - num · c2`, exactly.
    fn exact_weights(&self, num: u128, den: u128) -> Vec<BigInt> {
        let term =
            |a: u128, c: C| BigInt::from(BigUint::from(a) * BigUint::from(u128::from(c.into())));
        self.edges
            .iter()
            .map(|e| term(den, e.2[0]) - term(num, e.2[1]))
            .collect()
    }

    /// The simple goal path of smallest total weight, as edge indices,
    /// with its weight; `None` if no goal is reachable.
    fn minimum<W: Weight>(&self, weights: &[W]) -> Option<(W, Vec<usize>)> {
        match self.bellman_ford(weights) {
            Relaxed::Done(found) => found,
            Relaxed::NegativeCycle => self.exhaustive(weights),
        }
    }

    /// Bellman-Ford from the start; `None` if a negative cycle is
    /// reachable, else the best goal path.
    fn bellman_ford<W: Weight>(&self, weights: &[W]) -> Relaxed<W> {
        let n = self.states.len();
        let mut dist: Vec<Option<W>> = alloc::vec![None; n];
        let mut parent: Vec<Option<usize>> = alloc::vec![None; n];
        dist[0] = Some(W::default());
        // Candidates are built here and swapped in, so a relaxation reuses
        // limbs instead of allocating.
        let mut candidate = W::default();

        for round in 0..=n {
            let mut changed = false;
            for (k, &(from, to, _)) in self.edges.iter().enumerate() {
                let Some(d) = &dist[from] else { continue };
                candidate.clone_from(d);
                candidate += &weights[k];
                if dist[to].as_ref().is_none_or(|current| candidate < *current) {
                    if round == n {
                        return Relaxed::NegativeCycle;
                    }

                    match &mut dist[to] {
                        Some(current) => core::mem::swap(current, &mut candidate),
                        slot @ None => *slot = Some(candidate.clone()),
                    }

                    parent[to] = Some(k);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        let best = (0..n)
            .filter(|&v| self.goal[v])
            .filter_map(|v| dist[v].clone().map(|d| (d, v)))
            .min_by(|a, b| a.0.cmp(&b.0));
        Relaxed::Done(best.map(|(d, v)| {
            let mut path = Vec::new();
            let mut at = v;
            while let Some(k) = parent[at] {
                path.push(k);
                at = self.edges[k].0;
            }

            path.reverse();
            (d, path)
        }))
    }

    /// Exhaustive search over simple paths, for weights with negative
    /// cycles.
    fn exhaustive<W: Weight>(&self, weights: &[W]) -> Option<(W, Vec<usize>)> {
        let mut best: Option<(W, Vec<usize>)> = None;
        let mut visited = alloc::vec![false; self.states.len()];
        let mut path = Vec::new();
        let mut sum = W::default();
        self.descend(0, &mut sum, weights, &mut visited, &mut path, &mut best);
        best
    }

    /// One step of [`Explicit::exhaustive`]; `sum` is the weight of
    /// `path`, updated in place and restored on return.
    fn descend<W: Weight>(
        &self,
        at: usize,
        sum: &mut W,
        weights: &[W],
        visited: &mut [bool],
        path: &mut Vec<usize>,
        best: &mut Option<(W, Vec<usize>)>,
    ) {
        if self.goal[at] {
            if best.as_ref().is_none_or(|(b, _)| *sum < *b) {
                *best = Some((sum.clone(), path.clone()));
            }

            return;
        }

        visited[at] = true;
        for (k, &(from, to, _)) in self.edges.iter().enumerate() {
            if from == at && !visited[to] {
                path.push(k);
                *sum += &weights[k];
                self.descend(to, sum, weights, visited, path, best);
                *sum -= &weights[k];
                path.pop();
            }
        }

        visited[at] = false;
    }

    /// The summed costs of a simple path, as `u128`.
    fn totals(&self, path: &[usize]) -> [u128; 2] {
        path.iter().fold([0, 0], |s, &k| {
            let w = self.edges[k].2;
            [
                s[0] + u128::from(w[0].into()),
                s[1] + u128::from(w[1].into()),
            ]
        })
    }

    /// The solution for a path of edge indices.
    fn solution(&self, path: &[usize]) -> Solution<S, [C; 2]> {
        let states = core::iter::once(0).chain(path.iter().map(|&k| self.edges[k].1));
        let cost = path.iter().fold([C::ZERO; 2], |s, &k| {
            [s[0] + self.edges[k].2[0], s[1] + self.edges[k].2[1]]
        });

        Solution {
            cost,
            path: states.map(|i| self.states[i].clone()).collect(),
        }
    }
}

/// A path reliability as an exact fraction `(num, den)`, each as 32-bit
/// limbs, least significant first.
pub type ExactReliability = (Vec<u32>, Vec<u32>);

/// A goal path of the reliability front with its label.
type Found<S, C> = (Solution<S, C>, Label<C>);

/// An edge reliability `num / den`, a probability in `(0, 1]`.
pub type Reliability = (u32, u32);

/// The path from `start` to a goal that minimizes its cost divided by its
/// reliability, the cost being the sum of the edges' first components and
/// the reliability the product of their second components; `None` if no
/// goal is reachable. Returns the solution with its reliability as an
/// exact product `(num, den)` of 32-bit limbs.
///
/// Ahuja 1988: a path beaten in both cost and reliability has no smaller
/// ratio, so the optimum is on the Pareto front of (cost, reliability).
/// The front is found by label setting in increasing cost, more reliable
/// labels first, with whole-label dominance per state (reliability only
/// falls along a path, like a bottleneck), and scanned comparing
/// `c1 · den1 · num2` with `c2 · den2 · num1` in exact multi-precision
/// arithmetic. Reliabilities must have `0 < num <= den`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, min_cost_reliability_ratio};
///
/// /// A cheap shaky link and a dearer safe one from 0 to 1.
/// struct Links;
///
/// impl Graph for Links {
///     type State = u8;
///     type Weight = (u32, (u32, u32));
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, (u32, (u32, u32)))>) {
///         if *state == 0 {
///             out.extend([(1, (2, (1, 4))), (1, (5, (9, 10)))]);
///         }
///     }
/// }
///
/// let (best, _) = min_cost_reliability_ratio(&Links, 0, |s| *s == 1).unwrap();
/// assert_eq!(best.cost, 5); // 5 / 0.9 < 2 / 0.25
/// ```
///
/// # Panics
///
/// Panics if a reliability has `num == 0` or `num > den`.
pub fn min_cost_reliability_ratio<G, C, P>(
    graph: &G,
    start: G::State,
    is_goal: P,
) -> Option<(Solution<G::State, C>, ExactReliability)>
where
    G: Graph<Weight = (C, Reliability)>,
    C: Cost + Into<u64>,
    P: Fn(&G::State) -> bool,
{
    let front = reliability_front(graph, start, &is_goal);
    let ratio = |(c, num, den): &(C, Big, Big)| (Big::from((*c).into()).times(den), num.clone());
    front
        .into_iter()
        .min_by(|a, b| {
            let ((ca, na), (cb, nb)) = (ratio(&a.1), ratio(&b.1));
            ca.times(&nb).cmp(&cb.times(&na)).then(a.1.0.cmp(&b.1.0))
        })
        .map(|(solution, (_, num, den))| (solution, (num.limbs(), den.limbs())))
}

/// A label of the reliability front: cost and reliability `num / den`.
type Label<C> = (C, Big, Big);

/// The non-dominated (cost, reliability) goal paths, by label setting.
fn reliability_front<G, C, P>(graph: &G, start: G::State, is_goal: &P) -> Vec<Found<G::State, C>>
where
    G: Graph<Weight = (C, Reliability)>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut nodes: Vec<(G::State, Label<C>, Option<usize>)> =
        alloc::vec![(start, (C::ZERO, Big::from(1), Big::from(1)), None)];
    let mut open: Vec<usize> = alloc::vec![0];
    let mut closed: BTreeMap<G::State, Vec<Label<C>>> = BTreeMap::new();
    let mut found: Vec<Found<G::State, C>> = Vec::new();
    let mut buffer = Vec::new();
    let order = |a: &Label<C>, b: &Label<C>| {
        a.0.cmp(&b.0)
            .then_with(|| b.1.times(&a.2).cmp(&a.1.times(&b.2)))
    };

    while !open.is_empty() {
        // The smallest cost, the most reliable among equal ones.
        let pick = (0..open.len())
            .min_by(|&i, &j| order(&nodes[open[i]].1, &nodes[open[j]].1))
            .expect("non-empty");
        let index = open.swap_remove(pick);
        let (here, label) = (nodes[index].0.clone(), nodes[index].1.clone());
        let known = found
            .iter()
            .map(|(_, l)| l)
            .chain(closed.get(&here).into_iter().flatten());
        if known.into_iter().any(|k| covers(k, &label)) {
            continue;
        }

        closed.entry(here.clone()).or_default().push(label.clone());
        if is_goal(&here) {
            let mut path = Vec::new();
            let mut at = Some(index);
            while let Some(i) = at {
                path.push(nodes[i].0.clone());
                at = nodes[i].2;
            }

            path.reverse();
            found.push((
                Solution {
                    cost: label.0,
                    path,
                },
                label,
            ));
            continue;
        }

        graph.successors(&here, &mut buffer);
        for (to, (c, (num, den))) in buffer.drain(..) {
            assert!(num > 0 && num <= den, "reliabilities must lie in (0, 1]");
            let child = (
                label.0 + c,
                label.1.times(&Big::from(u64::from(num))),
                label.2.times(&Big::from(u64::from(den))),
            );
            nodes.push((to, child, Some(index)));
            open.push(nodes.len() - 1);
        }
    }

    found
}

/// Returns `true` if label `k` is no worse than `label`: no dearer, no
/// less reliable.
fn covers<C: Ord>(k: &Label<C>, label: &Label<C>) -> bool {
    k.0 <= label.0 && k.1.times(&label.2).cmp(&label.1.times(&k.2)) != Ordering::Less
}

#[cfg(test)]
mod tests {
    //! Both weight representations of [`super::dinkelbach`]: the `i128`
    //! fast path and the exact [`BigInt`](search_core::BigInt) fallback.

    use alloc::vec::Vec;

    use proptest::prelude::*;
    use search_core::BigUint;

    use super::{Explicit, dinkelbach};
    use crate::graph::Graph;

    /// An edge list over small states.
    struct Edges(Vec<(u8, u8, [u64; 2])>);

    impl Graph for Edges {
        type State = u8;
        type Weight = [u64; 2];

        fn successors(&self, state: &u8, out: &mut Vec<(u8, [u64; 2])>) {
            out.extend(self.0.iter().filter(|e| e.0 == *state).map(|e| (e.1, e.2)));
        }
    }

    /// The reachable graph from 0 with goal 3.
    fn net(edges: Vec<(u8, u8, [u64; 2])>) -> Explicit<u8, u64> {
        Explicit::explore(&Edges(edges), 0, &|s: &u8| *s == 3)
    }

    /// The ratio `Σc1 / Σc2` of a path, as an exact fraction.
    fn ratio(net: &Explicit<u8, u64>, path: &[usize]) -> (BigUint, BigUint) {
        let [a, b] = net.totals(path);
        (BigUint::from(a), BigUint::from(b))
    }

    /// Costs near `u64::MAX` overflow the `i128` weights, so the second
    /// round falls back to exact weights and still finds the better route.
    #[test]
    fn large_costs_fall_back_to_exact() {
        let max = u64::MAX;
        let net = net(alloc::vec![(0, 3, [max, max]), (0, 3, [max - 1, max])]);
        let big = u128::from(max);
        assert!(net.small_weights(big, big).is_none());
        assert!(net.small_weights(1, 0).is_some());

        let path = dinkelbach(&net, false).expect("a path");
        assert_eq!(net.solution(&path).cost, [max - 1, max]);
        assert_eq!(dinkelbach(&net, true), Some(path));
    }

    /// A cost: small, or near `u64::MAX` to force the exact fallback.
    fn cost() -> impl Strategy<Value = u64> {
        prop_oneof![0..20_u64, (u64::MAX - 20)..=u64::MAX]
    }

    proptest! {
        /// The mixed rounds (`i128` where it fits) find a path of the same
        /// ratio as exact-only rounds, on graphs with cycles.
        #[test]
        fn fast_path_matches_exact(
            edges in proptest::collection::vec((0..4_u8, 0..4_u8, [cost(), cost()]), 1..9),
        ) {
            let net = net(edges);
            let (fast, exact) = (dinkelbach(&net, false), dinkelbach(&net, true));
            prop_assert_eq!(fast.is_some(), exact.is_some());
            if let (Some(fast), Some(exact)) = (fast, exact) {
                let ((n1, d1), (n2, d2)) = (ratio(&net, &fast), ratio(&net, &exact));
                prop_assert_eq!(n1 * d2, n2 * d1);
            }
        }
    }
}
