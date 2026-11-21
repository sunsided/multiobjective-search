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

use search_core::{BigInt, BigUint, Cost, ExactNumber};

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
/// signed, so they are summed as search-core [`BigInt`]s, exactly and
/// without a width limit.
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
    // λ = num / den; the first round, den = 0, maximizes c2.
    let (mut num, mut den) = (BigUint::one(), BigUint::zero());
    let mut best: Option<Vec<usize>> = None;

    loop {
        let exact = |c: C| BigUint::from(u128::from(c.into()));
        let weight = |w: &[C; 2]| {
            BigInt::from(den.clone() * exact(w[0])) - BigInt::from(num.clone() * exact(w[1]))
        };
        let (value, path) = net.minimum(weight)?;
        if value >= BigInt::zero() {
            break;
        }

        let [c1, c2] = net.totals(&path);
        if c2 == BigUint::zero() {
            break;
        }

        (num, den) = (c1, c2);
        best = Some(path);
    }

    best.map(|path| net.solution(&path))
}

/// The outcome of a Bellman-Ford pass.
enum Relaxed {
    /// No negative cycle: the best goal path, if any.
    Done(Option<(BigInt, Vec<usize>)>),
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

    /// The simple goal path of smallest total `weight`, as edge indices,
    /// with its weight; `None` if no goal is reachable.
    fn minimum(&self, weight: impl Fn(&[C; 2]) -> BigInt) -> Option<(BigInt, Vec<usize>)> {
        let weights: Vec<BigInt> = self.edges.iter().map(|e| weight(&e.2)).collect();
        match self.bellman_ford(&weights) {
            Relaxed::Done(found) => found,
            Relaxed::NegativeCycle => self.exhaustive(&weights),
        }
    }

    /// Bellman-Ford from the start; `None` if a negative cycle is
    /// reachable, else the best goal path.
    fn bellman_ford(&self, weights: &[BigInt]) -> Relaxed {
        let n = self.states.len();
        let mut dist: Vec<Option<BigInt>> = alloc::vec![None; n];
        let mut parent: Vec<Option<usize>> = alloc::vec![None; n];
        dist[0] = Some(BigInt::zero());

        for round in 0..=n {
            let mut changed = false;
            for (k, &(from, to, _)) in self.edges.iter().enumerate() {
                let Some(d) = &dist[from] else { continue };
                let candidate = d.clone() + weights[k].clone();
                if dist[to].as_ref().is_none_or(|current| candidate < *current) {
                    if round == n {
                        return Relaxed::NegativeCycle;
                    }

                    dist[to] = Some(candidate);
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
    fn exhaustive(&self, weights: &[BigInt]) -> Option<(BigInt, Vec<usize>)> {
        let mut best: Option<(BigInt, Vec<usize>)> = None;
        let mut visited = alloc::vec![false; self.states.len()];
        let mut path = Vec::new();
        self.descend(
            0,
            &BigInt::zero(),
            weights,
            &mut visited,
            &mut path,
            &mut best,
        );
        best
    }

    /// One step of [`Explicit::exhaustive`].
    fn descend(
        &self,
        at: usize,
        sum: &BigInt,
        weights: &[BigInt],
        visited: &mut [bool],
        path: &mut Vec<usize>,
        best: &mut Option<(BigInt, Vec<usize>)>,
    ) {
        if self.goal[at] {
            if best.as_ref().is_none_or(|(b, _)| sum < b) {
                *best = Some((sum.clone(), path.clone()));
            }

            return;
        }

        visited[at] = true;
        for (k, &(from, to, _)) in self.edges.iter().enumerate() {
            if from == at && !visited[to] {
                path.push(k);
                self.descend(
                    to,
                    &(sum.clone() + weights[k].clone()),
                    weights,
                    visited,
                    path,
                    best,
                );
                path.pop();
            }
        }

        visited[at] = false;
    }

    /// The summed costs of a path, exactly.
    fn totals(&self, path: &[usize]) -> [BigUint; 2] {
        let exact = |c: C| BigUint::from(u128::from(c.into()));
        path.iter()
            .fold([BigUint::zero(), BigUint::zero()], |[a, b], &k| {
                let w = self.edges[k].2;
                [a + exact(w[0]), b + exact(w[1])]
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
