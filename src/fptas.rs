//! A fully polynomial-time approximation scheme for the resource
//! constrained shortest path.
//!
//! Hansen 1980 and Warburton 1987 gave the first FPTASes for bicriterion
//! and constrained paths; this follows the label-trimming form of
//! Tsaggouris and Zaroliagis ("Multiobjective optimization: improved FPTAS
//! for shortest paths and non-linear objectives with applications", 2009),
//! as surveyed by Craveirinha et al., Algorithms 17(6):222, 2024, Section
//! 2.3.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;

use search_core::Cost;

use crate::cost::add;
use crate::epsilon::Epsilon;
use crate::graph::Graph;
use crate::output::{Solution, Stats};

/// An FPTAS for the constrained shortest path: a path from `start` to a
/// goal with second cost at most `budget` and first cost at most `1 + ε`
/// times the optimum [`constrained_shortest_path`](crate::constrained_shortest_path)
/// finds; `None` if no path fits the budget.
///
/// The reachable graph (`n` states, which must be finite) is relaxed in
/// `n - 1` Bellman-Ford rounds. Each state keeps at most one label per
/// box of first costs, the boxes growing geometrically by `1 + δ` with
/// `δ = ε / (2 (n - 1))`, and in each box the label with the smallest
/// second cost. So every state holds `O(n log(C) / ε)` labels, polynomial
/// in the input and `1 / ε`, unlike the exact and ε-searches, whose label
/// counts can grow exponentially. Trimming loses at most a factor `1 + δ`
/// per edge of the optimal path, and `(1 + δ)^(n-1) <= e^(ε/2) <= 1 + ε`
/// for `ε <= 2`; larger tolerances are capped at 2. `ε = 0` is exact.
///
/// Requirements, not checked at run time: non-negative costs and a finite
/// reachable graph. Costs must convert into `u64` for the box arithmetic.
/// Goal states are not expanded. Counters go to `stats`.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Epsilon, Graph, Stats, fptas_constrained};
///
/// /// Three routes from 0 to 1: (time, fuel).
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [10, 9]), (1, [30, 5]), (1, [31, 4])]);
///         }
///     }
/// }
///
/// let mut stats = Stats::default();
/// let tenth = Epsilon::new(1, 10);
/// let best = fptas_constrained(&Routes, 0, |s| *s == 1, 5, tenth, &mut stats).unwrap();
/// assert!(best.cost[1] <= 5 && best.cost[0] <= 33); // optimum 30, within 10 %
/// ```
pub fn fptas_constrained<G, C, P>(
    graph: &G,
    start: G::State,
    is_goal: P,
    budget: C,
    eps: Epsilon,
    stats: &mut Stats,
) -> Option<Solution<G::State, [C; 2]>>
where
    G: Graph<Weight = [C; 2]>,
    C: Cost + Into<u64>,
    P: Fn(&G::State) -> bool,
{
    let reachable = Reachable::explore(graph, start, &is_goal);
    let rounds = reachable.states.len().saturating_sub(1).max(1);
    let mut grid = Grid::new(eps, rounds);

    // Labels: (state index, cost, parent label).
    let mut labels: Vec<(usize, [C; 2], Option<usize>)> = alloc::vec![(0, [C::ZERO; 2], None)];
    let mut boxes: Vec<BTreeMap<u64, usize>> = alloc::vec![BTreeMap::new(); reachable.states.len()];
    boxes[0].insert(0, 0);
    let mut frontier = alloc::vec![0];
    stats.generated += 1;

    for _ in 0..rounds {
        let mut next = Vec::new();
        for &label in &frontier {
            let (from, cost, _) = labels[label];
            if reachable.goal[from] {
                continue;
            }

            stats.expanded += 1;
            for &(to, weight) in &reachable.edges[from] {
                let c = add(cost, weight);
                if c[1] > budget {
                    continue;
                }

                let key = grid.index(c[0].into());
                if boxes[to]
                    .get(&key)
                    .is_some_and(|&kept| labels[kept].1[1] <= c[1])
                {
                    continue;
                }

                labels.push((to, c, Some(label)));
                boxes[to].insert(key, labels.len() - 1);
                next.push(labels.len() - 1);
                stats.generated += 1;
            }
        }

        if next.is_empty() {
            break;
        }

        frontier = next;
    }

    let best = (0..reachable.states.len())
        .filter(|&s| reachable.goal[s])
        .flat_map(|s| boxes[s].values().copied())
        .min_by_key(|&l| labels[l].1)?;

    let mut path = Vec::new();
    let mut current = Some(best);
    while let Some(l) = current {
        path.push(reachable.states[labels[l].0].clone());
        current = labels[l].2;
    }

    path.reverse();

    Some(Solution {
        cost: labels[best].1,
        path,
    })
}

/// The reachable part of the graph, with dense state indices (the start
/// is 0).
struct Reachable<S, W> {
    /// States by index.
    states: Vec<S>,
    /// Outgoing edges by state index.
    edges: Vec<Vec<(usize, W)>>,
    /// Goal flags by state index.
    goal: Vec<bool>,
}

impl<S: Clone + Ord, W> Reachable<S, W> {
    /// Breadth-first exploration from `start`; goals are not expanded.
    fn explore<G: Graph<State = S, Weight = W>>(
        graph: &G,
        start: S,
        is_goal: impl Fn(&S) -> bool,
    ) -> Self {
        let mut index = BTreeMap::new();
        let mut reachable = Self {
            states: Vec::new(),
            edges: Vec::new(),
            goal: Vec::new(),
        };
        let mut queue = VecDeque::new();
        let mut buffer = Vec::new();

        index.insert(start.clone(), 0);
        reachable.push(start, &is_goal);
        queue.push_back(0);

        while let Some(i) = queue.pop_front() {
            if reachable.goal[i] {
                continue;
            }

            graph.successors(&reachable.states[i].clone(), &mut buffer);
            for (to, weight) in buffer.drain(..) {
                let j = *index.entry(to.clone()).or_insert_with(|| {
                    queue.push_back(reachable.states.len());
                    reachable.push(to, &is_goal);
                    reachable.states.len() - 1
                });
                reachable.edges[i].push((j, weight));
            }
        }

        reachable
    }

    /// Adds a state.
    fn push(&mut self, state: S, is_goal: &impl Fn(&S) -> bool) {
        self.goal.push(is_goal(&state));
        self.states.push(state);
        self.edges.push(Vec::new());
    }
}

/// Geometric boxes of first costs: box 0 holds 0, then `[t_k, t_(k+1))`
/// with `t_0 = 1` and `t_(k+1) = max(t_k + 1, floor(t_k (1 + δ)))`, so any
/// two costs in one box are within `1 + δ` of each other.
struct Grid {
    /// `δ = num / den`.
    num: u128,
    /// See `num`; positive.
    den: u128,
    /// The thresholds `t_0, t_1, ...` computed so far.
    thresholds: Vec<u128>,
}

impl Grid {
    /// The grid for tolerance `eps` (capped at 2) over `rounds` rounds.
    fn new(eps: Epsilon, rounds: usize) -> Self {
        let (mut num, den) = (u128::from(eps.num()), u128::from(eps.den()));
        if num > 2 * den {
            num = 2 * den;
        }

        let rounds = u128::try_from(rounds).expect("state counts fit in u128");
        Self {
            num,
            den: den * 2 * rounds,
            thresholds: alloc::vec![1],
        }
    }

    /// The box of first cost `value`.
    fn index(&mut self, value: u64) -> u64 {
        if value == 0 {
            return 0;
        }

        let value = u128::from(value);
        while *self.thresholds.last().expect("never empty") <= value {
            let t = *self.thresholds.last().expect("never empty");
            let grown = t
                .checked_mul(self.den + self.num)
                .map_or(u128::MAX, |p| p / self.den);
            self.thresholds.push(grown.max(t + 1));
        }

        let k = self.thresholds.partition_point(|&t| t <= value);
        u64::try_from(k).expect("box counts fit in u64")
    }
}
