//! Graphs that encode other problems as additive multi-objective paths:
//! multi-objective knapsacks and the minimum-label path objective.
//!
//! Both are classes surveyed by Craveirinha et al., Algorithms 17(6):222,
//! 2024 (Sections 2.2.2 and 2.2.3: Captivo et al. 2003, Pascoal et al.
//! 2013). Every search of the crate then solves them.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use search_core::Cost;

use crate::cost::add;
use crate::graph::Graph;

/// A multi-objective, multi-constraint 0/1 knapsack as a layered graph.
///
/// State `(i, used, took)`: items `0..i` decided, `used` the resources
/// they take, `took` whether item `i - 1` was taken. From each state an edge skips item `i` at cost `values[i]` (the
/// value lost) and, if the item fits every capacity, an edge takes it at
/// cost zero. A path from [`KnapsackGraph::start`] to a goal
/// ([`KnapsackGraph::is_goal`]) is a feasible selection, and minimizing the
/// lost values maximizes the kept ones (Captivo et al. 2003 solve the
/// bi-objective knapsack this way). [`KnapsackGraph::taken`] reads the
/// selection off a path.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{KnapsackGraph, ZeroHeuristic, namoa_star};
///
/// // Three items with weights and two values; capacity 5.
/// let knapsack = KnapsackGraph::new(vec![([3_u32], [5, 1]), ([3], [1, 5]), ([2], [2, 2])], [5]);
/// let out = namoa_star(&knapsack, ZeroHeuristic, knapsack.start(), |s| knapsack.is_goal(s));
/// let picks: Vec<_> = out.solutions.iter().map(|s| knapsack.taken(&s.path)).collect();
/// assert_eq!(picks, [vec![0, 2], vec![1, 2]]); // values (7, 3) and (3, 7)
/// ```
#[derive(Debug, Clone)]
pub struct KnapsackGraph<C, const N: usize, const M: usize> {
    /// Per item: resource use and values.
    items: Vec<([C; M], [C; N])>,
    /// Capacity per resource.
    capacity: [C; M],
}

impl<C: Cost, const N: usize, const M: usize> KnapsackGraph<C, N, M> {
    /// A knapsack with `items` as `(weights, values)` and `capacity` per
    /// resource.
    #[must_use]
    pub const fn new(items: Vec<([C; M], [C; N])>, capacity: [C; M]) -> Self {
        Self { items, capacity }
    }

    /// The start state: nothing decided, nothing used.
    #[must_use]
    pub const fn start(&self) -> (usize, [C; M], bool) {
        (0, [C::ZERO; M], false)
    }

    /// Returns `true` once every item is decided.
    #[must_use]
    pub fn is_goal(&self, state: &(usize, [C; M], bool)) -> bool {
        state.0 == self.items.len()
    }

    /// The indices of the items a path takes.
    #[must_use]
    pub fn taken(&self, path: &[(usize, [C; M], bool)]) -> Vec<usize> {
        path.iter().filter(|s| s.2).map(|s| s.0 - 1).collect()
    }
}

impl<C: Cost, const N: usize, const M: usize> Graph for KnapsackGraph<C, N, M> {
    type State = (usize, [C; M], bool);
    type Weight = [C; N];

    fn successors(&self, state: &Self::State, out: &mut Vec<(Self::State, [C; N])>) {
        let (i, used, _) = *state;
        let Some(&(weights, values)) = self.items.get(i) else {
            return;
        };

        out.push(((i + 1, used, false), values));
        let after = add(used, weights);
        if (0..M).all(|r| after[r] <= self.capacity[r]) {
            out.push(((i + 1, after, true), [C::ZERO; N]));
        }
    }
}

/// A graph whose objective `slot` counts the distinct labels a path uses:
/// the minimum-label path objective (Pascoal et al. 2013), made additive by
/// carrying the set of labels seen in the state.
///
/// `label` names each edge `(from, to, weight)`; the edge's cost in `slot`
/// is replaced by one if its label is new on the path, else zero. The
/// other objectives are kept. States are `(state, labels)`, so the search
/// space grows with the label combinations reached; use it for few labels.
/// Goals are the pairs whose state is a goal.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, LabelCount, ZeroHeuristic, namoa_star};
///
/// /// Two routes from 0 to 2: one short with two operators, one long
/// /// with one. The second objective is a placeholder for the count.
/// struct Lines;
///
/// impl Graph for Lines {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         match state {
///             0 => out.extend([(1, [1, 0]), (2, [9, 0])]),
///             1 => out.push((2, [1, 0])),
///             _ => {}
///         }
///     }
/// }
///
/// // Operator: "a" from 0, "b" from 1.
/// let counted = LabelCount::new(&Lines, 1, |from: &u8, _: &u8, _: &[u32; 2]| *from);
/// let out = namoa_star(&counted, ZeroHeuristic, counted.start(0), |s: &(u8, _)| s.0 == 2);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[2, 2], [9, 1]]);
/// ```
pub struct LabelCount<'a, G, F> {
    /// The underlying graph.
    graph: &'a G,
    /// The objective that counts labels.
    slot: usize,
    /// The label of an edge.
    label: F,
}

impl<'a, G, F> LabelCount<'a, G, F> {
    /// Counts the labels given by `label` in objective `slot` of `graph`.
    pub const fn new(graph: &'a G, slot: usize, label: F) -> Self {
        Self { graph, slot, label }
    }

    /// The start pair: `start` with no labels seen.
    pub const fn start<S, L>(&self, start: S) -> (S, BTreeSet<L>) {
        (start, BTreeSet::new())
    }
}

impl<G, F, L, C, const N: usize> Graph for LabelCount<'_, G, F>
where
    G: Graph<Weight = [C; N]>,
    F: Fn(&G::State, &G::State, &[C; N]) -> L,
    L: Clone + Ord,
    C: Cost + From<u8>,
{
    type State = (G::State, BTreeSet<L>);
    type Weight = [C; N];

    fn successors(&self, state: &Self::State, out: &mut Vec<(Self::State, [C; N])>) {
        let (from, seen) = state;
        let mut edges = Vec::new();
        self.graph.successors(from, &mut edges);

        for (to, mut weight) in edges {
            let label = (self.label)(from, &to, &weight);
            let mut labels = seen.clone();
            let new = labels.insert(label);
            weight[self.slot] = C::from(u8::from(new));
            out.push(((to, labels), weight));
        }
    }
}
