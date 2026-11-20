//! Reference-point search: Pareto-optimal solutions in order of a
//! weighted Chebyshev distance to an aspiration point, for interactive
//! use.
//!
//! The reference-point (achievement scalarizing) approach of Wierzbicki,
//! as used by the interactive methods surveyed by Craveirinha et al.,
//! Algorithms 17(6):222, 2024 (the INT class).

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};

use search_core::{Cost, Heuristic, WideMul};

use crate::constrained::rebuild;
use crate::cost::add;
use crate::graph::Graph;
use crate::output::{Solution, Stats};
use crate::pareto::Front;

/// A signed achievement value `max_i w_i · (c_i - r_i)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Achievement<C> {
    /// Negative: the cost beats the reference in every objective by at
    /// least this weighted margin.
    Below(C),
    /// Non-negative: the weighted shortfall in the worst objective.
    Above(C),
}

impl<C: Ord> PartialOrd for Achievement<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Ord> Ord for Achievement<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Below(a), Self::Below(b)) => b.cmp(a),
            (Self::Above(a), Self::Above(b)) => a.cmp(b),
            (Self::Below(_), Self::Above(_)) => Ordering::Less,
            (Self::Above(_), Self::Below(_)) => Ordering::Greater,
        }
    }
}

/// The achievement of `cost`: `max_i weights[i] · (cost[i] -
/// reference[i])`, the products saturating.
#[must_use]
pub fn achievement<C: Cost + WideMul, const N: usize>(
    cost: &[C; N],
    reference: &[C; N],
    weights: &[u32; N],
) -> Achievement<C> {
    (0..N)
        .map(|i| {
            let gap = cost[i].abs_diff(reference[i]).saturating_times(weights[i]);
            if cost[i] >= reference[i] {
                Achievement::Above(gap)
            } else {
                Achievement::Below(gap)
            }
        })
        .max()
        .unwrap_or(Achievement::Above(C::ZERO))
}

/// An open-list key: the achievement of `f`, `f`, and the arena index.
type Key<C, const N: usize> = (Achievement<C>, [C; N], usize);

/// An interactive reference-point search: an iterator over the
/// Pareto-optimal solutions from a start to a goal, the one closest to the
/// reference point first, by the weighted Chebyshev achievement
/// `max_i w_i · (c_i - r_i)`.
///
/// Best-first search on the achievement of `f = g + h`, a lower bound
/// since the achievement is monotone; ties go to the lexicographically
/// smaller `f`, so a solution comes before every one it dominates and only
/// non-dominated solutions are returned. Labels are dropped when weakly
/// dominated by a label expanded earlier at their state, or by a returned
/// solution. [`Iterator::next`] continues the same search: it is the "next
/// non-dominated solution" query of interactive methods.
/// [`ReferencePoint::aim`] moves the reference or the weights between
/// queries without losing the work done, and the solutions already
/// returned are never proposed again.
///
/// Requirements, not checked at run time: non-negative costs and an
/// admissible `heuristic`, zero on every goal; positive weights.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ReferencePoint, ZeroHeuristic};
///
/// /// Three routes from 0 to 1: (time, money).
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [1, 9]), (1, [5, 5]), (1, [9, 1])]);
///         }
///     }
/// }
///
/// let mut search = ReferencePoint::new(&Routes, ZeroHeuristic, 0, |s: &u8| *s == 1, [4, 4], [1, 1]);
/// assert_eq!(search.next().map(|s| s.cost), Some([5, 5])); // closest to (4, 4)
/// search.aim([0, 0], [1, 10]); // money matters ten times more now
/// assert_eq!(search.next().map(|s| s.cost), Some([9, 1]));
/// assert_eq!(search.next().map(|s| s.cost), Some([1, 9]));
/// assert_eq!(search.next(), None);
/// ```
pub struct ReferencePoint<'a, G: Graph, H, C, P, const N: usize> {
    /// The graph being searched.
    graph: &'a G,
    /// Lower bounds on the remaining cost.
    heuristic: H,
    /// Goal test.
    is_goal: P,
    /// The aspiration point.
    reference: [C; N],
    /// Positive weights per objective.
    weights: [u32; N],
    /// Every label: state, `g`, `f`, parent.
    nodes: Vec<(G::State, [C; N], Option<usize>)>,
    /// Open list on `(achievement of f, f, index)`.
    open: BinaryHeap<Reverse<Key<C, N>>>,
    /// `g` of the labels expanded per state.
    closed: BTreeMap<G::State, Front<C, N>>,
    /// Costs of the solutions returned so far.
    returned: Front<C, N>,
    /// Reused successor buffer.
    successors: Vec<(G::State, [C; N])>,
    /// Effort counters.
    stats: Stats,
}

impl<'a, G, H, C, P, const N: usize> ReferencePoint<'a, G, H, C, P, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    /// Starts a search from `start` toward `reference` with `weights`.
    pub fn new(
        graph: &'a G,
        mut heuristic: H,
        start: G::State,
        is_goal: P,
        reference: [C; N],
        weights: [u32; N],
    ) -> Self {
        let first = heuristic.estimate(&start);
        let mut search = Self {
            graph,
            heuristic,
            is_goal,
            reference,
            weights,
            nodes: Vec::new(),
            open: BinaryHeap::new(),
            closed: BTreeMap::new(),
            returned: Front::new(),
            successors: Vec::new(),
            stats: Stats::default(),
        };

        if let Some(h) = first {
            search.push(start, [C::ZERO; N], h, None);
        }

        search
    }

    /// Moves the reference point and the weights for the following
    /// queries; the open labels are re-ranked, nothing is searched again.
    pub fn aim(&mut self, reference: [C; N], weights: [u32; N]) {
        self.reference = reference;
        self.weights = weights;

        let entries: Vec<_> = self.open.drain().map(|Reverse((_, f, i))| (f, i)).collect();
        for (f, index) in entries {
            let key = achievement(&f, &self.reference, &self.weights);
            self.open.push(Reverse((key, f, index)));
        }
    }

    /// Effort counters so far.
    #[must_use]
    pub const fn stats(&self) -> Stats {
        self.stats
    }

    /// Adds a label and queues it.
    fn push(&mut self, state: G::State, g: [C; N], h: [C; N], parent: Option<usize>) {
        let f = add(g, h);
        let key = achievement(&f, &self.reference, &self.weights);
        self.nodes.push((state, g, parent));
        self.open.push(Reverse((key, f, self.nodes.len() - 1)));
        self.stats.generated += 1;
    }

    /// Generates the successors of the label at `index`.
    fn expand(&mut self, index: usize) {
        self.stats.expanded += 1;

        let mut successors = core::mem::take(&mut self.successors);
        let (state, g) = (self.nodes[index].0.clone(), self.nodes[index].1);
        self.graph.successors(&state, &mut successors);

        for (next, weight) in successors.drain(..) {
            let g_next = add(g, weight);
            if self
                .closed
                .get(&next)
                .is_some_and(|c| c.weakly_dominates(&g_next))
            {
                continue;
            }

            let Some(h) = self.heuristic.estimate(&next) else {
                continue;
            };

            if !self.returned.weakly_dominates(&add(g_next, h)) {
                self.push(next, g_next, h, Some(index));
            }
        }

        self.successors = successors;
    }
}

impl<G, H, C, P, const N: usize> Iterator for ReferencePoint<'_, G, H, C, P, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    type Item = Solution<G::State, [C; N]>;

    /// The next non-dominated solution, by achievement.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(Reverse((_, f, index))) = self.open.pop() {
            let (state, g) = (self.nodes[index].0.clone(), self.nodes[index].1);
            if self.returned.weakly_dominates(&f)
                || self
                    .closed
                    .get(&state)
                    .is_some_and(|c| c.weakly_dominates(&g))
            {
                continue;
            }

            self.closed.entry(state.clone()).or_default().insert(g);
            if (self.is_goal)(&state) {
                self.returned.insert(g);
                return Some(rebuild(&self.nodes, index));
            }

            self.expand(index);
        }

        None
    }
}
