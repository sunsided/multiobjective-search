//! The multi-objective label-correcting algorithm.
//!
//! Guerriero and Musmanno, "Label correcting methods to solve
//! multicriteria shortest path problems", JOTA 2001; the first of the two
//! classic approaches the survey of Salzman et al. (arXiv 2510.25504,
//! Section 3.1) names, beside the A*-based label setting.

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec::Vec;

use search_core::Cost;

use crate::cost::{add, weakly_dominates};
use crate::graph::Graph;
use crate::output::{Output, Solution, Stats};
use crate::pareto::Front;

/// Finds one path for every Pareto-optimal cost vector from `start` to a
/// state that satisfies `is_goal`, by label correcting.
///
/// Every state keeps its set of non-dominated labels. A FIFO queue holds
/// the states whose set changed; taking one, the labels added since its
/// last visit are extended along every edge and merged into the successors'
/// sets, which drops the labels they dominate and enqueues the successors
/// that changed. Unlike label setting, a label can be replaced after it
/// was extended. Search at a goal are kept but not extended, and a label
/// weakly dominated by a goal label is not extended either. The result is
/// the non-dominated goal labels.
///
/// Requirements: non-negative edge costs (no heuristic is used).
///
/// # Panics
///
/// Never: every queued state has a label set.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, label_correcting, namoa_star};
///
/// /// A diamond: two routes from 0 to 3.
/// struct Diamond;
///
/// impl Graph for Diamond {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         match state {
///             0 => out.extend([(1, [1, 3]), (2, [3, 1])]),
///             1 | 2 => out.push((3, [1, 1])),
///             _ => {}
///         }
///     }
/// }
///
/// let a = label_correcting(&Diamond, 0, |s| *s == 3);
/// let b = namoa_star(&Diamond, ZeroHeuristic, 0, |s| *s == 3);
/// assert_eq!(a.solutions, b.solutions);
/// ```
pub fn label_correcting<G, C, P, const N: usize>(
    graph: &G,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let mut search = Search::new(start);
    let mut goal_costs: Front<C, N> = Front::new();
    let mut buffer = Vec::new();

    while let Some(state) = search.queue.pop_front() {
        search.queued.remove(&state);
        let fresh = core::mem::take(
            &mut search
                .sets
                .get_mut(&state)
                .expect("queued states have labels")
                .1,
        );
        if is_goal(&state) {
            for &l in &fresh {
                goal_costs.insert(search.labels[l].1);
            }

            continue;
        }

        graph.successors(&state, &mut buffer);
        let edges = core::mem::take(&mut buffer);
        for &l in &fresh {
            // Skip labels replaced since they were added, or beaten by a goal.
            let live = search.sets[&state].0.contains(&l);
            if !live || goal_costs.weakly_dominates(&search.labels[l].1) {
                continue;
            }

            search.stats.expanded += 1;
            for (next, weight) in &edges {
                search.extend(l, next, *weight);
            }
        }
    }

    search.into_output(&is_goal)
}

/// The labels, per-state label sets and FIFO queue of a label-correcting
/// search.
struct Search<S, C, const N: usize> {
    /// Every label ever created as `(state, cost, parent label)`; indices
    /// are stable, replaced labels stay as parents of others.
    labels: Vec<(S, [C; N], Option<usize>)>,
    /// Per state: the live label indices, and those not yet extended.
    sets: BTreeMap<S, (Vec<usize>, Vec<usize>)>,
    /// States whose label set changed since their last visit.
    queue: VecDeque<S>,
    /// The states currently in `queue`, so none is queued twice.
    queued: BTreeSet<S>,
    /// Expansion and generation counters.
    stats: Stats,
}

impl<S: Clone + Ord, C: Cost, const N: usize> Search<S, C, N> {
    /// The search state before the first step: one zero-cost label at
    /// `start`, which is queued.
    fn new(start: S) -> Self {
        Self {
            labels: alloc::vec![(start.clone(), [C::ZERO; N], None)],
            sets: BTreeMap::from([(start.clone(), (alloc::vec![0], alloc::vec![0]))]),
            queue: VecDeque::from([start]),
            queued: BTreeSet::new(),
            stats: Stats {
                expanded: 0,
                generated: 1,
            },
        }
    }

    /// Extends label `l` along an edge to `next` with cost `weight`, and
    /// merges the new label into `next`'s set unless a live label there
    /// weakly dominates it; labels it dominates leave the set, and `next`
    /// is queued if it is not already.
    fn extend(&mut self, l: usize, next: &S, weight: [C; N]) {
        let cost = add(self.labels[l].1, weight);
        let labels = &self.labels;
        let entry = self.sets.entry(next.clone()).or_default();
        if entry
            .0
            .iter()
            .any(|&k| weakly_dominates(&labels[k].1, &cost))
        {
            return;
        }

        entry.0.retain(|&k| !weakly_dominates(&cost, &labels[k].1));
        entry.1.retain(|&k| !weakly_dominates(&cost, &labels[k].1));
        let id = self.labels.len();
        entry.0.push(id);
        entry.1.push(id);
        self.labels.push((next.clone(), cost, Some(l)));
        self.stats.generated += 1;
        if self.queued.insert(next.clone()) {
            self.queue.push_back(next.clone());
        }
    }

    /// The non-dominated live labels at goal states, in lexicographic cost
    /// order, with their paths traced back to the start.
    fn into_output(self, is_goal: impl Fn(&S) -> bool) -> Output<S, [C; N]> {
        let labels = self.labels;
        let mut ends: Vec<usize> = self
            .sets
            .iter()
            .filter(|(s, _)| is_goal(s))
            .flat_map(|(_, (live, _))| live.iter().copied())
            .collect();
        ends.sort_by_key(|&l| labels[l].1);

        let mut front = Front::new();
        let solutions = ends
            .into_iter()
            .filter(|&l| front.insert(labels[l].1))
            .map(|l| {
                let mut path = Vec::new();
                let mut at = Some(l);
                while let Some(i) = at {
                    path.push(labels[i].0.clone());
                    at = labels[i].2;
                }

                path.reverse();
                Solution {
                    cost: labels[l].1,
                    path,
                }
            })
            .collect();

        Output {
            solutions,
            stats: self.stats,
        }
    }
}
