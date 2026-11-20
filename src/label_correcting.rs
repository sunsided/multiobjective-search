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
/// was extended. Labels at a goal are kept but not extended, and a label
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
    // Labels: state, cost, parent label.
    let mut labels: Vec<(G::State, [C; N], Option<usize>)> =
        alloc::vec![(start.clone(), [C::ZERO; N], None)];
    // Per state: live label indices, and those not yet extended.
    let mut sets: BTreeMap<G::State, (Vec<usize>, Vec<usize>)> =
        BTreeMap::from([(start.clone(), (alloc::vec![0], alloc::vec![0]))]);
    let mut queue = VecDeque::from([start]);
    let mut queued = BTreeSet::new();
    let mut goal_costs: Front<C, N> = Front::new();
    let mut buffer = Vec::new();
    let mut stats = Stats {
        expanded: 0,
        generated: 1,
    };

    while let Some(state) = queue.pop_front() {
        queued.remove(&state);
        let fresh =
            core::mem::take(&mut sets.get_mut(&state).expect("queued states have labels").1);
        if is_goal(&state) {
            for &l in &fresh {
                goal_costs.insert(labels[l].1);
            }

            continue;
        }

        graph.successors(&state, &mut buffer);
        let edges = core::mem::take(&mut buffer);
        for &l in &fresh {
            // Skip labels replaced since they were added, or beaten by a goal.
            let live = sets[&state].0.contains(&l);
            if !live || goal_costs.weakly_dominates(&labels[l].1) {
                continue;
            }

            stats.expanded += 1;
            for (next, weight) in &edges {
                let cost = add(labels[l].1, *weight);
                let entry = sets.entry(next.clone()).or_default();
                if entry
                    .0
                    .iter()
                    .any(|&k| weakly_dominates(&labels[k].1, &cost))
                {
                    continue;
                }

                entry.0.retain(|&k| !weakly_dominates(&cost, &labels[k].1));
                entry.1.retain(|&k| !weakly_dominates(&cost, &labels[k].1));
                labels.push((next.clone(), cost, Some(l)));
                let id = labels.len() - 1;
                entry.0.push(id);
                entry.1.push(id);
                stats.generated += 1;
                if queued.insert(next.clone()) {
                    queue.push_back(next.clone());
                }
            }
        }
    }

    let mut ends: Vec<usize> = sets
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

    Output { solutions, stats }
}
