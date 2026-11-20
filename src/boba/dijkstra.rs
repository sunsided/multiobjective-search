//! One-to-all single-objective Dijkstra, for BOBA*'s exact heuristics.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::Cost;

/// Exact shortest distances from one source, computed independently for
/// each of the two objectives.
pub(super) struct Distances<S, C> {
    /// `by_objective[i]` maps each reached state to its shortest distance
    /// in objective `i`. Both maps hold the same states.
    by_objective: [BTreeMap<S, C>; 2],
}

impl<S: Clone + Ord, C: Cost> Distances<S, C> {
    /// Runs Dijkstra from `source` once per objective over the edges that
    /// `neighbours` appends (its buffer is empty on entry).
    pub(super) fn new(source: &S, neighbours: impl Fn(&S, &mut Vec<(S, [C; 2])>)) -> Self {
        Self {
            by_objective: [0, 1].map(|objective| shortest(source, objective, &neighbours)),
        }
    }

    /// The distance vector of `state`, or `None` if it was not reached.
    pub(super) fn get(&self, state: &S) -> Option<[C; 2]> {
        Some([
            *self.by_objective[0].get(state)?,
            *self.by_objective[1].get(state)?,
        ])
    }
}

/// Dijkstra from `source` on objective `objective` alone.
fn shortest<S: Clone + Ord, C: Cost>(
    source: &S,
    objective: usize,
    neighbours: &impl Fn(&S, &mut Vec<(S, [C; 2])>),
) -> BTreeMap<S, C> {
    let mut dist = BTreeMap::new();
    let mut open = BinaryHeap::new();
    let mut edges = Vec::new();
    dist.insert(source.clone(), C::ZERO);
    open.push(Reverse((C::ZERO, source.clone())));

    while let Some(Reverse((d, state))) = open.pop() {
        if dist.get(&state).is_some_and(|&best| best < d) {
            continue;
        }

        neighbours(&state, &mut edges);
        for (next, weight) in edges.drain(..) {
            let candidate = d + weight[objective];
            if dist.get(&next).is_none_or(|&best| candidate < best) {
                dist.insert(next.clone(), candidate);
                open.push(Reverse((candidate, next)));
            }
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`Distances`].

    use alloc::vec::Vec;

    use super::Distances;

    /// Each objective gets its own shortest path; unreached states have no
    /// distance.
    #[test]
    fn per_objective_distances() {
        let edges = [(0_u8, 1_u8, [1_u32, 9]), (0, 2, [5, 1]), (2, 1, [5, 1])];
        let neighbours = |s: &u8, out: &mut Vec<(u8, [u32; 2])>| {
            out.extend(edges.iter().filter(|e| e.0 == *s).map(|e| (e.1, e.2)));
        };
        let dist = Distances::new(&0, neighbours);

        assert_eq!(dist.get(&0), Some([0, 0]));
        assert_eq!(dist.get(&1), Some([1, 2]));
        assert_eq!(dist.get(&3), None);
    }
}
