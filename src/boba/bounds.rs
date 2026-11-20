//! BOBA*'s preliminary searches: cost-bounded single-objective searches that
//! yield exact per-objective heuristics for both directions.
//!
//! Following Ahmadi et al. (Algorithm 1), four lexicographic best-first
//! searches run in two phases. With `ub1` the first cost of the path that is
//! lexicographically optimal in `(c2, c1)` and `ub2` the second cost of the
//! one optimal in `(c1, c2)`, every Pareto-optimal path has `c1 <= ub1` and
//! `c2 <= ub2`, so states only on costlier paths never need a heuristic:
//!
//! ```mermaid
//! graph TD
//!     a["A: start to goal, (c1, c2), Dijkstra"] -- "reaches goal: ub2" --> b2[A continues to c1 > ub1]
//!     b["B: goal to start, (c2, c1), Dijkstra"] -- "reaches start: ub1" --> a2[B continues to c2 > ub2]
//!     b2 --> d["D: goal backwards, c1, A* with A's distances, f1 <= ub1"]
//!     a2 --> c["C: start forwards, c2, A* with B's distances, f2 <= ub2"]
//!     d --> fwd["forward h = (D, B)"]
//!     c --> bwd["backward h = (A, C)"]
//! ```
//!
//! A state without a distance in some objective lies only on paths that
//! exceed that objective's bound. Such a path is strictly dominated (by the
//! `(c2, c1)`-optimal path for `c1 > ub1`, by the `(c1, c2)`-optimal one for
//! `c2 > ub2`), so the main searches may drop the state.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use search_core::Cost;

/// Exact per-objective distances for BOBA*'s two directions, restricted to
/// the states that can lie on a Pareto-optimal path.
pub(super) struct Bounds<S, C> {
    /// Search A: first-objective distance from the start.
    from_start_1: Lex<S, C>,
    /// Search C: second-objective distance from the start.
    from_start_2: Lex<S, C>,
    /// Search D: first-objective distance to the goal.
    to_goal_1: Lex<S, C>,
    /// Search B: second-objective distance to the goal.
    to_goal_2: Lex<S, C>,
}

impl<S: Clone + Ord, C: Cost> Bounds<S, C> {
    /// Runs the four preliminary searches. `successors` and `predecessors`
    /// append the outgoing and incoming edges of a state to an empty buffer.
    ///
    /// Returns `None` if the goal is unreachable from the start (or, for a
    /// graph whose predecessors do not mirror its successors, the start
    /// from the goal backwards).
    pub(super) fn new(
        start: &S,
        goal: &S,
        successors: impl Fn(&S, &mut Vec<(S, [C; 2])>),
        predecessors: impl Fn(&S, &mut Vec<(S, [C; 2])>),
    ) -> Option<Self> {
        let none = |_: &S| Some(C::ZERO);

        let mut a = Lex::new(start.clone(), 0);
        let ub2 = a.run_to(goal, &successors, &none)?;
        let mut b = Lex::new(goal.clone(), 1);
        // `None` only if the predecessors do not mirror the successors.
        let ub1 = b.run_to(start, &predecessors, &none)?;

        a.run_within(ub1, &successors, &none);
        b.run_within(ub2, &predecessors, &none);

        let mut c = Lex::new(start.clone(), 1);
        c.run_within(ub2, &successors, &|s| b.distance(s));
        let mut d = Lex::new(goal.clone(), 0);
        d.run_within(ub1, &predecessors, &|s| a.distance(s));

        Some(Self {
            from_start_1: a,
            from_start_2: c,
            to_goal_1: d,
            to_goal_2: b,
        })
    }

    /// The forward heuristic: distances from `state` to the goal, or `None`
    /// if `state` cannot lie on a Pareto-optimal path.
    pub(super) fn forward(&self, state: &S) -> Option<[C; 2]> {
        Some([
            self.to_goal_1.distance(state)?,
            self.to_goal_2.distance(state)?,
        ])
    }

    /// The backward heuristic: distances from the start to `state`, or
    /// `None` if `state` cannot lie on a Pareto-optimal path.
    pub(super) fn backward(&self, state: &S) -> Option<[C; 2]> {
        Some([
            self.from_start_1.distance(state)?,
            self.from_start_2.distance(state)?,
        ])
    }
}

/// A lexicographic best-first search from one source over one objective
/// pair: A* on the primary objective with a consistent heuristic, ties
/// broken by the secondary path cost.
struct Lex<S, C> {
    /// Index of the primary objective; the other one is secondary.
    primary: usize,
    /// Min-heap on `((f_primary, g_secondary), g_primary, state)`.
    open: BinaryHeap<Reverse<OpenEntry<S, C>>>,
    /// Per reached state: the lexicographically best `(g_primary,
    /// g_secondary)` pushed, and whether the state was expanded, which
    /// makes that cost exact.
    labels: BTreeMap<S, Label<C>>,
    /// Reused edge buffer, empty between expansions.
    edges: Vec<(S, [C; 2])>,
}

/// The bookkeeping of one state reached by a [`Lex`] search.
struct Label<C> {
    /// Best `(g_primary, g_secondary)` pushed so far.
    g: [C; 2],
    /// `true` once the state was expanded; `g` is then exact.
    closed: bool,
}

/// One open-list entry of [`Lex`]: the key `(f_primary, g_secondary)`, the
/// primary path cost and the state.
type OpenEntry<S, C> = ([C; 2], C, S);

impl<S: Clone + Ord, C: Cost> Lex<S, C> {
    /// Creates a search from `source` ordered by objective `primary` first.
    fn new(source: S, primary: usize) -> Self {
        let mut open = BinaryHeap::new();
        let mut labels = BTreeMap::new();
        labels.insert(
            source.clone(),
            Label {
                g: [C::ZERO; 2],
                closed: false,
            },
        );
        open.push(Reverse(([C::ZERO; 2], C::ZERO, source)));

        Self {
            primary,
            open,
            labels,
            edges: Vec::new(),
        }
    }

    /// The exact primary distance of `state` if it was expanded.
    fn distance(&self, state: &S) -> Option<C> {
        self.labels
            .get(state)
            .and_then(|label| label.closed.then_some(label.g[0]))
    }

    /// Expands states until `target` is expanded and returns the secondary
    /// cost of the lexicographically optimal path to it, or `None` if
    /// `target` is unreachable.
    fn run_to(
        &mut self,
        target: &S,
        neighbours: &impl Fn(&S, &mut Vec<(S, [C; 2])>),
        heuristic: &impl Fn(&S) -> Option<C>,
    ) -> Option<C> {
        loop {
            let ([_, g_secondary], state) = self.expand_next(neighbours, heuristic)?;
            if state == *target {
                return Some(g_secondary);
            }
        }
    }

    /// Expands every state whose primary `f` is at most `bound`.
    fn run_within(
        &mut self,
        bound: C,
        neighbours: &impl Fn(&S, &mut Vec<(S, [C; 2])>),
        heuristic: &impl Fn(&S) -> Option<C>,
    ) {
        while self
            .open
            .peek()
            .is_some_and(|Reverse((key, _, _))| key[0] <= bound)
        {
            self.expand_next(neighbours, heuristic);
        }
    }

    /// Pops and expands the next unexpanded state; returns its key and the
    /// state, or `None` once the open list is empty.
    fn expand_next(
        &mut self,
        neighbours: &impl Fn(&S, &mut Vec<(S, [C; 2])>),
        heuristic: &impl Fn(&S) -> Option<C>,
    ) -> Option<([C; 2], S)> {
        loop {
            let Reverse((key, g_primary, state)) = self.open.pop()?;
            let label = self
                .labels
                .get_mut(&state)
                .expect("pushed states have a label");
            if label.closed {
                continue;
            }

            label.closed = true;
            self.push_neighbours(&state, [g_primary, key[1]], neighbours, heuristic);

            return Some((key, state));
        }
    }

    /// Pushes every neighbour of `state`, reached at cost `g` (primary,
    /// secondary), that improves on its best pushed cost and has a
    /// heuristic value.
    fn push_neighbours(
        &mut self,
        state: &S,
        g: [C; 2],
        neighbours: &impl Fn(&S, &mut Vec<(S, [C; 2])>),
        heuristic: &impl Fn(&S) -> Option<C>,
    ) {
        let secondary = 1 - self.primary;
        neighbours(state, &mut self.edges);

        for (next, weight) in self.edges.drain(..) {
            let g_next = [g[0] + weight[self.primary], g[1] + weight[secondary]];
            let label = self.labels.get(&next);
            if label.is_some_and(|label| label.closed || label.g <= g_next) {
                continue;
            }

            let Some(h) = heuristic(&next) else {
                continue;
            };

            self.labels.insert(
                next.clone(),
                Label {
                    g: g_next,
                    closed: false,
                },
            );
            self.open
                .push(Reverse(([g_next[0] + h, g_next[1]], g_next[0], next)));
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`Bounds`].

    use alloc::vec::Vec;

    use super::Bounds;

    /// The edges of a small graph as `(from, to, cost)`.
    const EDGES: [(u8, u8, [u32; 2]); 5] = [
        (0, 1, [1, 9]),
        (0, 2, [5, 1]),
        (2, 1, [5, 1]),
        (1, 3, [1, 1]),
        (0, 4, [50, 50]),
    ];

    /// Appends the outgoing edges of `s`.
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "the neighbour callback signature"
    )]
    fn successors(s: &u8, out: &mut Vec<(u8, [u32; 2])>) {
        out.extend(EDGES.iter().filter(|e| e.0 == *s).map(|e| (e.1, e.2)));
    }

    /// Appends the incoming edges of `s`.
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "the neighbour callback signature"
    )]
    fn predecessors(s: &u8, out: &mut Vec<(u8, [u32; 2])>) {
        out.extend(EDGES.iter().filter(|e| e.1 == *s).map(|e| (e.0, e.2)));
    }

    /// Distances are exact on the states of Pareto-optimal paths, and states
    /// only on dominated paths have none.
    #[test]
    fn bounded_distances() {
        let bounds = Bounds::new(&0, &3, successors, predecessors).expect("reachable");

        // Pareto paths 0-1-3 = [2, 10] and 0-2-1-3 = [11, 3]: ub1 = 11, ub2 = 10.
        assert_eq!(bounds.forward(&0), Some([2, 3]));
        assert_eq!(bounds.forward(&2), Some([6, 2]));
        assert_eq!(bounds.backward(&3), Some([2, 3]));
        assert_eq!(bounds.backward(&1), Some([1, 2]));
        assert_eq!(bounds.backward(&4), None);
        assert_eq!(bounds.forward(&4), None);
    }

    /// An unreachable goal is reported as `None`.
    #[test]
    fn unreachable_goal() {
        assert!(Bounds::new(&3, &0, successors, predecessors).is_none());
    }
}
