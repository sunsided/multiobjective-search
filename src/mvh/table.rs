//! Building multi-valued heuristics: backward A*pex and the Bellman-Ford
//! consistency fix of Wolff, Felner, Salzman (arXiv 2606.20644, Section 6).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use search_core::{Cost, WideMul};

use crate::apex::expanded_apexes;
use crate::cost::add;
use crate::epsilon::Epsilon;
use crate::graph::{Graph, ReverseGraph};
use crate::mvh::MvHeuristic;
use crate::pareto::Front;

/// A multi-valued heuristic stored as a table: per state, a set of
/// mutually non-dominated estimates. A state without an entry is a dead
/// end.
///
/// Build one with [`apex_mvh`], and make it consistent with
/// [`MvhTable::make_consistent`] for
/// [`namoa_star_dr_mvh`](crate::namoa_star_dr_mvh).
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Epsilon, Graph, MvHeuristic, ReverseGraph, apex_mvh};
///
/// /// Two routes from 0 to the goal 1.
/// struct Routes;
///
/// impl Graph for Routes {
///     type State = u8;
///     type Weight = [u32; 2];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 0 {
///             out.extend([(1, [1, 5]), (1, [4, 2])]);
///         }
///     }
/// }
///
/// impl ReverseGraph for Routes {
///     fn predecessors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
///         if *state == 1 {
///             out.extend([(0, [1, 5]), (0, [4, 2])]);
///         }
///     }
/// }
///
/// let mut h = apex_mvh(&Routes, [1], [Epsilon::ZERO; 2]);
/// let mut out = Vec::new();
/// h.estimates(&0, &mut out);
/// assert_eq!(out, [[1, 5], [4, 2]]); // exact with zero tolerance
/// ```
#[derive(Debug, Clone)]
pub struct MvhTable<S, C, const N: usize> {
    /// The estimates per state; never an empty set.
    sets: BTreeMap<S, Front<C, N>>,
}

impl<S: Ord, C: Cost, const N: usize> MvhTable<S, C, N> {
    /// The estimates of `state`, sorted lexicographically; empty for a
    /// dead end.
    #[must_use]
    pub fn get(&self, state: &S) -> &[[C; N]] {
        self.sets.get(state).map_or(&[], Front::as_slice)
    }

    /// The total number of estimates over all states.
    #[must_use]
    pub fn size(&self) -> usize {
        self.sets.values().map(Front::len).sum()
    }

    /// Makes the table consistent in the monotone sense (see
    /// [`MvHeuristic`]) by Bellman-Ford label updates: while some edge
    /// `(s, t)` with cost `c` has an estimate `h'` of `t` such that no
    /// estimate of `s` weakly dominates `c + h'`, add `c + h'` to `s` and
    /// drop the estimates it dominates.
    ///
    /// Only lower bounds are added, so an admissible table stays
    /// admissible. Wolff et al. add `c + h'` when it strictly dominates an
    /// estimate of `s`; adding it whenever it is not covered is what
    /// guarantees the monotone form. The fix-point exists on finite graphs:
    /// every added vector is an original estimate plus a path cost, and one
    /// that runs around a cycle is weakly dominated by the vector without
    /// the cycle. States whose estimates change are revisited through
    /// their predecessors.
    pub fn make_consistent<G>(&mut self, graph: &G)
    where
        S: Clone,
        G: ReverseGraph<State = S, Weight = [C; N]>,
    {
        let mut work: Vec<S> = self.sets.keys().cloned().collect();
        let mut queued: BTreeSet<S> = work.iter().cloned().collect();
        let mut edges = Vec::new();

        while let Some(target) = work.pop() {
            queued.remove(&target);
            let estimates = self.get(&target).to_vec();
            graph.predecessors(&target, &mut edges);

            for (source, weight) in edges.drain(..) {
                if self.lower(&source, weight, &estimates) && queued.insert(source.clone()) {
                    work.push(source);
                }
            }
        }
    }

    /// Adds `weight + h'` to `source` for every `h'` in `estimates` that
    /// no estimate of `source` covers; returns `true` if any was added.
    fn lower(&mut self, source: &S, weight: [C; N], estimates: &[[C; N]]) -> bool
    where
        S: Clone,
    {
        let set = self.sets.entry(source.clone()).or_default();
        let mut changed = false;
        for &h in estimates {
            let bound = add(weight, h);
            if !set.weakly_dominates(&bound) {
                set.insert(bound);
                changed = true;
            }
        }

        changed
    }
}

impl<S: Ord, C: Cost, const N: usize> MvHeuristic<S, [C; N]> for MvhTable<S, C, N> {
    fn estimates(&mut self, state: &S, out: &mut Vec<[C; N]>) {
        out.extend_from_slice(self.get(state));
    }
}

impl<S: Ord, C: Cost, const N: usize> MvHeuristic<S, [C; N]> for &MvhTable<S, C, N> {
    fn estimates(&mut self, state: &S, out: &mut Vec<[C; N]>) {
        out.extend_from_slice(self.get(state));
    }
}

/// Builds an admissible multi-valued heuristic by A*pex backwards from
/// `goals` with tolerances `eps` (Wolff et al., Section 6, "A*pex-MVH").
///
/// The estimates of a state are the apexes of the pairs A*pex expands
/// there on the reversed graph. Each apex weakly dominates the cost of
/// every path it stands for, so every path from the state to a goal is
/// bounded by one of them: the table is admissible. It is generally not
/// consistent; see [`MvhTable::make_consistent`]. Larger `eps` merges more
/// paths, which gives fewer and looser estimates; `eps = 0` gives the exact
/// Pareto sets of the remaining cost. States that reach no goal get no
/// entry.
///
/// Requires non-negative edge costs.
pub fn apex_mvh<G, C, const N: usize>(
    graph: &G,
    goals: impl IntoIterator<Item = G::State>,
    eps: [Epsilon; N],
) -> MvhTable<G::State, C, N>
where
    G: ReverseGraph<Weight = [C; N]>,
    C: Cost + WideMul,
{
    MvhTable {
        sets: expanded_apexes(&Reversed(graph), goals, eps),
    }
}

/// `G` with its edges reversed.
struct Reversed<'a, G>(&'a G);

impl<G: ReverseGraph> Graph for Reversed<'_, G> {
    type State = G::State;
    type Weight = G::Weight;

    fn successors(&self, state: &G::State, out: &mut Vec<(G::State, G::Weight)>) {
        self.0.predecessors(state, out);
    }
}
