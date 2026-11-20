//! Parallel MOA*: one lexicographic MOA* per cyclic objective order, with
//! shared upper bounds.
//!
//! Ahmadi, Sturtevant, Raith, Harabor, Jalili: "Parallelizing
//! Multi-objective A* Search", arXiv 2503.10075.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use search_core::{Cost, Heuristic};

use crate::cost::{add, weakly_dominates};
use crate::graph::Graph;
use crate::output::{Output, Solution, Stats};
use crate::pareto::{Front, TruncatedFront};

/// Parallel MOA*: finds one path for every Pareto-optimal cost vector from
/// `start` to a state that satisfies `is_goal`, with `N` threads.
///
/// Thread `i` runs a lexicographic MOA* (lazy dominance checks, one
/// truncated closed set per state, Algorithm 3) with objective `i` first
/// and the others in cyclic order (Algorithm 2). Its solutions feed a
/// multi-level upper bound (Algorithms 4 and 5): a solution's truncated
/// cost moves one level down, dropping one more objective `j`, once thread
/// `j` has found a solution at least as expensive in `j`, since every
/// cheaper one in `j` is then known. A cost truncated `N - 1` times is a
/// scalar bound on the last objective that every thread prunes against,
/// and the thread led by that objective stops once its primary cost reaches
/// it. The search ends when the first thread ends; the union of all
/// threads' solutions, filtered to its non-dominated vectors, is the
/// front, whatever the thread timing.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is consistent in every objective and zero on every goal
///   (each thread uses its own clone).
///
/// Solutions are sorted lexicographically, one per Pareto-optimal cost.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, ZeroHeuristic, namoa_star, parallel_moa_star};
///
/// /// A small diamond with three objectives.
/// struct Diamond;
///
/// impl Graph for Diamond {
///     type State = u8;
///     type Weight = [u32; 3];
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 3])>) {
///         match state {
///             0 => out.extend([(1, [1, 2, 3]), (2, [3, 2, 1]), (3, [5, 5, 5])]),
///             1 | 2 => out.push((3, [1, 1, 1])),
///             _ => {}
///         }
///     }
/// }
///
/// let out = parallel_moa_star(&Diamond, ZeroHeuristic, 0, |s| *s == 3);
/// let exact = namoa_star(&Diamond, ZeroHeuristic, 0, |s| *s == 3);
/// let costs = |o: &multiobjective_search::Output<u8, [u32; 3]>| {
///     o.solutions.iter().map(|s| s.cost).collect::<Vec<_>>()
/// };
/// assert_eq!(costs(&out), costs(&exact));
/// ```
///
/// # Panics
///
/// Panics if a thread panics.
#[expect(
    clippy::needless_pass_by_value,
    reason = "takes the heuristic by value like every other search; each thread clones it"
)]
pub fn parallel_moa_star<G, H, C, P, const N: usize>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]> + Sync,
    G::State: Send + Sync,
    H: Heuristic<G::State, [C; N]> + Clone + Send,
    C: Cost + Send + Sync,
    P: Fn(&G::State) -> bool + Sync,
{
    let shared = Shared::<G::State, C, N> {
        threads: (0..N)
            .map(|_| {
                Mutex::new(Found {
                    solutions: Vec::new(),
                    latest: None,
                })
            })
            .collect(),
        bounds: Mutex::new([None; N]),
        stop: AtomicBool::new(false),
        expanded: AtomicUsize::new(0),
        generated: AtomicUsize::new(0),
    };

    std::thread::scope(|scope| {
        for lead in 0..N {
            let (shared, heuristic, is_goal, start) =
                (&shared, heuristic.clone(), &is_goal, start.clone());
            scope.spawn(move || Thread::new(graph, heuristic, is_goal, lead, shared).run(start));
        }
    });

    merge(shared)
}

/// What one thread has found.
struct Found<S, C, const N: usize> {
    /// Its solutions, in the order found.
    solutions: Vec<Solution<S, [C; N]>>,
    /// The primary cost of its latest solution: every Pareto-optimal cost
    /// with a smaller value in this thread's objective is in the union.
    latest: Option<C>,
}

/// State shared by the threads.
struct Shared<S, C, const N: usize> {
    /// Per thread, its solutions.
    threads: Vec<Mutex<Found<S, C, N>>>,
    /// The global scalar upper bound per objective; `None` is infinite.
    bounds: Mutex<[Option<C>; N]>,
    /// Set by the first thread to finish.
    stop: AtomicBool,
    /// Expansions over all threads.
    expanded: AtomicUsize,
    /// Generations over all threads.
    generated: AtomicUsize,
}

/// One thread's search.
struct Thread<'a, G: Graph, H, C, P, const N: usize> {
    /// The graph.
    graph: &'a G,
    /// This thread's heuristic.
    heuristic: H,
    /// Goal test.
    is_goal: &'a P,
    /// The objective that leads this thread's order.
    lead: usize,
    /// Shared state.
    shared: &'a Shared<G::State, C, N>,
    /// Labels: state, `g` (original order), parent.
    nodes: Vec<(G::State, [C; N], Option<usize>)>,
    /// Open list on rotated `f`.
    open: BinaryHeap<Reverse<([C; N], usize)>>,
    /// Rotated `g` of the labels expanded per state, truncated once.
    closed: BTreeMap<G::State, TruncatedFront<C, N>>,
    /// `levels[λ]`: rotated solution costs with their first `λ`
    /// components zeroed (truncated `λ` times), mutually non-dominated.
    levels: Vec<Vec<[C; N]>>,
    /// Reused successor buffer.
    successors: Vec<(G::State, [C; N])>,
    /// The global bounds as last read, refreshed once per pop so that the
    /// lock is not taken per generated child.
    bounds: [Option<C>; N],
}

impl<'a, G, H, C, P, const N: usize> Thread<'a, G, H, C, P, N>
where
    G: Graph<Weight = [C; N]>,
    H: Heuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    /// A thread led by objective `lead`.
    fn new(
        graph: &'a G,
        heuristic: H,
        is_goal: &'a P,
        lead: usize,
        shared: &'a Shared<G::State, C, N>,
    ) -> Self {
        Self {
            graph,
            heuristic,
            is_goal,
            lead,
            shared,
            nodes: Vec::new(),
            open: BinaryHeap::new(),
            closed: BTreeMap::new(),
            levels: (0..N).map(|_| Vec::new()).collect(),
            successors: Vec::new(),
            bounds: [None; N],
        }
    }

    /// `v` in this thread's order: objective `lead` first.
    fn rotate(&self, v: [C; N]) -> [C; N] {
        core::array::from_fn(|p| v[(self.lead + p) % N])
    }

    /// Runs until done or stopped (Algorithm 3), then stops the others.
    fn run(mut self, start: G::State) {
        if let Some(h) = self.heuristic.estimate(&start) {
            self.nodes.push((start, [C::ZERO; N], None));
            self.open.push(Reverse((self.rotate(h), 0)));
            self.shared.generated.fetch_add(1, Ordering::Relaxed);
        }

        while let Some(Reverse((f, index))) = self.open.pop() {
            if self.shared.stop.load(Ordering::Relaxed) {
                return;
            }

            // Early termination: the primary cost reached its global bound.
            self.bounds = *self
                .shared
                .bounds
                .lock()
                .expect("no thread panics holding the bounds");
            if self.bounds[self.lead].is_some_and(|b| f[0] >= b) {
                break;
            }

            let (state, g) = (
                self.nodes[index].0.clone(),
                self.rotate(self.nodes[index].1),
            );
            if self
                .closed
                .get(&state)
                .is_some_and(|c| c.weakly_dominates(&g))
                || self.dominated(&f)
            {
                continue;
            }

            self.closed
                .entry(state.clone())
                .or_insert_with(TruncatedFront::new)
                .insert(g);
            if (self.is_goal)(&state) {
                self.record(index, g);
                continue;
            }

            self.expand(index);
        }

        self.shared.stop.store(true, Ordering::Relaxed);
    }

    /// Generates the successors of the label at `index` (lines 16 to 23).
    fn expand(&mut self, index: usize) {
        self.shared.expanded.fetch_add(1, Ordering::Relaxed);

        let mut successors = core::mem::take(&mut self.successors);
        let (state, g) = (self.nodes[index].0.clone(), self.nodes[index].1);
        self.graph.successors(&state, &mut successors);

        for (next, weight) in successors.drain(..) {
            let Some(h) = self.heuristic.estimate(&next) else {
                continue;
            };

            let g_next = add(g, weight);
            let (g_rot, f_rot) = (self.rotate(g_next), self.rotate(add(g_next, h)));
            if self
                .closed
                .get(&next)
                .is_some_and(|c| c.weakly_dominates(&g_rot))
                || self.dominated(&f_rot)
            {
                continue;
            }

            self.nodes.push((next, g_next, Some(index)));
            self.open.push(Reverse((f_rot, self.nodes.len() - 1)));
            self.shared.generated.fetch_add(1, Ordering::Relaxed);
        }

        self.successors = successors;
    }

    /// `IsDominatedMD` (Algorithm 4) on a rotated `f`: the global bounds,
    /// then every truncation level.
    fn dominated(&self, f: &[C; N]) -> bool {
        let over = (0..N).any(|p| self.bounds[(self.lead + p) % N].is_some_and(|b| f[p] >= b));

        over || (1..N).any(|level| {
            let probe = zero_first(*f, level);
            self.levels[level]
                .iter()
                .any(|v| weakly_dominates(v, &probe))
        })
    }

    /// Records a solution, publishes it, and updates the bounds (lines 13
    /// and 14, Algorithm 5).
    fn record(&mut self, index: usize, g: [C; N]) {
        insert(&mut self.levels[1], zero_first(g, 1));

        let solution = rebuild(&self.nodes, index);
        {
            let mut found = self.shared.threads[self.lead]
                .lock()
                .expect("no thread panics holding its solutions");
            found.solutions.push(solution);
            found.latest = Some(g[0]);
        }

        self.update_bounds();
    }

    /// `UpdateUpperBound` (Algorithm 5): moves solution costs down the
    /// levels while the next objective's thread has passed them, and turns
    /// a fully truncated cost into a global bound.
    fn update_bounds(&mut self) {
        for level in 1..N.saturating_sub(1) {
            let objective = (self.lead + level) % N;
            let latest = self.shared.threads[objective]
                .lock()
                .expect("no thread panics holding its solutions")
                .latest;
            let Some(z) = latest else { continue };

            let (moved, kept): (Vec<_>, Vec<_>) =
                self.levels[level].drain(..).partition(|v| v[level] <= z);
            self.levels[level] = kept;
            for v in moved {
                insert(&mut self.levels[level + 1], zero_first(v, level + 1));
            }
        }

        if N < 2 {
            return;
        }

        if let Some(scalar) = self.levels[N - 1].iter().map(|v| v[N - 1]).min() {
            let objective = (self.lead + N - 1) % N;
            let mut bounds = self
                .shared
                .bounds
                .lock()
                .expect("no thread panics holding the bounds");
            if bounds[objective].is_none_or(|b| scalar < b) {
                bounds[objective] = Some(scalar);
            }

            self.bounds = *bounds;
        }
    }
}

/// `v` with its first `count` components set to zero.
fn zero_first<C: Cost, const N: usize>(mut v: [C; N], count: usize) -> [C; N] {
    for c in v.iter_mut().take(count) {
        *c = C::ZERO;
    }

    v
}

/// Adds `v` to a mutually non-dominated list unless weakly dominated,
/// dropping what it dominates.
fn insert<C: Cost, const N: usize>(list: &mut Vec<[C; N]>, v: [C; N]) {
    if list.iter().any(|w| weakly_dominates(w, &v)) {
        return;
    }

    list.retain(|w| !weakly_dominates(&v, w));
    list.push(v);
}

/// Rebuilds the path ending in label `index`.
fn rebuild<S: Clone, C: Copy, const N: usize>(
    nodes: &[(S, [C; N], Option<usize>)],
    index: usize,
) -> Solution<S, [C; N]> {
    let mut path = Vec::new();
    let mut current = Some(index);

    while let Some(i) = current {
        path.push(nodes[i].0.clone());
        current = nodes[i].2;
    }

    path.reverse();

    Solution {
        cost: nodes[index].1,
        path,
    }
}

/// The union of the threads' solutions, non-dominated and sorted.
fn merge<S, C: Cost, const N: usize>(shared: Shared<S, C, N>) -> Output<S, [C; N]> {
    let mut all: Vec<Solution<S, [C; N]>> = shared
        .threads
        .into_iter()
        .flat_map(|t| t.into_inner().expect("no thread panicked").solutions)
        .collect();
    all.sort_by_key(|s| s.cost);

    // In lexicographic order, a later vector cannot dominate an earlier one.
    let mut front = Front::new();
    let solutions = all.into_iter().filter(|s| front.insert(s.cost)).collect();

    Output {
        solutions,
        stats: Stats {
            expanded: shared.expanded.into_inner(),
            generated: shared.generated.into_inner(),
        },
    }
}
