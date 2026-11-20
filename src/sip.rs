//! SIP-MOSP: asynchronous parallel exact multi-objective search with
//! versioned frontier snapshots and indexed dominance pruning.
//!
//! Xu, Zhang, Qian, Liu, Wu, Tang: "Asynchronous Parallel Search for Exact
//! Multi-Objective Shortest Paths with Versioned Frontier Snapshots and
//! Indexed Dominance Pruning", arXiv 2609.11944, 2026.
//!
//! *Search workers* expand labels from private best-first queues, each
//! under its own cyclic objective order, and prune the children against
//! published, immutable snapshots of the endpoint's and the goals'
//! frontiers. A surviving child goes to the *update worker* that owns its
//! endpoint (one owner per state, by hash), or to the single *goal worker*.
//! An owner alone modifies the exact ("canonical") frontier of its states:
//! it re-checks the candidate unless the snapshot it was checked against
//! is still current, inserts it, publishes the next snapshot version and
//! hands the label to the less loaded of two search workers. Snapshots are
//! a lexicographically sorted, indexed base ([`IndexedFront`]) plus a short
//! delta of recent additions, so most publications share the base.
//!
//! ```mermaid
//! graph LR
//!     sw["search workers<br/>(private queues)"] -- "child, snapshot version" --> uw["update worker<br/>owning the endpoint"]
//!     sw -- "goal child" --> gw[goal worker]
//!     uw -- "admitted label" --> sw
//!     uw -. publish .-> snap[("snapshots<br/>base + delta + index")]
//!     gw -. publish .-> snap
//!     snap -. read .-> sw
//! ```
//!
//! The result is the exact Pareto-optimal front whatever the thread timing
//! (the paper's Theorem 1); which path represents a cost vector may
//! differ between runs.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};
use core::hash::{Hash, Hasher};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering as Atomic};
use std::collections::HashMap;
use std::hash::DefaultHasher;
use std::sync::{Condvar, Mutex, RwLock};

use search_core::{Cost, Heuristic};

use crate::cost::add;
use crate::graph::Graph;
use crate::output::{Output, Solution, Stats};
use crate::pareto::{Front, FrontIndex, IndexedFront};

/// Worker counts and the snapshot index of [`sip_mosp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SipConfig {
    /// Search workers, at least 1.
    pub search_workers: usize,
    /// Update workers for non-goal states, at least 1. A goal worker runs
    /// in addition.
    pub update_workers: usize,
    /// The index over snapshot bases: [`FrontIndex::Blocks`] is SIP-MOSP-BM,
    /// [`FrontIndex::Tree`] SIP-MOSP-ST.
    pub index: FrontIndex,
}

impl Default for SipConfig {
    /// Two search and two update workers with the block-minimum index.
    fn default() -> Self {
        Self {
            search_workers: 2,
            update_workers: 2,
            index: FrontIndex::Blocks,
        }
    }
}

/// Frontiers smaller than this are published as an unindexed base only
/// (the paper's `θ_I`).
const INDEX_THRESHOLD: usize = 64;

/// The delta capacity before the base is rebuilt (the paper's adaptive
/// `θ_Δ(|F|)`): an eighth of the frontier, at least 8.
fn delta_capacity(frontier: usize) -> usize {
    (frontier / 8).max(8)
}

/// SIP-MOSP: finds one path for every Pareto-optimal cost vector from
/// `start` to a state that satisfies `is_goal`, with one thread per search
/// and update worker plus a goal worker.
///
/// Requirements, not checked at run time:
///
/// - edge costs are non-negative;
/// - `heuristic` is admissible in every objective and zero on every goal
///   (the paper uses exact per-objective distances, which are
///   consistent). Each search worker uses its own clone.
///
/// Solutions are sorted lexicographically, one per Pareto-optimal cost
/// vector; the same vectors as [`namoa_star`](crate::namoa_star) in every
/// run. Goal states are not expanded.
///
/// # Panics
///
/// Panics if a worker count is zero, or if a worker thread panics.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, SipConfig, ZeroHeuristic, namoa_star, sip_mosp};
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
/// let out = sip_mosp(&Diamond, ZeroHeuristic, 0, |s| *s == 3, SipConfig::default());
/// let exact = namoa_star(&Diamond, ZeroHeuristic, 0, |s| *s == 3);
/// let costs = |o: &multiobjective_search::Output<u8, [u32; 3]>| {
///     o.solutions.iter().map(|s| s.cost).collect::<Vec<_>>()
/// };
/// assert_eq!(costs(&out), costs(&exact));
/// ```
pub fn sip_mosp<G, H, C, P, const N: usize>(
    graph: &G,
    mut heuristic: H,
    start: G::State,
    is_goal: P,
    config: SipConfig,
) -> Output<G::State, [C; N]>
where
    G: Graph<Weight = [C; N]> + Sync,
    G::State: Hash + Send + Sync,
    H: Heuristic<G::State, [C; N]> + Clone + Send,
    C: Cost + Send + Sync,
    P: Fn(&G::State) -> bool + Sync,
{
    assert!(
        config.search_workers > 0 && config.update_workers > 0,
        "SIP-MOSP needs at least one search and one update worker"
    );

    let Some(h) = heuristic.estimate(&start) else {
        return Output {
            solutions: Vec::new(),
            stats: Stats::default(),
        };
    };

    let shared = Shared::new(config);
    let start_label = Arc::new(Label {
        state: start,
        g: [C::ZERO; N],
        f: h,
        parent: None,
        seen: 0,
    });
    shared.generated.fetch_add(1, Atomic::Relaxed);

    let goal_labels = std::thread::scope(|scope| {
        for r in 0..config.search_workers {
            let (shared, heuristic, is_goal) = (&shared, heuristic.clone(), &is_goal);
            scope.spawn(move || search_worker(shared, graph, heuristic, is_goal, r));
        }

        for p in 0..config.update_workers {
            let shared = &shared;
            scope.spawn(move || update_worker(shared, p));
        }

        let goal = scope.spawn(|| goal_worker(&shared));
        shared.route(start_label, &is_goal);

        goal.join().expect("the goal worker panicked")
    });

    let stats = Stats {
        expanded: shared.expanded.load(Atomic::Relaxed),
        generated: shared.generated.load(Atomic::Relaxed),
    };

    Output {
        solutions: goal_labels
            .into_iter()
            .map(|label| rebuild(&label))
            .collect(),
        stats,
    }
}

/// A label: one concrete path, shared by its children through `parent`.
struct Label<S, C, const N: usize> {
    /// The state the path ends in.
    state: S,
    /// Cost of the path.
    g: [C; N],
    /// `g + h`.
    f: [C; N],
    /// The path without its last edge; `None` for the start.
    parent: Option<Arc<Label<S, C, N>>>,
    /// Version of the endpoint's (or, for a goal, the goal) snapshot the
    /// label was checked against when generated.
    seen: u64,
}

/// A label in a private queue, ordered by `key` and then arrival.
struct Entry<S, C, const N: usize> {
    /// `f` under the worker's objective order.
    key: [C; N],
    /// Arrival number, for a deterministic order among equal keys.
    seq: usize,
    /// The label.
    label: Arc<Label<S, C, N>>,
}

impl<S, C: Ord, const N: usize> PartialEq for Entry<S, C, N> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<S, C: Ord, const N: usize> Eq for Entry<S, C, N> {}

impl<S, C: Ord, const N: usize> PartialOrd for Entry<S, C, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S, C: Ord, const N: usize> Ord for Entry<S, C, N> {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.key, self.seq).cmp(&(&other.key, other.seq))
    }
}

/// An immutable published frontier: the indexed base plus the recent
/// additions (`S_v^k = (B, Δ, I, k)`).
struct Snapshot<C, const N: usize> {
    /// The consolidated, indexed base, shared between versions.
    base: Arc<IndexedFront<C, N>>,
    /// Vectors added since the base was built.
    delta: Vec<[C; N]>,
    /// Component-wise minimum of `delta`; `None` when it is empty.
    delta_min: Option<[C; N]>,
    /// The version `k`, counted per frontier from 0.
    version: u64,
}

impl<C: Cost, const N: usize> Snapshot<C, N> {
    /// The empty snapshot, version 0.
    fn empty() -> Self {
        Self {
            base: Arc::new(IndexedFront::new(Vec::new(), FrontIndex::None)),
            delta: Vec::new(),
            delta_min: None,
            version: 0,
        }
    }

    /// `SnapshotDominates` (Algorithm 4): the delta first, guarded by its
    /// minimum, then the base through its index.
    fn dominates(&self, q: &[C; N]) -> bool {
        let below = |p: &[C; N]| p.iter().zip(q).all(|(a, b)| a <= b);
        if self.delta_min.as_ref().is_some_and(below) && self.delta.iter().any(below) {
            return true;
        }

        self.base.weakly_dominates(q)
    }

    /// The next version after admitting `g` into `front` (lines 13 to 24
    /// of Algorithm 3): a fresh unindexed base for a small frontier, a
    /// rebuilt indexed base when the delta is full or the frontier just
    /// outgrew the threshold, and otherwise the shared base with `g`
    /// appended to the delta.
    fn next(&self, front: &Front<C, N>, g: [C; N], index: FrontIndex) -> Self {
        let version = self.version + 1;
        let rebuild = |index| Self {
            base: Arc::new(IndexedFront::new(front.as_slice().to_vec(), index)),
            delta: Vec::new(),
            delta_min: None,
            version,
        };

        if front.len() < INDEX_THRESHOLD {
            return rebuild(FrontIndex::None);
        }

        let unindexed = self.base.len() < INDEX_THRESHOLD;
        if unindexed || self.delta.len() + 1 >= delta_capacity(front.len()) {
            return rebuild(index);
        }

        let mut delta = self.delta.clone();
        delta.push(g);
        let delta_min = Some(match self.delta_min {
            Some(m) => core::array::from_fn(|i| m[i].min(g[i])),
            None => g,
        });

        Self {
            base: Arc::clone(&self.base),
            delta,
            delta_min,
            version,
        }
    }
}

/// A worker's incoming queue: many producers, one consumer.
struct Inbox<T> {
    /// Arrivals not yet moved into the worker's private queue.
    items: Mutex<Vec<T>>,
    /// Signalled on every arrival and on termination.
    ready: Condvar,
    /// Labels sent to this worker and not yet processed; the search
    /// workers' load for the power-of-two-choices placement.
    load: AtomicUsize,
}

impl<T> Inbox<T> {
    /// An empty inbox.
    fn new() -> Self {
        Self {
            items: Mutex::new(Vec::new()),
            ready: Condvar::new(),
            load: AtomicUsize::new(0),
        }
    }
}

/// A published-snapshot table for the states one update worker owns.
type Shard<S, C, const N: usize> = RwLock<HashMap<S, Arc<Snapshot<C, N>>>>;

/// Everything the workers share.
struct Shared<S, C, const N: usize> {
    /// The configuration.
    config: SipConfig,
    /// Incoming queues of the search workers.
    search: Vec<Inbox<Arc<Label<S, C, N>>>>,
    /// Incoming queues of the update workers.
    update: Vec<Inbox<Arc<Label<S, C, N>>>>,
    /// Incoming queue of the goal worker.
    goal: Inbox<Arc<Label<S, C, N>>>,
    /// Published snapshots per update worker.
    shards: Vec<Shard<S, C, N>>,
    /// The published goal snapshot.
    goal_snapshot: RwLock<Arc<Snapshot<C, N>>>,
    /// Labels sent and not yet fully processed; the search has ended when
    /// it drops to zero.
    pending: AtomicUsize,
    /// Set once `pending` reached zero.
    done: AtomicBool,
    /// Labels expanded by search workers.
    expanded: AtomicUsize,
    /// Labels generated, the start included.
    generated: AtomicUsize,
}

impl<S: Clone + Hash + Eq, C: Cost, const N: usize> Shared<S, C, N> {
    /// Queues and snapshot tables for `config`.
    fn new(config: SipConfig) -> Self {
        Self {
            config,
            search: (0..config.search_workers).map(|_| Inbox::new()).collect(),
            update: (0..config.update_workers).map(|_| Inbox::new()).collect(),
            goal: Inbox::new(),
            shards: (0..config.update_workers)
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
            goal_snapshot: RwLock::new(Arc::new(Snapshot::empty())),
            pending: AtomicUsize::new(0),
            done: AtomicBool::new(false),
            expanded: AtomicUsize::new(0),
            generated: AtomicUsize::new(0),
        }
    }

    /// The update worker that owns `state`.
    fn owner(&self, state: &S) -> usize {
        bucket(state, 0, self.config.update_workers)
    }

    /// The published snapshot of a non-goal `state`.
    fn snapshot(&self, state: &S) -> Arc<Snapshot<C, N>> {
        let shard = self.shards[self.owner(state)]
            .read()
            .expect("no worker panics while holding a shard");

        shard
            .get(state)
            .map_or_else(|| Arc::new(Snapshot::empty()), Arc::clone)
    }

    /// The published goal snapshot.
    fn goal_snapshot(&self) -> Arc<Snapshot<C, N>> {
        Arc::clone(
            &self
                .goal_snapshot
                .read()
                .expect("no worker panics while publishing"),
        )
    }

    /// Sends a candidate to the goal worker or to its endpoint's owner.
    fn route(&self, label: Arc<Label<S, C, N>>, is_goal: impl Fn(&S) -> bool) {
        if is_goal(&label.state) {
            self.send(&self.goal, label);
        } else {
            self.send(&self.update[self.owner(&label.state)], label);
        }
    }

    /// Hands an admitted label to the less loaded of two search workers
    /// chosen by hashes of its state (power of two choices).
    fn dispatch(&self, label: Arc<Label<S, C, N>>) {
        let n = self.config.search_workers;
        let (a, b) = (bucket(&label.state, 1, n), bucket(&label.state, 2, n));
        let load = |r: usize| self.search[r].load.load(Atomic::Relaxed);
        let r = if load(a) <= load(b) { a } else { b };
        self.send(&self.search[r], label);
    }

    /// Appends `item` to `inbox`, counting it as pending.
    fn send<T>(&self, inbox: &Inbox<T>, item: T) {
        self.pending.fetch_add(1, Atomic::SeqCst);
        inbox.load.fetch_add(1, Atomic::Relaxed);
        inbox
            .items
            .lock()
            .expect("no worker panics while holding an inbox")
            .push(item);
        inbox.ready.notify_one();
    }

    /// Marks one item of `inbox` as processed, and ends the search when
    /// nothing is pending any more.
    fn finish<T>(&self, inbox: &Inbox<T>) {
        inbox.load.fetch_sub(1, Atomic::Relaxed);
        if self.pending.fetch_sub(1, Atomic::SeqCst) == 1 {
            self.done.store(true, Atomic::SeqCst);
            Self::wake(&self.goal);
            for inbox in self.search.iter().chain(&self.update) {
                Self::wake(inbox);
            }
        }
    }

    /// Wakes a worker waiting on `inbox`, holding its lock so that the
    /// wake-up cannot be missed.
    fn wake<T>(inbox: &Inbox<T>) {
        let _guard = inbox
            .items
            .lock()
            .expect("no worker panics while holding an inbox");
        inbox.ready.notify_all();
    }

    /// Moves the arrivals of `inbox` into `queue`, waiting while both are
    /// empty. Returns `false` once the search has ended.
    fn refill<T>(&self, inbox: &Inbox<T>, mut queue: impl FnMut(T), queued: bool) -> bool {
        let mut items = inbox
            .items
            .lock()
            .expect("no worker panics while holding an inbox");
        while items.is_empty() && !queued {
            if self.done.load(Atomic::SeqCst) {
                return false;
            }

            items = inbox
                .ready
                .wait(items)
                .expect("no worker panics while holding an inbox");
        }

        for item in items.drain(..) {
            queue(item);
        }

        true
    }
}

/// A deterministic bucket in `0..n` for `state`, salted by `seed`.
fn bucket<S: Hash>(state: &S, seed: u64, n: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    state.hash(&mut hasher);
    let n = u64::try_from(n).expect("worker counts fit in u64");
    usize::try_from(hasher.finish() % n).expect("a bucket is below the worker count")
}

/// A private best-first queue under one cyclic objective order.
struct Private<S, C, const N: usize> {
    /// The queue.
    heap: BinaryHeap<Reverse<Entry<S, C, N>>>,
    /// The first objective of the order.
    first: usize,
    /// Arrivals so far.
    seq: usize,
}

impl<S, C: Cost, const N: usize> Private<S, C, N> {
    /// An empty queue ordering `f` from objective `first` on, cyclically.
    fn new(first: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            first,
            seq: 0,
        }
    }

    /// Queues a label.
    fn push(&mut self, label: Arc<Label<S, C, N>>) {
        let key = core::array::from_fn(|i| label.f[(i + self.first) % N]);
        self.seq += 1;
        self.heap.push(Reverse(Entry {
            key,
            seq: self.seq,
            label,
        }));
    }

    /// Takes the label with the smallest key.
    fn pop(&mut self) -> Option<Arc<Label<S, C, N>>> {
        self.heap.pop().map(|Reverse(entry)| entry.label)
    }
}

/// A search worker (Algorithm 2): expands admitted labels and sends the
/// children that survive snapshot pruning to their owners.
fn search_worker<G, H, C, P, const N: usize>(
    shared: &Shared<G::State, C, N>,
    graph: &G,
    mut heuristic: H,
    is_goal: &P,
    worker: usize,
) where
    G: Graph<Weight = [C; N]>,
    G::State: Hash,
    H: Heuristic<G::State, [C; N]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
{
    let inbox = &shared.search[worker];
    let mut queue = Private::new(worker % N);
    let mut successors = Vec::new();

    loop {
        let queued = !queue.heap.is_empty();
        if !shared.refill(inbox, |label| queue.push(label), queued) {
            break;
        }

        let Some(parent) = queue.pop() else { continue };
        shared.expanded.fetch_add(1, Atomic::Relaxed);

        let goals = shared.goal_snapshot();
        graph.successors(&parent.state, &mut successors);
        for (state, weight) in successors.drain(..) {
            let Some(estimate) = heuristic.estimate(&state) else {
                continue;
            };

            let (g, goal) = (add(parent.g, weight), is_goal(&state));
            let f = add(g, estimate);
            let endpoint = if goal {
                Arc::clone(&goals)
            } else {
                shared.snapshot(&state)
            };

            if endpoint.dominates(&g) || goals.dominates(&f) {
                continue;
            }

            let child = Arc::new(Label {
                state,
                g,
                f,
                parent: Some(Arc::clone(&parent)),
                seen: endpoint.version,
            });
            shared.generated.fetch_add(1, Atomic::Relaxed);
            shared.route(child, is_goal);
        }

        shared.finish(inbox);
    }
}

/// The canonical frontier of one state and its last published snapshot.
struct Canonical<C, const N: usize> {
    /// The exact non-dominated costs admitted so far.
    front: Front<C, N>,
    /// The last snapshot published for it.
    snapshot: Arc<Snapshot<C, N>>,
}

impl<C: Cost, const N: usize> Canonical<C, N> {
    /// An empty frontier with the version-0 snapshot.
    fn new() -> Self {
        Self {
            front: Front::new(),
            snapshot: Arc::new(Snapshot::empty()),
        }
    }

    /// The admission step of Algorithm 3 (lines 8 to 24): rejects `g` if it
    /// was checked against an older snapshot and the current one dominates
    /// it, else inserts it and returns the next snapshot to publish.
    fn admit(&mut self, g: [C; N], seen: u64, index: FrontIndex) -> Option<Arc<Snapshot<C, N>>> {
        if seen != self.snapshot.version && self.snapshot.dominates(&g) {
            return None;
        }

        if !self.front.insert(g) {
            return None;
        }

        self.snapshot = Arc::new(self.snapshot.next(&self.front, g, index));
        Some(Arc::clone(&self.snapshot))
    }
}

/// An update worker (Algorithm 3) for the non-goal states it owns.
fn update_worker<S, C, const N: usize>(shared: &Shared<S, C, N>, p: usize)
where
    S: Clone + Hash + Eq,
    C: Cost,
{
    let inbox = &shared.update[p];
    let mut queue = Private::new(0);
    let mut frontiers: HashMap<S, Canonical<C, N>> = HashMap::new();

    loop {
        let queued = !queue.heap.is_empty();
        if !shared.refill(inbox, |label| queue.push(label), queued) {
            break;
        }

        let Some(y) = queue.pop() else { continue };
        let canonical = frontiers
            .entry(y.state.clone())
            .or_insert_with(Canonical::new);

        if let Some(snapshot) = canonical.admit(y.g, y.seen, shared.config.index) {
            shared.shards[p]
                .write()
                .expect("no worker panics while holding a shard")
                .insert(y.state.clone(), snapshot);
            shared.dispatch(y);
        }

        shared.finish(inbox);
    }
}

/// The goal worker: keeps the goal frontier and one representative per
/// admitted cost, and returns the representatives of the final front in
/// lexicographic order.
fn goal_worker<S, C, const N: usize>(shared: &Shared<S, C, N>) -> Vec<Arc<Label<S, C, N>>>
where
    S: Clone + Hash + Eq,
    C: Cost,
{
    let inbox = &shared.goal;
    let mut queue = Private::new(0);
    let mut frontier = Canonical::new();
    let mut representatives = BTreeMap::new();

    loop {
        let queued = !queue.heap.is_empty();
        if !shared.refill(inbox, |label| queue.push(label), queued) {
            break;
        }

        let Some(y) = queue.pop() else { continue };

        if let Some(snapshot) = frontier.admit(y.g, y.seen, shared.config.index) {
            *shared
                .goal_snapshot
                .write()
                .expect("no worker panics while publishing") = snapshot;
            representatives.insert(y.g, y);
        }

        shared.finish(inbox);
    }

    frontier
        .front
        .as_slice()
        .iter()
        .map(|cost| Arc::clone(&representatives[cost]))
        .collect()
}

/// The solution ending in `label`.
fn rebuild<S: Clone, C: Copy, const N: usize>(label: &Arc<Label<S, C, N>>) -> Solution<S, [C; N]> {
    let mut path = Vec::new();
    let mut current = Some(label);

    while let Some(l) = current {
        path.push(l.state.clone());
        current = l.parent.as_ref();
    }

    path.reverse();

    Solution {
        cost: label.g,
        path,
    }
}
