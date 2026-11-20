//! Multi-threaded BOBA*: the two directions on two threads, as in the paper.
//!
//! Each direction runs its own BOA* search and, after every label,
//! publishes the second cost of its last solution (the other direction's
//! primary bound) and its first expansions (for heuristic tuning). The
//! other direction picks both up before its next label. Reading a bound
//! late only prunes less, and a direction that stops lets the other stop
//! too: at that point every Pareto-optimal solution is in one of the two
//! outputs (see [`boba_star`](crate::boba_star)). The join drops the
//! dominated or duplicate boundary vectors that a late bound can leave.
//!
//! The front is the same as the sequential version's; for equal cost
//! vectors the path may differ, and the counters vary from run to run.

use alloc::vec::Vec;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Mutex, PoisonError};

use search_core::{Cost, Heuristic, heuristic_fn};

use super::{Reversed, bounds, equals, merge, swap};
use crate::boa::{Completions, NoCompletions, Search, Tolerance};
use crate::graph::{Graph, ReverseGraph};
use crate::output::{Output, Stats};

/// [`boba_star`](crate::boba_star) with the preliminary searches two at a
/// time and the two directions on two threads.
///
/// Same requirements and the same front as [`boba_star`](crate::boba_star).
///
/// # Panics
///
/// Panics if a search thread panics, which only a panicking graph or cost
/// implementation can cause.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{RoadGraph, boba_star, boba_star_parallel};
///
/// let distance = "p sp 3 3\na 1 2 10\na 2 3 10\na 1 3 25\n";
/// let time = "p sp 3 3\na 1 2 5\na 2 3 5\na 1 3 6\n";
/// let graph = RoadGraph::from_dimacs(distance, time)?;
///
/// let costs = |out: multiobjective_search::Output<u32, [u32; 2]>| -> Vec<_> {
///     out.solutions.into_iter().map(|s| s.cost).collect()
/// };
/// assert_eq!(costs(boba_star_parallel(&graph, 1, 3)), costs(boba_star(&graph, 1, 3)));
/// # Ok::<(), multiobjective_search::DimacsError>(())
/// ```
pub fn boba_star_parallel<G, C>(
    graph: &G,
    start: G::State,
    goal: G::State,
) -> Output<G::State, [C; 2]>
where
    G: ReverseGraph<Weight = [C; 2]> + Sync,
    G::State: Send + Sync,
    C: Cost + Send + Sync,
{
    let successors = |s: &G::State, out: &mut Vec<_>| graph.successors(s, out);
    let predecessors = |s: &G::State, out: &mut Vec<_>| graph.predecessors(s, out);
    let Some(bounds) = bounds::Bounds::new_parallel(&start, &goal, successors, predecessors) else {
        return Output {
            solutions: Vec::new(),
            stats: Stats::default(),
        };
    };

    let forward_h = heuristic_fn(|s: &G::State| bounds.forward(s));
    let backward_h = heuristic_fn(|s: &G::State| bounds.backward(s));
    let completions = (bounds::Forward(&bounds), bounds::Backward(&bounds));

    bidirectional(graph, start, goal, forward_h, backward_h, completions)
}

/// [`boba_star_with`](crate::boba_star_with) with the two directions on two
/// threads.
///
/// Same requirements and the same front as
/// [`boba_star_with`](crate::boba_star_with).
///
/// # Panics
///
/// Panics if a search thread panics, which only a panicking graph,
/// heuristic or cost implementation can cause.
pub fn boba_star_with_parallel<G, C, F, B>(
    graph: &G,
    start: G::State,
    goal: G::State,
    forward_h: F,
    backward_h: B,
) -> Output<G::State, [C; 2]>
where
    G: ReverseGraph<Weight = [C; 2]> + Sync,
    G::State: Send + Sync,
    C: Cost + Send + Sync,
    F: Heuristic<G::State, [C; 2]> + Send,
    B: Heuristic<G::State, [C; 2]> + Send,
{
    let completions = (NoCompletions, NoCompletions);

    bidirectional(graph, start, goal, forward_h, backward_h, completions)
}

/// What one direction shares with the other: its best second cost and a
/// channel for its first expansions, plus the common stop flag.
struct Link<'s, S, C> {
    /// This direction's second cost of its last solution.
    mine: &'s Mutex<Option<C>>,
    /// The other direction's, read as this direction's primary bound.
    theirs: &'s Mutex<Option<C>>,
    /// Sends this direction's first expansions to the other.
    tune_out: Sender<(S, C)>,
    /// Receives the other direction's first expansions.
    tune_in: Receiver<(S, C)>,
    /// Set by whichever direction finishes first.
    done: &'s AtomicBool,
}

/// Runs both directions on two threads and joins their outputs.
fn bidirectional<G, C, F, B, KF, KB>(
    graph: &G,
    start: G::State,
    goal: G::State,
    forward_h: F,
    mut backward_h: B,
    (forward_k, backward_k): (KF, KB),
) -> Output<G::State, [C; 2]>
where
    G: ReverseGraph<Weight = [C; 2]> + Sync,
    G::State: Send + Sync,
    C: Cost + Send + Sync,
    F: Heuristic<G::State, [C; 2]> + Send,
    B: Heuristic<G::State, [C; 2]> + Send,
    KF: Completions<G::State, C> + Send,
    KB: Completions<G::State, C> + Send,
{
    let backward_h = heuristic_fn(move |s: &G::State| backward_h.estimate(s).map(swap));
    let reversed = Reversed(graph);
    let (forward_best, backward_best) = (Mutex::new(None), Mutex::new(None));
    let done = AtomicBool::new(false);
    let (to_backward, from_forward) = channel();
    let (to_forward, from_backward) = channel();

    let mut forward = Search::with_completions(
        graph,
        forward_h,
        equals(goal.clone()),
        Tolerance::EXACT,
        forward_k,
    );
    let mut backward = Search::with_completions(
        &reversed,
        backward_h,
        equals(start.clone()),
        Tolerance::EXACT,
        backward_k,
    );
    forward.push_start(start);
    backward.push_start(goal);

    let forward_link = Link {
        mine: &forward_best,
        theirs: &backward_best,
        tune_out: to_backward,
        tune_in: from_backward,
        done: &done,
    };
    let backward_link = Link {
        mine: &backward_best,
        theirs: &forward_best,
        tune_out: to_forward,
        tune_in: from_forward,
        done: &done,
    };

    let (forward, backward) = std::thread::scope(|scope| {
        let forward = scope.spawn(move || run(forward, &forward_link));
        let backward = run(backward, &backward_link);

        (forward.join().expect("forward search panicked"), backward)
    });

    merge(forward, backward)
}

/// Drives one direction until it is done or the other one is.
fn run<G, H, C, P, K>(
    mut search: Search<'_, G, H, C, P, K>,
    link: &Link<'_, G::State, C>,
) -> Output<G::State, [C; 2]>
where
    G: Graph<Weight = [C; 2]>,
    H: Heuristic<G::State, [C; 2]>,
    C: Cost,
    P: Fn(&G::State) -> bool,
    K: Completions<G::State, C>,
{
    while !link.done.load(Ordering::Acquire) {
        search.set_primary_bound(*lock(link.theirs));
        for (state, floor) in link.tune_in.try_iter() {
            search.raise_secondary(state, floor);
        }

        if !search.step() {
            link.done.store(true, Ordering::Release);
            break;
        }

        if let Some(expansion) = search.take_first_expansion() {
            // The other direction may have finished and dropped its end.
            let _ = link.tune_out.send(expansion);
        }

        *lock(link.mine) = search.secondary_bound();
    }

    search.into_output()
}

/// Locks `mutex`, ignoring poisoning: the guarded bound is a plain value
/// that a panicking thread cannot leave half-written.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
