//! Search results shared by all algorithms.

use alloc::vec::Vec;

/// One Pareto-optimal path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution<S, W> {
    /// Total cost vector of [`Solution::path`].
    pub cost: W,
    /// States from the start to a goal, both included. Never empty.
    pub path: Vec<S>,
}

/// Counters collected during a search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Labels taken from the open list and expanded (goal labels excluded).
    pub expanded: usize,
    /// Labels that survived pruning at generation and entered the open list,
    /// the start label included.
    pub generated: usize,
}

/// The result of a multi-objective search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output<S, W> {
    /// One solution per Pareto-optimal cost vector, sorted by cost
    /// lexicographically ascending (for two objectives: first strictly
    /// ascending, second strictly descending). Empty when no goal is
    /// reachable. [`boa_star_approx`](crate::boa_star_approx) returns a
    /// covering subset instead.
    pub solutions: Vec<Solution<S, W>>,
    /// Search effort counters.
    pub stats: Stats,
}
