//! # Multi-objective search
//!
//! Multi-objective shortest paths: BOA*, BOBA*, EMOA* and NAMOA* over Pareto frontiers.
//!
//! Given a graph whose edges carry cost vectors, the algorithms find one path
//! for every Pareto-optimal cost vector from a start state to a goal.
//!
//! ## Contents
//!
//! | Function | Objectives | Heuristic | Notes |
//! |---|---|---|---|
//! | [`boa_star`] | 2 | consistent | Fastest unidirectional choice for two objectives. |
//! | [`boa_star_approx`] | 2 | consistent | BOA* with an additive tolerance on the second objective. |
//! | [`boa_star_eps`] | 2 | consistent | BOA*ε: BOA* with a factor `1 + ε` on the second objective. |
//! | [`pp_a_star`] | 2 | consistent | PP-A*: path pairs, factors `1 + ε1` and `1 + ε2` on both objectives. |
//! | [`boba_star`] | 2 | computed internally | Bidirectional; needs a [`ReverseGraph`] and a single goal. |
//! | [`emoa_star`] | any `N` | consistent | Lazy open list, truncated dominance; `O(log n)` checks for 3 objectives. |
//! | [`emoa_star_tree`] | any `N` | consistent | EMOA* with the paper's balanced-tree frontiers. |
//! | [`namoa_star_dr`] | any `N` | consistent | NAMOA* with dimensionality reduction. |
//! | [`namoa_star`] | any `N` | admissible | Classic baseline; label-setting with [`ZeroHeuristic`]. |
//! | [`brute_force`] | any `N` | none | Exhaustive reference for tests on small graphs. |
//!
//! Supporting types:
//!
//! - [`Front`] and [`BiFront`]: Pareto frontiers, `O(log n)` queries for two
//!   objectives; [`TreeFront`]: a balanced-tree frontier for four or more.
//! - [`Graph`] and [`ReverseGraph`]: the search-space interface.
//! - [`Heuristic`], [`heuristic_fn`] and [`ZeroHeuristic`]: per-objective
//!   lower bounds. The trait comes from `search-core` and takes `&mut self`;
//!   algorithms take the heuristic by value, so pass `&mut h` to keep it.
//! - [`Cost`] (from `search-core`) and [`Dominates`]: scalar costs and
//!   Pareto dominance.
//! - [`Output`]: solutions sorted lexicographically by cost, plus counters.
//! - [`Epsilon`]: a rational tolerance `ε = num / den` for the approximate
//!   searches, tested with [`Cost::saturating_times`].
//!
//! ## Status
//!
//! Pre-release. Version 0.0.0 reserves the name; the API is a draft and will change.
//!
//! Repository: <https://github.com/sunsided/multiobjective-search>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod boa;
mod boba;
mod cost;
mod emoa;
mod epsilon;
mod frontier_search;
mod graph;
mod namoa;
mod output;
mod pareto;
mod pp_a;
mod reference;

pub use boa::{boa_star, boa_star_approx, boa_star_eps};
pub use boba::boba_star;
pub use cost::{Dominates, add, weakly_dominates};
pub use emoa::{emoa_star, emoa_star_tree};
pub use epsilon::Epsilon;
pub use graph::{Graph, ReverseGraph, ZeroHeuristic};
pub use namoa::{namoa_star, namoa_star_dr};
pub use output::{Output, Solution, Stats};
pub use pareto::{BiFront, Front, TreeFront};
pub use pp_a::pp_a_star;
pub use reference::brute_force;
pub use search_core::{Cost, Heuristic, HeuristicFn, heuristic_fn};
