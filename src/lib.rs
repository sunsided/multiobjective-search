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
//! | [`boa_star_anytime`] | 2 | consistent | A-BOA*ε: anytime, ever better approximations down to the exact front; `_ws` adds weighted-sum pruning, `_basic` is the no-reuse baseline. |
//! | [`pp_a_star_iterative`] | 2 | consistent | Iterative-PP-A*: PP-A* from scratch over a falling tolerance schedule, the anytime baseline. |
//! | [`pp_a_star`] | 2 | consistent | PP-A*: path pairs, factors `1 + ε1` and `1 + ε2` on both objectives. |
//! | [`apex_star`] | any `N` | consistent | A*pex: apex-path pairs, a factor `1 + ε_i` per objective. |
//! | [`boba_star`] | 2 | computed internally | Bidirectional; needs a [`ReverseGraph`] and a single goal. |
//! | [`boba_star_with`] | 2 | consistent, both directions | [`boba_star`] with caller-supplied heuristics. |
//! | [`emoa_star`] | any `N` | consistent | Lazy open list, truncated dominance; `O(log n)` checks for 3 objectives. |
//! | [`emoa_star_tree`] | any `N` | consistent | EMOA* with the paper's balanced-tree frontiers. |
//! | [`namoa_star_dr`] | any `N` | consistent | NAMOA* with dimensionality reduction. |
//! | [`namoa_star_dr_mvh_lazy`] | any `N` | admissible, multi-valued | L-NAMOA*dr-mvh: NAMOA*dr with a [`MvHeuristic`] and lazy dimensionality reduction. |
//! | [`namoa_star_dr_mvh`] | any `N` | consistent, multi-valued | NAMOA*dr-mvh: one label per estimate; a baseline, slower than the lazy variant. |
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
//! - [`MvHeuristic`], [`mv_heuristic_fn`] and [`SingleValued`]: multi-valued
//!   heuristics, a set of lower bounds per state.
//! - [`Cost`] (from `search-core`) and [`Dominates`]: scalar costs and
//!   Pareto dominance.
//! - [`Output`]: solutions sorted lexicographically by cost, plus counters.
//! - [`RoadGraph`]: a loader for 9th DIMACS Challenge road networks
//!   (`.gr` distance and time files, `.co` coordinates).
//! - [`Epsilon`]: a rational tolerance `ε = num / den` for the approximate
//!   searches, tested with [`Cost::saturating_times`].
//!
//! ## Features
//!
//! - `std`: links `std`; no API of its own.
//! - `parallel` (implies `std`): multi-threaded BOBA*,
//!   `boba_star_parallel` and `boba_star_with_parallel`.
//!
//! Without features the crate is `no_std` with `alloc`.
//!
//! ## Status
//!
//! Pre-release. Version 0.0.0 reserves the name; the API is a draft and will change.
//!
//! Repository: <https://github.com/sunsided/multiobjective-search>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod anytime_boa;
mod apex;
mod boa;
mod boba;
mod cost;
mod dimacs;
mod emoa;
mod epsilon;
mod frontier_search;
mod graph;
mod mvh;
mod namoa;
mod namoa_mvh;
mod output;
mod pareto;
mod pp_a;
mod reference;

pub use anytime_boa::{
    AnytimeFront, ApproxFactor, WeightedSum, boa_star_anytime, boa_star_anytime_basic,
    boa_star_anytime_ws,
};
pub use apex::{MergeRule, apex_star, apex_star_with};
pub use boa::{boa_star, boa_star_approx, boa_star_eps};
pub use boba::{boba_star, boba_star_with};
#[cfg(feature = "parallel")]
pub use boba::{boba_star_parallel, boba_star_with_parallel};
pub use cost::{Dominates, add, weakly_dominates};
pub use dimacs::{DimacsError, DimacsErrorKind, RoadGraph};
pub use emoa::{emoa_star, emoa_star_tree};
pub use epsilon::Epsilon;
pub use graph::{Graph, ReverseGraph, ZeroHeuristic};
pub use mvh::{MvHeuristic, MvHeuristicFn, MvhTable, SingleValued, apex_mvh, mv_heuristic_fn};
pub use namoa::{namoa_star, namoa_star_dr};
pub use namoa_mvh::{namoa_star_dr_mvh, namoa_star_dr_mvh_lazy};
pub use output::{Output, Solution, Stats};
pub use pareto::{BiFront, Front, TreeFront};
pub use pp_a::{pp_a_star, pp_a_star_iterative};
pub use reference::brute_force;
pub use search_core::{Cost, Heuristic, HeuristicFn, heuristic_fn};
