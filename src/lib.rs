//! # Multi-objective search
//!
//! Multi-objective shortest paths: BOA*, BOBA*, EMOA* and NAMOA* over Pareto frontiers.
//!
//! Given a graph whose edges carry cost vectors, the algorithms find one path
//! for every Pareto-optimal cost vector from a start state to a goal.
//!
//! ## Contents
//!
//! - [`boa_star`]: bi-objective A* (Hernández et al., ICAPS 2020).
//! - [`BiFront`]: a two-objective Pareto frontier.
//! - [`Graph`] and [`Heuristic`]: the search-space interface.
//! - [`Cost`] and [`Dominates`]: scalar costs and Pareto dominance.
//!
//! ## Planned
//!
//! - BOBA*: bidirectional bi-objective A*
//! - EMOA*: multi-objective A* with fast frontier checks
//! - NAMOA*: classical multi-objective A* baseline
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
mod frontier_search;
mod graph;
mod namoa;
mod output;
mod pareto;
mod reference;

pub use boa::{boa_star, boa_star_approx};
pub use boba::boba_star;
pub use cost::{Cost, Dominates, add, weakly_dominates};
pub use emoa::emoa_star;
pub use graph::{Graph, Heuristic, ReverseGraph, ZeroHeuristic};
pub use namoa::{namoa_star, namoa_star_dr};
pub use output::{Output, Solution, Stats};
pub use pareto::{BiFront, Front};
pub use reference::brute_force;
