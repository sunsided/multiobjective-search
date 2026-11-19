//! # Multi-objective search
//!
//! Multi-objective shortest paths: BOA*, BOBA*, EMOA* and NAMOA* over Pareto frontiers.
//!
//! ## Contents
//!
//! - [`BiFront`]: a two-objective Pareto frontier.
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

mod cost;
mod pareto;

pub use cost::{Cost, Dominates, add, weakly_dominates};
pub use pareto::BiFront;
