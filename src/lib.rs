//! # Multi-objective search
//!
//! Multi-objective shortest paths: BOA*, BOBA*, EMOA* and NAMOA* over Pareto frontiers.
//!
//! ## Planned scope
//!
//! - BOA*: bi-objective A*
//! - BOBA*: bidirectional bi-objective A*
//! - EMOA*: multi-objective A*
//! - NAMOA*: classical multi-objective A* baseline
//! - Fast Pareto-frontier storage per state
//!
//! ## Status
//!
//! Pre-release. Version 0.0.0 reserves the name; the API is not started yet and will change.
//!
//! Repository: <https://github.com/sunsided/multiobjective-search>

#![no_std]

/// Pareto dominance between cost vectors.
pub trait Dominates {
    /// Returns `true` if `self` is no worse than `other` in every objective
    /// and strictly better in at least one.
    fn dominates(&self, other: &Self) -> bool;
}
