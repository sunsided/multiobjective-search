//! Integration tests: every algorithm and the Pareto frontier,
//! cross-checked against a brute-force reference that shares no code with
//! the crate.

mod aggregate;
mod anytime;
mod apex;
mod approx;
mod benchmarks;
mod boa;
mod boba;
mod brute_force;
mod check;
mod dimacs;
mod disjoint;
mod fixtures;
mod front;
mod mvh;
mod namoa;
#[cfg(feature = "parallel")]
mod parallel;
#[cfg(feature = "parallel")]
mod parallel_moa;
mod reference;
#[cfg(feature = "parallel")]
mod sip;
mod survey;
