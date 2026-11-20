//! Integration tests: every algorithm and the Pareto frontier,
//! cross-checked against a brute-force reference that shares no code with
//! the crate.

mod apex;
mod approx;
mod boa;
mod boba;
mod brute_force;
mod check;
mod dimacs;
mod fixtures;
mod front;
mod mvh;
mod namoa;
#[cfg(feature = "parallel")]
mod parallel;
mod reference;
