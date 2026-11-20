//! The public brute-force reference against the test-only one.

use multiobjective_search::brute_force;
use proptest::prelude::*;

use crate::check::check;
use crate::fixtures::{Problem, problems};

/// Runs the public reference and checks it against the test oracle.
fn check_reference<const N: usize>(p: &Problem<N>) -> Result<(), TestCaseError> {
    check(p, &brute_force(&p.graph, p.start, |v| p.goals[*v]))
}

proptest! {
    /// Two objectives.
    #[test]
    fn two_objectives(p in problems::<2>()) {
        check_reference(&p)?;
    }

    /// Three objectives.
    #[test]
    fn three_objectives(p in problems::<3>()) {
        check_reference(&p)?;
    }
}
