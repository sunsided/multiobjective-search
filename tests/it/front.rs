//! Property tests for [`Front`] against the naive filter, in two to four
//! dimensions.

use multiobjective_search::Front;
use proptest::prelude::*;

use crate::brute_force::non_dominated;

/// A strategy for points with small coordinates, so ties are common.
fn point<const N: usize>() -> impl Strategy<Value = [u32; N]> {
    prop::array::uniform(0..=6_u32)
}

/// Checks that after inserting `points` the frontier equals the naive
/// non-dominated set, and that `weakly_dominates` and `insert` agree with a
/// linear scan for `probe`.
fn check<const N: usize>(points: &[[u32; N]], probe: [u32; N]) -> Result<(), TestCaseError> {
    let mut front = Front::new();
    for p in points {
        front.insert(*p);
    }

    let expected = non_dominated(points);
    prop_assert_eq!(front.as_slice(), expected.as_slice());

    let dominated = points.iter().any(|p| (0..N).all(|i| p[i] <= probe[i]));
    prop_assert_eq!(front.weakly_dominates(&probe), dominated);
    prop_assert_eq!(front.insert(probe), !dominated);

    Ok(())
}

proptest! {
    /// Two objectives: the binary-search path.
    #[test]
    fn two_objectives(points in prop::collection::vec(point::<2>(), 0..40), probe in point()) {
        check(&points, probe)?;
    }

    /// Three objectives: the prefix-scan path.
    #[test]
    fn three_objectives(points in prop::collection::vec(point::<3>(), 0..40), probe in point()) {
        check(&points, probe)?;
    }

    /// Four objectives.
    #[test]
    fn four_objectives(points in prop::collection::vec(point::<4>(), 0..40), probe in point()) {
        check(&points, probe)?;
    }
}
