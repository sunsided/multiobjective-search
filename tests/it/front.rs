//! Property tests for [`Front`] against the naive filter, in two to four
//! dimensions, and for [`TreeFront`] against [`Front`].

use multiobjective_search::{Front, TreeFront};
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

    /// The tree frontier holds the same points as the `Vec` frontier and
    /// answers every query and insertion the same way, after every step.
    #[test]
    fn tree_matches_vec_four(points in prop::collection::vec(point::<4>(), 0..60)) {
        check_tree(&points)?;
    }

    /// As [`tree_matches_vec_four`], for five objectives.
    #[test]
    fn tree_matches_vec_five(points in prop::collection::vec(point::<5>(), 0..60)) {
        check_tree(&points)?;
    }

    /// As [`tree_matches_vec_four`], for two objectives.
    #[test]
    fn tree_matches_vec_two(points in prop::collection::vec(point::<2>(), 0..60)) {
        check_tree(&points)?;
    }
}

/// Inserts `points` into a [`TreeFront`] and a [`Front`] side by side and
/// checks that both agree on every query, insertion and final content.
fn check_tree<const N: usize>(points: &[[u32; N]]) -> Result<(), TestCaseError> {
    let mut tree = TreeFront::new();
    let mut vec = Front::new();

    for p in points {
        prop_assert_eq!(tree.weakly_dominates(p), vec.weakly_dominates(p));
        prop_assert_eq!(tree.insert(*p), vec.insert(*p));
        prop_assert_eq!(tree.len(), vec.len());
    }

    let points = tree.to_vec();
    prop_assert_eq!(points.as_slice(), vec.as_slice());

    Ok(())
}
