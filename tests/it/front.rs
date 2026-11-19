//! Property tests for [`BiFront`] against the naive filter.

use multiobjective_search::BiFront;
use proptest::prelude::*;

use crate::brute_force::non_dominated;

/// A strategy for 2D points with small coordinates, so ties are common.
fn point() -> impl Strategy<Value = [u32; 2]> {
    [0..=6_u32, 0..=6_u32]
}

proptest! {
    /// After inserting any sequence of points, the frontier equals the
    /// naive non-dominated set of those points.
    #[test]
    fn front_matches_naive_filter(points in prop::collection::vec(point(), 0..40)) {
        let mut front = BiFront::new();
        for p in &points {
            front.insert(*p);
        }

        let expected = non_dominated(&points);
        prop_assert_eq!(front.as_slice(), expected.as_slice());
    }

    /// `weakly_dominates` agrees with a linear scan, and `insert` returns
    /// its negation.
    #[test]
    fn weak_dominance_matches_scan(
        points in prop::collection::vec(point(), 0..40),
        probe in point(),
    ) {
        let mut front = BiFront::new();
        for p in &points {
            front.insert(*p);
        }

        let expected = points.iter().any(|p| p[0] <= probe[0] && p[1] <= probe[1]);
        prop_assert_eq!(front.weakly_dominates(&probe), expected);
        prop_assert_eq!(front.insert(probe), !expected);
    }
}
