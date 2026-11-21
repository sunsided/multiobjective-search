//! Property tests for Pareto dominance on cost vectors: strict dominance is
//! a strict partial order, weak dominance is its reflexive closure, and
//! lexicographic order extends both, which the searches rely on when they
//! pop labels in lexicographic order.

use multiobjective_search::{Dominates, weakly_dominates};
use proptest::prelude::*;

/// A strategy for vectors with small coordinates, so ties and comparable
/// pairs are common.
fn vector() -> impl Strategy<Value = [u32; 3]> {
    prop::array::uniform(0..=3_u32)
}

proptest! {
    /// Strict dominance is irreflexive, asymmetric and transitive.
    #[test]
    fn strict_partial_order(a in vector(), b in vector(), c in vector()) {
        prop_assert!(!a.dominates(&a));
        prop_assert!(!(a.dominates(&b) && b.dominates(&a)));
        if a.dominates(&b) && b.dominates(&c) {
            prop_assert!(a.dominates(&c));
        }
    }

    /// Weak dominance is strict dominance or equality, and is antisymmetric.
    #[test]
    fn weak_is_reflexive_closure(a in vector(), b in vector()) {
        prop_assert_eq!(weakly_dominates(&a, &b), a.dominates(&b) || a == b);
        if weakly_dominates(&a, &b) && weakly_dominates(&b, &a) {
            prop_assert_eq!(a, b);
        }
    }

    /// A dominating vector is lexicographically smaller.
    #[test]
    fn lexicographic_order_extends_dominance(a in vector(), b in vector()) {
        if a.dominates(&b) {
            prop_assert!(a < b);
        }
    }
}
