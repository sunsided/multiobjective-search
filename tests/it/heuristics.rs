//! Heuristic builders and label correcting against exact references.

use multiobjective_search::{
    DifferentialHeuristic, Heuristic, IdealPoint, label_correcting, namoa_star,
};
use proptest::prelude::*;

use crate::check::check;
use crate::fixtures::{exact_heuristic, problems, single_goal_problems};

proptest! {
    /// The ideal point is the exact per-objective distance.
    #[test]
    fn ideal_point_is_exact(p in problems::<3>()) {
        let goals: Vec<usize> = (0..p.graph.nodes).filter(|&v| p.goals[v]).collect();
        let mut h = IdealPoint::new(&p.graph, goals);
        let exact = exact_heuristic(&p);
        for (v, e) in exact.iter().enumerate() {
            prop_assert_eq!(h.estimate(&v), *e);
        }
    }

    /// Differential heuristics never exceed the exact distance, are
    /// consistent, and guide NAMOA* to the exact front.
    #[test]
    fn differential_heuristics(p in single_goal_problems::<2>(), landmarks in prop::collection::vec(0..7_usize, 1..3)) {
        let goal = (0..p.graph.nodes).find(|&v| p.goals[v]).unwrap();
        let landmarks: Vec<usize> = landmarks.into_iter().map(|l| l % p.graph.nodes).collect();
        let dh = DifferentialHeuristic::new(&p.graph, landmarks);
        let mut h = dh.towards(&goal);
        let exact = exact_heuristic(&p);
        for (v, x) in exact.iter().enumerate() {
            let e = h.estimate(&v).unwrap();
            if let Some(x) = x {
                prop_assert!(e[0] <= x[0] && e[1] <= x[1]);
            }
        }

        // Consistency holds wherever the successor can reach the goal.
        for &(u, v, c) in p.graph.edges.iter().filter(|e| exact[e.1].is_some()) {
            let (hu, hv) = (h.estimate(&u).unwrap(), h.estimate(&v).unwrap());
            prop_assert!(hu[0] <= c[0] + hv[0] && hu[1] <= c[1] + hv[1]);
        }

        check(&p, &namoa_star(&p.graph, dh.towards(&goal), p.start, |v: &usize| *v == goal))?;
    }

    /// Label correcting returns the Pareto front.
    #[test]
    fn label_correcting_matches_brute_force(p2 in problems::<2>(), p3 in problems::<3>()) {
        check(&p2, &label_correcting(&p2.graph, p2.start, |v: &usize| p2.goals[*v]))?;
        check(&p3, &label_correcting(&p3.graph, p3.start, |v: &usize| p3.goals[*v]))?;
    }
}
