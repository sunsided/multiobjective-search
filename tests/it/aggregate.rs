//! Objective-aggregation tests against brute force over the aggregated
//! front, exact and approximate.

use multiobjective_search::{Aggregation, Epsilon, ZeroHeuristic, aggregated_search};
use proptest::prelude::*;

use crate::fixtures::{Problem, problems};

/// A path-cost extension over three hidden objectives.
type Extend<'a> = dyn Fn(&[u32; 3], &[u32; 3]) -> [u32; 3] + 'a;

/// Hidden costs of every simple path that ends at its first goal, under
/// the extension `extend`.
fn hidden_costs(p: &Problem<3>, extend: &Extend<'_>) -> Vec<[u32; 3]> {
    fn walk(
        p: &Problem<3>,
        at: usize,
        g: [u32; 3],
        seen: &mut Vec<bool>,
        extend: &Extend<'_>,
        out: &mut Vec<[u32; 3]>,
    ) {
        if p.goals[at] {
            out.push(g);
            return;
        }

        seen[at] = true;
        for &(f, t, c) in &p.graph.edges {
            if f == at && !seen[t] {
                walk(p, t, extend(&g, &c), seen, extend, out);
            }
        }

        seen[at] = false;
    }

    let mut out = Vec::new();
    walk(
        p,
        p.start,
        [0; 3],
        &mut vec![false; p.graph.nodes],
        extend,
        &mut out,
    );
    out
}

/// The Pareto front of `points`, sorted and without duplicates.
fn pareto(points: &[[u32; 2]]) -> Vec<[u32; 2]> {
    let mut front: Vec<_> = points
        .iter()
        .copied()
        .filter(|a| {
            !points
                .iter()
                .any(|b| b != a && b[0] <= a[0] && b[1] <= a[1])
        })
        .collect();
    front.sort_unstable();
    front.dedup();
    front
}

/// Checks one extension and aggregation, exact and with tolerance `eps`.
fn check(
    p: &Problem<3>,
    extend: impl Fn(&[u32; 3], &[u32; 3]) -> [u32; 3] + Copy,
    aggregate: impl Fn(&[u32; 3]) -> [u32; 2] + Copy,
    eps: [Epsilon; 2],
) -> Result<(), TestCaseError> {
    let all: Vec<[u32; 2]> = hidden_costs(p, &extend).iter().map(aggregate).collect();
    let front = pareto(&all);
    let goal = |v: &usize| p.goals[*v];
    let functions = Aggregation { extend, aggregate };

    let (exact, hidden) = aggregated_search(
        &p.graph,
        ZeroHeuristic,
        p.start,
        goal,
        functions,
        [Epsilon::ZERO; 2],
    );
    let costs: Vec<_> = exact.solutions.iter().map(|s| s.cost).collect();
    prop_assert_eq!(&costs, &front);
    for (s, h) in exact.solutions.iter().zip(&hidden) {
        prop_assert_eq!(aggregate(h), s.cost);
        prop_assert_eq!(s.path.first(), Some(&p.start));
        prop_assert!(p.goals[*s.path.last().unwrap()]);
    }

    let within = |e: Epsilon, x: u32, y: u32| {
        u64::from(e.den()) * u64::from(x) <= u64::from(e.den() + e.num()) * u64::from(y)
    };
    let (approx, _) = aggregated_search(&p.graph, ZeroHeuristic, p.start, goal, functions, eps);
    let costs: Vec<_> = approx.solutions.iter().map(|s| s.cost).collect();
    for c in &costs {
        prop_assert!(all.contains(c));
    }

    for s in &front {
        prop_assert!(
            costs
                .iter()
                .any(|r| within(eps[0], r[0], s[0]) && within(eps[1], r[1], s[1])),
            "{:?} not covered",
            s
        );
    }

    Ok(())
}

proptest! {
    /// Additive hidden objectives aggregated into two, and a bottleneck
    /// hidden objective, exact and approximate.
    #[test]
    fn aggregated_fronts_match_brute_force(p in problems::<3>(), n in 0..3_u32, d in 1..4_u32) {
        let eps = [Epsilon::new(n, d); 2];
        let add = |g: &[u32; 3], c: &[u32; 3]| [g[0] + c[0], g[1] + c[1], g[2] + c[2]];
        check(&p, add, |h: &[u32; 3]| [h[0] + h[1], h[1].max(h[2])], eps)?;
        let worst = |g: &[u32; 3], c: &[u32; 3]| [g[0] + c[0], g[1] + c[1], g[2].max(c[2])];
        check(&p, worst, |h: &[u32; 3]| [h[0], h[1] + h[2]], eps)?;
    }
}
