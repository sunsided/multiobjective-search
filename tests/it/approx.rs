//! BOA*ε and PP-A* tests: the multiplicative approximation guarantee,
//! property-tested against the brute-force Pareto front.

use multiobjective_search::{
    Epsilon, Heuristic, Output, ZeroHeuristic, boa_star_eps, heuristic_fn, pp_a_star,
};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::check::{check, check_paths};
use crate::fixtures::{EdgeList, Problem, exact_heuristic, problems};

/// The two approximate searches under test.
#[derive(Debug, Clone, Copy)]
enum Algorithm {
    /// [`boa_star_eps`], which uses only the second tolerance.
    BoaEps,
    /// [`pp_a_star`].
    PpA,
}

/// Runs `algorithm` on `problem` with `heuristic` and tolerances `eps`.
fn solve<H: Heuristic<usize, [u32; 2]>>(
    algorithm: Algorithm,
    problem: &Problem<2>,
    heuristic: H,
    eps: [Epsilon; 2],
) -> Output<usize, [u32; 2]> {
    let goal = |v: &usize| problem.goals[*v];
    let (graph, start) = (&problem.graph, problem.start);

    match algorithm {
        Algorithm::BoaEps => boa_star_eps(graph, heuristic, start, goal, eps[1]),
        Algorithm::PpA => pp_a_star(graph, heuristic, start, goal, eps),
    }
}

/// The tolerances `algorithm` actually applies: BOA*ε never relaxes the
/// first objective.
fn effective(algorithm: Algorithm, eps: [Epsilon; 2]) -> [Epsilon; 2] {
    match algorithm {
        Algorithm::BoaEps => [Epsilon::ZERO, eps[1]],
        Algorithm::PpA => eps,
    }
}

/// Returns `true` if `x <= (1 + eps) * y`, in exact integer arithmetic.
fn within(eps: Epsilon, x: u32, y: u32) -> bool {
    u64::from(eps.den()) * u64::from(x) <= u64::from(eps.den() + eps.num()) * u64::from(y)
}

/// Checks the approximation contract of `out`: every Pareto-optimal cost
/// is covered within `eps`, the solutions are real paths, mutually
/// non-dominated and sorted, and (for BOA*ε) Pareto-optimal themselves.
fn check_approx(
    algorithm: Algorithm,
    problem: &Problem<2>,
    out: &Output<usize, [u32; 2]>,
    eps: [Epsilon; 2],
) -> Result<(), TestCaseError> {
    let front = pareto_costs(problem);
    let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
    let eps = effective(algorithm, eps);

    for s in &front {
        prop_assert!(
            costs
                .iter()
                .any(|r| within(eps[0], r[0], s[0]) && within(eps[1], r[1], s[1])),
            "{:?} not covered by {:?}",
            s,
            costs
        );
    }

    for pair in costs.windows(2) {
        prop_assert!(pair[0][0] < pair[1][0] && pair[0][1] > pair[1][1]);
    }

    prop_assert_eq!(costs.is_empty(), front.is_empty());
    if matches!(algorithm, Algorithm::BoaEps) {
        prop_assert!(costs.iter().all(|c| front.contains(c)));
    }

    check_paths(problem, out)
}

/// A strategy for small tolerances `num / den`, zero included.
fn epsilon() -> impl Strategy<Value = Epsilon> {
    (0..=3_u32, 1..=4_u32).prop_map(|(num, den)| Epsilon::new(num, den))
}

/// A two-objective problem from 0 to 1 over parallel edges.
fn parallel(edges: &[[u32; 2]]) -> Problem<2> {
    Problem {
        graph: EdgeList {
            nodes: 2,
            edges: edges.iter().map(|&c| (0, 1, c)).collect(),
        },
        start: 0,
        goals: vec![false, true],
    }
}

/// PP-A* merges near-equal first costs; BOA*ε cannot.
#[test]
fn pp_a_merges_on_first_objective() {
    let p = parallel(&[[100, 40], [105, 30], [150, 20], [200, 19]]);
    let eps = [Epsilon::new(1, 10); 2];
    let costs = |a| -> Vec<_> {
        solve(a, &p, ZeroHeuristic, eps)
            .solutions
            .into_iter()
            .map(|s| s.cost)
            .collect()
    };

    assert_eq!(costs(Algorithm::PpA), [[105, 30], [150, 20]]);
    assert_eq!(costs(Algorithm::BoaEps), [[100, 40], [105, 30], [150, 20]]);
}

/// A goal pair merges into the last solution when the result is bounded.
#[test]
fn pp_a_merges_into_last_solution() {
    // Both reach the goal through different states, so they only meet as
    // solutions.
    let p = Problem {
        graph: EdgeList {
            nodes: 4,
            edges: vec![
                (0, 1, [10, 9]),
                (0, 2, [11, 5]),
                (1, 3, [0, 0]),
                (2, 3, [0, 0]),
            ],
        },
        start: 0,
        goals: vec![false, false, false, true],
    };
    let out = solve(
        Algorithm::PpA,
        &p,
        ZeroHeuristic,
        [Epsilon::new(1, 10), Epsilon::ZERO],
    );
    let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();

    assert_eq!(costs, [[11, 5]]);
    assert_eq!(out.solutions[0].path, [0, 2, 3]);
}

proptest! {
    /// Both searches meet the guarantee for random tolerances, with the
    /// zero and the exact heuristic.
    #[test]
    fn covers_front(p in problems::<2>(), e0 in epsilon(), e1 in epsilon()) {
        let h = exact_heuristic(&p);
        let eps = [e0, e1];

        for algorithm in [Algorithm::BoaEps, Algorithm::PpA] {
            check_approx(algorithm, &p, &solve(algorithm, &p, ZeroHeuristic, eps), eps)?;
            let exact = heuristic_fn(|v: &usize| h[*v]);
            check_approx(algorithm, &p, &solve(algorithm, &p, exact, eps), eps)?;
        }
    }

    /// Zero tolerances give the exact Pareto front.
    #[test]
    fn zero_is_exact(p in problems::<2>()) {
        let h = exact_heuristic(&p);

        for algorithm in [Algorithm::BoaEps, Algorithm::PpA] {
            check(&p, &solve(algorithm, &p, ZeroHeuristic, [Epsilon::ZERO; 2]))?;
            let exact = heuristic_fn(|v: &usize| h[*v]);
            check(&p, &solve(algorithm, &p, exact, [Epsilon::ZERO; 2]))?;
        }
    }

    /// Edge costs scaled up, so that small tolerances matter, with large
    /// tolerances too.
    #[test]
    fn covers_scaled_front(p in problems::<2>(), num in 0..=20_u32, den in 1..=20_u32) {
        let mut p = p;
        for edge in &mut p.graph.edges {
            edge.2 = edge.2.map(|c| c * 25 + 1);
        }

        let eps = [Epsilon::new(num, den), Epsilon::new(den, num + 1)];
        for algorithm in [Algorithm::BoaEps, Algorithm::PpA] {
            check_approx(algorithm, &p, &solve(algorithm, &p, ZeroHeuristic, eps), eps)?;
        }
    }
}
