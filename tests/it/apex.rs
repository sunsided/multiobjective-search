//! A*pex tests: the multiplicative approximation guarantee for two to four
//! objectives, property-tested against the brute-force Pareto front.

use multiobjective_search::{Epsilon, Output, ZeroHeuristic, apex_star, heuristic_fn};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::check::{check, check_paths};
use crate::fixtures::{Problem, exact_heuristic, problems};

/// Returns `true` if `x <= (1 + eps) * y`, in exact integer arithmetic.
fn within(eps: Epsilon, x: u32, y: u32) -> bool {
    u64::from(eps.den()) * u64::from(x) <= u64::from(eps.den() + eps.num()) * u64::from(y)
}

/// Checks the approximation contract of `out`: every Pareto-optimal cost
/// is covered within `eps`, and the solutions are real paths, mutually
/// non-dominated and lexicographically ascending.
fn check_apex<const N: usize>(
    problem: &Problem<N>,
    out: &Output<usize, [u32; N]>,
    eps: [Epsilon; N],
) -> Result<(), TestCaseError> {
    let front = pareto_costs(problem);
    let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();

    for s in &front {
        prop_assert!(
            costs
                .iter()
                .any(|r| (0..N).all(|i| within(eps[i], r[i], s[i]))),
            "{:?} not covered by {:?}",
            s,
            costs
        );
    }

    for (k, a) in costs.iter().enumerate() {
        for b in &costs[k + 1..] {
            prop_assert!(a < b);
            prop_assert!(!(0..N).all(|i| a[i] <= b[i]) && !(0..N).all(|i| b[i] <= a[i]));
        }
    }

    prop_assert_eq!(costs.is_empty(), front.is_empty());
    check_paths(problem, out)
}

/// Runs the guarantee check with the zero and the exact heuristic, and the
/// exactness check with zero tolerances.
fn check_all<const N: usize>(problem: &Problem<N>, eps: [Epsilon; N]) -> Result<(), TestCaseError> {
    let h = exact_heuristic(problem);
    let goal = |v: &usize| problem.goals[*v];
    let (graph, start) = (&problem.graph, problem.start);

    check_apex(
        problem,
        &apex_star(graph, ZeroHeuristic, start, goal, eps),
        eps,
    )?;
    let exact = heuristic_fn(|v: &usize| h[*v]);
    check_apex(problem, &apex_star(graph, exact, start, goal, eps), eps)?;
    check(
        problem,
        &apex_star(graph, ZeroHeuristic, start, goal, [Epsilon::ZERO; N]),
    )?;
    check(
        problem,
        &apex_star(graph, exact, start, goal, [Epsilon::ZERO; N]),
    )
}

/// A strategy for tolerances `num / den` up to 300 %, zero included.
fn epsilon() -> impl Strategy<Value = Epsilon> {
    (0..=3_u32, 1..=4_u32).prop_map(|(num, den)| Epsilon::new(num, den))
}

/// `problem` with every edge cost scaled to `25 c + 1`, so that small
/// tolerances matter.
fn scaled<const N: usize>(mut problem: Problem<N>) -> Problem<N> {
    for edge in &mut problem.graph.edges {
        edge.2 = edge.2.map(|c| c * 25 + 1);
    }

    problem
}

proptest! {
    /// Two objectives.
    #[test]
    fn two_objectives(p in problems::<2>(), eps in prop::array::uniform(epsilon())) {
        check_all(&p, eps)?;
        check_all(&scaled(p), eps)?;
    }

    /// Three objectives.
    #[test]
    fn three_objectives(p in problems::<3>(), eps in prop::array::uniform(epsilon())) {
        check_all(&p, eps)?;
        check_all(&scaled(p), eps)?;
    }

    /// Four objectives.
    #[test]
    fn four_objectives(p in problems::<4>(), eps in prop::array::uniform(epsilon())) {
        check_all(&p, eps)?;
        check_all(&scaled(p), eps)?;
    }
}
