//! A*pex tests: the multiplicative approximation guarantee for two to four
//! objectives, property-tested against the brute-force Pareto front.

use multiobjective_search::{Epsilon, Output, ZeroHeuristic, apex_star, heuristic_fn};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::check::{check, check_paths};
use crate::fixtures::{EdgeList, Problem, exact_heuristic, problems};

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

/// Ten percent in every objective.
const TEN_PERCENT: [Epsilon; 2] = [Epsilon::new(1, 10); 2];

/// The solution cost vectors of `out`.
fn costs(out: &Output<usize, [u32; 2]>) -> Vec<[u32; 2]> {
    out.solutions.iter().map(|s| s.cost).collect()
}

/// Two routes to state 1 merge there only under the paper's `f`-based
/// bound: `g = [20, 10]` is not within 10 % of the apex `[10, 10]`, but
/// `f = [120, 110]` is within 10 % of `[110, 110]`. One pair is expanded
/// at state 1 instead of two.
#[test]
fn merges_under_the_f_based_bound() {
    let graph = EdgeList {
        nodes: 3,
        edges: vec![(0, 1, [10, 20]), (0, 1, [20, 10]), (1, 2, [100, 100])],
    };
    let h = [[100, 100], [100, 100], [0, 0]];
    let heuristic = heuristic_fn(|v: &usize| Some(h[*v]));

    let out = apex_star(&graph, heuristic, 0, |v| *v == 2, TEN_PERCENT);

    // [20, 10] has the lexicographically smaller reverse cost, so it
    // represents the merged pair.
    assert_eq!(costs(&out), [[120, 110]]);
    assert_eq!(out.stats.expanded, 2);
}

/// A goal pair popped after a solution merges into it (paper line 11): the
/// solution `[10, 20]` and the later goal pair `[11, 11]` become one pair
/// with apex `[10, 11]` represented by `[11, 11]`.
#[test]
fn merges_goal_pairs_into_solutions() {
    let graph = EdgeList {
        nodes: 3,
        edges: vec![(0, 1, [10, 20]), (0, 2, [11, 1]), (2, 1, [0, 10])],
    };

    let out = apex_star(&graph, ZeroHeuristic, 0, |v| *v == 1, TEN_PERCENT);

    assert_eq!(costs(&out), [[11, 11]]);
    assert_eq!(out.solutions[0].path, [0, 2, 1]);
}

/// A solution that covers a pruned pair lowers its apex to that pair's `f`
/// (paper lines 20 to 22), which stops a later merge that would uncover
/// it. Solution `[2, 2, 1]` prunes `[3, 1, 1]` and its apex drops to
/// `[2, 1, 1]`; merging goal pair `[3, 4, 0]` would then need
/// `4 <= 2 · 1`, so both stay. Without the update the apex `[2, 2, 0]`
/// would let `[3, 4, 0]` represent both, and `[3, 1, 1]` would be lost.
#[test]
fn pruning_lowers_the_solution_apex() {
    let graph = EdgeList {
        nodes: 3,
        edges: vec![(1, 2, [3, 4, 0]), (1, 0, [2, 2, 1]), (1, 2, [3, 1, 1])],
    };
    let eps = [Epsilon::new(2, 1), Epsilon::new(1, 1), Epsilon::ZERO];

    let out = apex_star(&graph, ZeroHeuristic, 1, |v| *v != 1, eps);
    let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();

    assert_eq!(costs, [[2, 2, 1], [3, 4, 0]]);
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
