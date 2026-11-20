//! Multi-valued heuristic tests: the paper's example graph, and property
//! cross-checks of NAMOA*dr-mvh and L-NAMOA*dr-mvh against the brute-force
//! reference with consistent and with merely admissible MVHs.

use multiobjective_search::{
    Graph, Output, SingleValued, heuristic_fn, mv_heuristic_fn, namoa_star, namoa_star_dr_mvh,
    namoa_star_dr_mvh_lazy,
};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::check::check;
use crate::fixtures::{Problem, exact_heuristic, problems};

/// The search graph of Figure 2 of Wolff, Felner, Salzman (arXiv
/// 2606.20644): start 0, `s1` = 1, `s2` = 2, `s3` = 3, goal 4.
struct Figure2;

impl Graph for Figure2 {
    type State = u8;
    type Weight = [u32; 2];

    fn successors(&self, state: &u8, out: &mut Vec<(u8, [u32; 2])>) {
        match *state {
            0 => out.extend([(4, [3, 15]), (1, [7, 9]), (3, [10, 6])]),
            1 => out.push((3, [1, 1])),
            3 => out.extend([(4, [2, 7]), (2, [7, 1])]),
            2 => out.push((4, [1, 1])),
            _ => {}
        }
    }
}

/// The admissible and consistent MVH of Table 1 for [`Figure2`].
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the shape mv_heuristic_fn expects"
)]
fn table1(state: &u8, out: &mut Vec<[u32; 2]>) {
    match *state {
        0 => out.extend([[2, 14], [11, 12], [15, 11], [17, 7]]),
        1 => out.extend([[2, 7], [8, 2]]),
        2 => out.push([1, 1]),
        3 => out.extend([[1, 6], [7, 1]]),
        _ => out.push([0, 0]),
    }
}

/// The solution costs of `out`.
fn costs<S, const N: usize>(out: &Output<S, [u32; N]>) -> Vec<[u32; N]> {
    out.solutions.iter().map(|s| s.cost).collect()
}

/// Both MVH searches find all four Pareto-optimal paths of Figure 2, the
/// ones the naive combination of the paper's Lemma 1 misses included, and
/// agree with NAMOA* without a heuristic.
#[test]
fn figure_2_has_four_solutions() {
    let expected = [[3, 15], [12, 13], [16, 12], [18, 8]];
    let goal = |s: &u8| *s == 4;

    let lazy = namoa_star_dr_mvh_lazy(&Figure2, mv_heuristic_fn(table1), 0, goal);
    let eager = namoa_star_dr_mvh(&Figure2, mv_heuristic_fn(table1), 0, goal);
    let plain = namoa_star(&Figure2, multiobjective_search::ZeroHeuristic, 0, goal);

    assert_eq!(costs(&lazy), expected);
    assert_eq!(costs(&eager), expected);
    assert_eq!(costs(&plain), expected);
    assert_eq!(lazy.solutions[2].path, [0, 1, 3, 2, 4]);
    assert_eq!(eager.solutions[2].path, [0, 1, 3, 2, 4]);
}

/// The exact MVH: per state, the Pareto-optimal costs to a goal. It is
/// consistent; a dead end gets no estimate.
fn exact_mvh<const N: usize>(problem: &Problem<N>) -> Vec<Vec<[u32; N]>> {
    (0..problem.graph.nodes)
        .map(|start| {
            pareto_costs(&Problem {
                start,
                ..problem.clone()
            })
        })
        .collect()
}

/// An admissible, generally inconsistent MVH: the exact one, weakened per
/// state by `kind[v] % 4`: kept, the zero vector, the component-wise
/// minimum (the ideal point), or every estimate halved.
fn weakened_mvh<const N: usize>(problem: &Problem<N>, kind: &[u8]) -> Vec<Vec<[u32; N]>> {
    exact_mvh(problem)
        .into_iter()
        .zip(kind)
        .map(|(set, &kind)| match kind % 4 {
            0 => set,
            1 => vec![[0; N]],
            2 if set.is_empty() => set,
            2 => vec![core::array::from_fn(|i| {
                set.iter().map(|v| v[i]).min().unwrap()
            })],
            _ => set.iter().map(|v| v.map(|c| c / 2)).collect(),
        })
        .collect()
}

/// Runs both MVH searches with `table` and checks them against brute force;
/// the eager one only when `consistent`.
fn check_mvh<const N: usize>(
    problem: &Problem<N>,
    table: &[Vec<[u32; N]>],
    consistent: bool,
) -> Result<(), TestCaseError> {
    let (graph, start) = (&problem.graph, problem.start);
    let goal = |v: &usize| problem.goals[*v];
    let mvh = || mv_heuristic_fn(|v: &usize, out: &mut Vec<[u32; N]>| out.extend(&table[*v]));

    check(problem, &namoa_star_dr_mvh_lazy(graph, mvh(), start, goal))?;
    if consistent {
        check(problem, &namoa_star_dr_mvh(graph, mvh(), start, goal))?;
    }

    Ok(())
}

/// All MVH checks for one problem: exact, weakened and single-valued.
fn check_all<const N: usize>(problem: &Problem<N>, kind: &[u8]) -> Result<(), TestCaseError> {
    check_mvh(problem, &exact_mvh(problem), true)?;
    check_mvh(problem, &weakened_mvh(problem, kind), false)?;

    let h = exact_heuristic(problem);
    let (graph, start) = (&problem.graph, problem.start);
    let goal = |v: &usize| problem.goals[*v];
    let single = || SingleValued(heuristic_fn(|v: &usize| h[*v]));
    check(
        problem,
        &namoa_star_dr_mvh_lazy(graph, single(), start, goal),
    )?;
    check(problem, &namoa_star_dr_mvh(graph, single(), start, goal))
}

proptest! {
    /// Two objectives.
    #[test]
    fn two_objectives(p in problems::<2>(), kind in prop::collection::vec(any::<u8>(), 7)) {
        check_all(&p, &kind)?;
    }

    /// Three objectives.
    #[test]
    fn three_objectives(p in problems::<3>(), kind in prop::collection::vec(any::<u8>(), 7)) {
        check_all(&p, &kind)?;
    }

    /// Four objectives.
    #[test]
    fn four_objectives(p in problems::<4>(), kind in prop::collection::vec(any::<u8>(), 7)) {
        check_all(&p, &kind)?;
    }
}
