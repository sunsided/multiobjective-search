//! Tests for the problem classes taken from the Craveirinha et al. survey:
//! minsum–maxmin paths, constrained shortest paths and the ranking method,
//! each against exhaustive enumeration of simple paths.

use multiobjective_search::{
    Graph, Stats, ZeroHeuristic, constrained_shortest_path, heuristic_fn, minsum_maxmin,
    minsum_maxmin_multi, ranking,
};
use proptest::prelude::*;

use crate::brute_force::pareto_costs;
use crate::check::{check, check_paths};
use crate::fixtures::{EdgeList, Problem, exact_heuristic, problems};

/// Every simple path from `node` to a goal, as the list of its edge costs.
fn simple_paths(
    problem: &Problem<2>,
    node: usize,
    visited: &mut Vec<bool>,
    edges: &mut Vec<[u32; 2]>,
    out: &mut Vec<Vec<[u32; 2]>>,
) {
    if problem.goals[node] {
        out.push(edges.clone());
    }

    visited[node] = true;
    for &(from, to, cost) in &problem.graph.edges {
        if from == node && !visited[to] {
            edges.push(cost);
            simple_paths(problem, to, visited, edges, out);
            edges.pop();
        }
    }

    visited[node] = false;
}

/// All simple start–goal paths of `problem`.
fn all_paths(problem: &Problem<2>) -> Vec<Vec<[u32; 2]>> {
    let mut out = Vec::new();
    simple_paths(
        problem,
        problem.start,
        &mut vec![false; problem.graph.nodes],
        &mut Vec::new(),
        &mut out,
    );
    out
}

/// The graph of `problem` with `(cost, capacity)` edges.
struct Capacities<'a>(&'a EdgeList<2>);

impl Graph for Capacities<'_> {
    type State = usize;
    type Weight = (u32, u32);

    fn successors(&self, state: &usize, out: &mut Vec<(usize, (u32, u32))>) {
        let mut edges = Vec::new();
        self.0.successors(state, &mut edges);
        out.extend(edges.into_iter().map(|(t, c)| (t, (c[0], c[1]))));
    }
}

/// `a` is at least as good as `b`: no larger cost, no smaller bottleneck
/// (`None` is unbounded).
fn width_dominates(a: (u32, Option<u32>), b: (u32, Option<u32>)) -> bool {
    let wider = match (a.1, b.1) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(x), Some(y)) => x >= y,
    };

    a.0 <= b.0 && wider
}

proptest! {
    /// `minsum_maxmin` returns exactly the non-dominated (sum, bottleneck)
    /// pairs, by cost ascending.
    #[test]
    fn minsum_maxmin_matches_enumeration(p in problems::<2>()) {
        let pairs: Vec<(u32, Option<u32>)> = all_paths(&p)
            .iter()
            .map(|edges| (edges.iter().map(|c| c[0]).sum(), edges.iter().map(|c| c[1]).min()))
            .collect();
        let mut front: Vec<_> = pairs
            .iter()
            .copied()
            .filter(|&a| !pairs.iter().any(|&b| b != a && width_dominates(b, a)))
            .collect();
        front.sort_by_key(|&(c, _)| c);
        front.dedup();

        let goal = |v: &usize| p.goals[*v];
        let out = minsum_maxmin(&Capacities(&p.graph), heuristic_fn(|_: &usize| Some(0_u32)), p.start, goal);
        let found: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
        prop_assert_eq!(&found, &front);

        for s in &out.solutions {
            prop_assert_eq!(s.path.first(), Some(&p.start));
            prop_assert!(p.goals[*s.path.last().unwrap()]);
        }
    }

    /// `constrained_shortest_path` is the lexicographic optimum among paths
    /// within the budget, for every budget.
    #[test]
    fn constrained_matches_enumeration(p in problems::<2>(), budget in 0..=12_u32) {
        let expected = all_paths(&p)
            .iter()
            .map(|edges| edges.iter().fold([0, 0], |s, c| [s[0] + c[0], s[1] + c[1]]))
            .filter(|c| c[1] <= budget)
            .min();
        let h = exact_heuristic(&p);
        let goal = |v: &usize| p.goals[*v];
        let mut stats = Stats::default();

        let zero = constrained_shortest_path(&p.graph, ZeroHeuristic, p.start, goal, budget, &mut stats);
        prop_assert_eq!(zero.as_ref().map(|s| s.cost), expected);
        let exact = heuristic_fn(|v: &usize| h[*v]);
        let informed = constrained_shortest_path(&p.graph, exact, p.start, goal, budget, &mut stats);
        prop_assert_eq!(informed.map(|s| s.cost), expected);
    }

    /// `ranking` returns the Pareto front.
    #[test]
    fn ranking_matches_brute_force(p in problems::<2>()) {
        let h = exact_heuristic(&p);
        let goal = |v: &usize| p.goals[*v];
        check(&p, &ranking(&p.graph, ZeroHeuristic, p.start, goal))?;
        check(&p, &ranking(&p.graph, heuristic_fn(|v: &usize| h[*v]), p.start, goal))?;
        prop_assert!(!pareto_costs(&p).is_empty() || ranking(&p.graph, ZeroHeuristic, p.start, goal).solutions.is_empty());
    }
}

/// Every simple start–goal path of `problem` as its list of edge costs,
/// for any number of objectives.
fn paths_n<const N: usize>(problem: &Problem<N>) -> Vec<Vec<[u32; N]>> {
    fn walk<const N: usize>(
        problem: &Problem<N>,
        node: usize,
        visited: &mut Vec<bool>,
        edges: &mut Vec<[u32; N]>,
        out: &mut Vec<Vec<[u32; N]>>,
    ) {
        if problem.goals[node] {
            out.push(edges.clone());
        }

        visited[node] = true;
        for &(from, to, cost) in &problem.graph.edges {
            if from == node && !visited[to] {
                edges.push(cost);
                walk(problem, to, visited, edges, out);
                edges.pop();
            }
        }

        visited[node] = false;
    }

    let mut out = Vec::new();
    walk(
        problem,
        problem.start,
        &mut vec![false; problem.graph.nodes],
        &mut Vec::new(),
        &mut out,
    );
    out
}

/// A three-component edge list read as `S` sums and `3 - S` capacities.
struct Split<'a, const S: usize, const M: usize>(&'a EdgeList<3>);

impl<const S: usize, const M: usize> Graph for Split<'_, S, M> {
    type State = usize;
    type Weight = ([u32; S], [u32; M]);

    fn successors(&self, state: &usize, out: &mut Vec<(usize, ([u32; S], [u32; M]))>) {
        let mut edges = Vec::new();
        self.0.successors(state, &mut edges);
        out.extend(edges.into_iter().map(|(t, c)| {
            (
                t,
                (
                    core::array::from_fn(|i| c[i]),
                    core::array::from_fn(|i| c[S + i]),
                ),
            )
        }));
    }
}

/// The non-dominated `(sums, widths)` of the paths of `problem` (sums
/// minimized, widths maximized, `None` unbounded), sorted by sums.
fn sums_widths_front<const S: usize, const M: usize>(
    problem: &Problem<3>,
) -> Vec<([u32; S], [Option<u32>; M])> {
    let wide = |a: Option<u32>, b: Option<u32>| match (a, b) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(x), Some(y)) => x >= y,
    };
    let labels: Vec<([u32; S], [Option<u32>; M])> = paths_n(problem)
        .iter()
        .map(|edges| {
            (
                core::array::from_fn(|i| edges.iter().map(|c| c[i]).sum()),
                core::array::from_fn(|i| edges.iter().map(|c| c[S + i]).min()),
            )
        })
        .collect();
    let no_worse = |a: &([u32; S], [Option<u32>; M]), b: &([u32; S], [Option<u32>; M])| {
        (0..S).all(|i| a.0[i] <= b.0[i]) && (0..M).all(|i| wide(a.1[i], b.1[i]))
    };
    let mut front: Vec<_> = labels
        .iter()
        .copied()
        .filter(|a| !labels.iter().any(|b| b != a && no_worse(b, a)))
        .collect();
    front.sort_by_key(|a| a.0);
    front.dedup();
    front
}

/// Checks [`minsum_maxmin_multi`] with `S` sums and `M = 3 - S` widths.
fn check_multi<const S: usize, const M: usize>(p: &Problem<3>) -> Result<(), TestCaseError> {
    let goal = |v: &usize| p.goals[*v];
    let out = minsum_maxmin_multi(&Split::<S, M>(&p.graph), ZeroHeuristic, p.start, goal);
    let mut found: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
    let mut expected = sums_widths_front::<S, M>(p);
    // Equal sums order by width; compare as sets.
    found.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    expected.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    prop_assert_eq!(found, expected);
    Ok(())
}

/// Exact `sum / width` order for the brute force (`None` width is 0).
fn ratio_key(sum: u32, width: Option<u32>) -> (bool, u128, u128) {
    match width {
        Some(0) if sum > 0 => (true, 0, 1),
        None | Some(0) => (false, 0, 1),
        Some(w) => (false, u128::from(sum), u128::from(w)),
    }
}

proptest! {
    /// Several bottlenecks: one sum with two widths, two sums with one.
    #[test]
    fn multi_bottleneck_matches_enumeration(p in problems::<3>()) {
        check_multi::<1, 2>(&p)?;
        check_multi::<2, 1>(&p)?;
    }

    /// The cost/capacity ratio and the quickest path match the best over
    /// all simple paths, compared exactly.
    #[test]
    fn ratio_and_quickest_match_enumeration(p in problems::<2>(), sigma in 0..50_u32) {
        let goal = |v: &usize| p.goals[*v];
        let zero = heuristic_fn(|_: &usize| Some(0_u32));
        let pairs: Vec<(u32, Option<u32>)> = all_paths(&p)
            .iter()
            .map(|edges| (edges.iter().map(|c| c[0]).sum(), edges.iter().map(|c| c[1]).min()))
            .collect();

        let less = |a: (bool, u128, u128), b: (bool, u128, u128)| match (a.0, b.0) {
            (true, _) => false,
            (false, true) => true,
            _ => a.1 * b.2 < b.1 * a.2,
        };
        let best_ratio = pairs.iter().map(|&(s, w)| ratio_key(s, w)).reduce(|a, b| if less(b, a) { b } else { a });
        let got = multiobjective_search::min_cost_capacity_ratio(&Capacities(&p.graph), zero, p.start, goal);
        prop_assert_eq!(got.is_some(), best_ratio.is_some());
        if let (Some(got), Some(best)) = (got, best_ratio) {
            let key = ratio_key(got.cost.0, got.cost.1);
            prop_assert!(!less(best, key) && !less(key, best));
        }

        // Time as an exact fraction (d·c + σ) / c; `None` and σ = 0 give d.
        let time = |d: u32, c: Option<u32>| -> (bool, u128, u128) {
            match c {
                Some(0) if sigma > 0 => (true, 0, 1),
                Some(c) if sigma > 0 => (false, u128::from(d) * u128::from(c) + u128::from(sigma), u128::from(c)),
                _ => (false, u128::from(d), 1),
            }
        };
        let best_time = pairs.iter().map(|&(d, c)| time(d, c)).reduce(|a, b| if less(b, a) { b } else { a });
        let got = multiobjective_search::quickest_path(&Capacities(&p.graph), zero, p.start, goal, sigma);
        prop_assert_eq!(got.is_some(), best_time.is_some());
        if let (Some(got), Some(best)) = (got, best_time) {
            let key = time(got.cost.0, got.cost.1);
            prop_assert!(!less(best, key) && !less(key, best), "{:?} vs {:?}", key, best);
        }
    }
}

/// The vertices of the lower-left convex hull of a sorted 2D Pareto front
/// (Andrew's monotone chain, collinear points dropped).
fn hull_vertices(front: &[[u32; 2]]) -> Vec<[u32; 2]> {
    let cross = |o: [u32; 2], a: [u32; 2], b: [u32; 2]| {
        let (ox, oy) = (i64::from(o[0]), i64::from(o[1]));
        (i64::from(a[0]) - ox) * (i64::from(b[1]) - oy)
            - (i64::from(a[1]) - oy) * (i64::from(b[0]) - ox)
    };
    let mut hull: Vec<[u32; 2]> = Vec::new();
    for &p in front {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull
}

proptest! {
    /// The two-phase method returns the Pareto front, and its first phase
    /// exactly the hull vertices.
    #[test]
    fn two_phase_matches_brute_force(p in problems::<2>()) {
        let h = exact_heuristic(&p);
        let goal = |v: &usize| p.goals[*v];
        check(&p, &multiobjective_search::two_phase(&p.graph, ZeroHeuristic, p.start, goal))?;
        check(&p, &multiobjective_search::two_phase(&p.graph, heuristic_fn(|v: &usize| h[*v]), p.start, goal))?;

        let front = pareto_costs(&p);
        let supported = multiobjective_search::supported_solutions(&p.graph, ZeroHeuristic, p.start, goal);
        let costs: Vec<_> = supported.solutions.iter().map(|s| s.cost).collect();
        prop_assert_eq!(costs, hull_vertices(&front));
    }
}

/// The brute-force achievement `max_i w_i (c_i - r_i)` as a signed integer.
fn achieve<const N: usize>(c: &[u32; N], r: &[u32; N], w: &[u32; N]) -> i64 {
    (0..N)
        .map(|i| i64::from(w[i]) * (i64::from(c[i]) - i64::from(r[i])))
        .max()
        .unwrap()
}

/// Checks a reference-point session: the first `split` solutions follow
/// `(r1, w1)`, the rest `(r2, w2)`, each in (achievement, cost) order over
/// the not yet proposed Pareto-optimal costs.
fn check_reference<const N: usize>(
    problem: &Problem<N>,
    (r1, w1): ([u32; N], [u32; N]),
    (r2, w2): ([u32; N], [u32; N]),
    split: usize,
) -> Result<(), TestCaseError> {
    let h = exact_heuristic(problem);
    let goal = |v: &usize| problem.goals[*v];
    let mut search = multiobjective_search::ReferencePoint::new(
        &problem.graph,
        heuristic_fn(|v: &usize| h[*v]),
        problem.start,
        goal,
        r1,
        w1,
    );
    let mut left = pareto_costs(problem);
    let mut got = Vec::new();

    for step in 0.. {
        if step == split {
            search.aim(r2, w2);
        }

        let (aim, weight) = if step < split { (r1, w1) } else { (r2, w2) };
        left.sort_by_key(|c| (achieve(c, &aim, &weight), *c));
        let Some(solution) = search.next() else { break };
        prop_assert!(!left.is_empty());
        prop_assert_eq!(solution.cost, left.remove(0));
        got.push(solution);
    }

    prop_assert!(left.is_empty());
    check_paths(
        problem,
        &multiobjective_search::Output {
            solutions: got,
            stats: Stats::default(),
        },
    )
}

proptest! {
    /// Reference-point sessions with a change of aim, two and three
    /// objectives.
    #[test]
    fn reference_point_matches_brute_force(
        p2 in problems::<2>(), p3 in problems::<3>(),
        r in prop::array::uniform3(0..12_u32), q in prop::array::uniform3(0..12_u32),
        w in prop::array::uniform3(1..4_u32), v in prop::array::uniform3(1..4_u32),
        split in 0..4_usize,
    ) {
        check_reference(&p2, ([r[0], r[1]], [w[0], w[1]]), ([q[0], q[1]], [v[0], v[1]]), split)?;
        check_reference(&p3, (r, w), (q, v), split)?;
    }
}

/// The Pareto-maximal kept values of a knapsack by enumerating subsets,
/// sorted descending in the first value (ascending in lost value).
fn knapsack_front<const N: usize, const M: usize>(
    items: &[([u32; M], [u32; N])],
    capacity: [u32; M],
) -> Vec<[u32; N]> {
    let kept: Vec<[u32; N]> = (0..1_u32 << items.len())
        .filter(|mask| {
            (0..M).all(|r| {
                items
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask >> i & 1 == 1)
                    .map(|(_, it)| it.0[r])
                    .sum::<u32>()
                    <= capacity[r]
            })
        })
        .map(|mask| {
            core::array::from_fn(|k| {
                items
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask >> i & 1 == 1)
                    .map(|(_, it)| it.1[k])
                    .sum()
            })
        })
        .collect();
    let mut front: Vec<_> = kept
        .iter()
        .copied()
        .filter(|a| !kept.iter().any(|b| b != a && (0..N).all(|k| b[k] >= a[k])))
        .collect();
    front.sort_unstable();
    front.dedup();
    front
}

/// Solves a knapsack with NAMOA* and checks it against enumeration.
fn check_knapsack<const N: usize, const M: usize>(
    items: &[([u32; M], [u32; N])],
    capacity: [u32; M],
) -> Result<(), TestCaseError> {
    let total: [u32; N] = core::array::from_fn(|k| items.iter().map(|it| it.1[k]).sum());
    let expected = knapsack_front(items, capacity);
    let knapsack = multiobjective_search::KnapsackGraph::new(items.to_vec(), capacity);
    let out = multiobjective_search::namoa_star(&knapsack, ZeroHeuristic, knapsack.start(), |s| {
        knapsack.is_goal(s)
    });

    let mut kept: Vec<[u32; N]> = Vec::new();
    for s in &out.solutions {
        let picks = knapsack.taken(&s.path);
        let value: [u32; N] = core::array::from_fn(|k| picks.iter().map(|&i| items[i].1[k]).sum());
        prop_assert_eq!(value, core::array::from_fn(|k| total[k] - s.cost[k]));
        prop_assert!(
            (0..M).all(|r| picks.iter().map(|&i| items[i].0[r]).sum::<u32>() <= capacity[r])
        );
        kept.push(value);
    }

    kept.sort_unstable();
    prop_assert_eq!(kept, expected);
    Ok(())
}

/// A graph read with objective 1 zeroed, for label counting.
struct FirstOnly<'a>(&'a EdgeList<2>);

impl Graph for FirstOnly<'_> {
    type State = usize;
    type Weight = [u32; 2];

    fn successors(&self, state: &usize, out: &mut Vec<(usize, [u32; 2])>) {
        self.0.successors(state, out);
    }
}

proptest! {
    /// Multi-objective knapsacks: two values with two resources, three
    /// values with one.
    #[test]
    fn knapsack_matches_enumeration(
        items2 in prop::collection::vec((prop::array::uniform2(0..5_u32), prop::array::uniform2(0..6_u32)), 0..7),
        items3 in prop::collection::vec((prop::array::uniform1(0..5_u32), prop::array::uniform3(0..6_u32)), 0..7),
        cap in prop::array::uniform2(0..12_u32),
    ) {
        check_knapsack(&items2, cap)?;
        check_knapsack(&items3, [cap[0]])?;
    }

    /// Label counting: (cost, distinct labels) with label `c[1] mod 3`.
    #[test]
    fn label_count_matches_enumeration(p in problems::<2>()) {
        let pairs: Vec<[u32; 2]> = all_paths(&p)
            .iter()
            .map(|edges| {
                let labels: std::collections::BTreeSet<u32> = edges.iter().map(|c| c[1] % 3).collect();
                [edges.iter().map(|c| c[0]).sum(), u32::try_from(labels.len()).unwrap()]
            })
            .collect();
        let mut front: Vec<_> = pairs.iter().copied().filter(|a| !pairs.iter().any(|b| b != a && b[0] <= a[0] && b[1] <= a[1])).collect();
        front.sort_unstable();
        front.dedup();

        let graph = FirstOnly(&p.graph);
        let counted = multiobjective_search::LabelCount::new(&graph, 1, |_: &usize, _: &usize, w: &[u32; 2]| w[1] % 3);
        let out = multiobjective_search::namoa_star(&counted, ZeroHeuristic, counted.start(p.start), |s: &(usize, _)| p.goals[s.0]);
        let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
        prop_assert_eq!(costs, front);
    }
}

proptest! {
    /// The FPTAS stays within the budget and within `1 + ε` of the exact
    /// constrained optimum, and is exact for `ε = 0`.
    #[test]
    fn fptas_within_tolerance(p in problems::<2>(), budget in 0..=40_u32, num in 0..=3_u32, den in 1..=4_u32) {
        let p = scaled2(p);
        let goal = |v: &usize| p.goals[*v];
        let mut stats = Stats::default();
        let exact = constrained_shortest_path(&p.graph, ZeroHeuristic, p.start, goal, budget, &mut stats);
        let eps = multiobjective_search::Epsilon::new(num, den);
        let approx = multiobjective_search::fptas_constrained(&p.graph, p.start, goal, budget, eps, &mut stats);

        prop_assert_eq!(exact.is_some(), approx.is_some());
        if let (Some(exact), Some(approx)) = (exact, approx) {
            prop_assert!(approx.cost[1] <= budget);
            prop_assert!(u64::from(den) * u64::from(approx.cost[0]) <= u64::from(den + num) * u64::from(exact.cost[0]));
            if num == 0 {
                prop_assert_eq!(approx.cost[0], exact.cost[0]);
            }

            check_paths(&p, &multiobjective_search::Output { solutions: vec![approx], stats })?;
        }
    }
}

/// Costs scaled to `5 c + 1`, so that tolerances merge labels.
fn scaled2(mut problem: Problem<2>) -> Problem<2> {
    for edge in &mut problem.graph.edges {
        edge.2 = edge.2.map(|c| c * 5 + 1);
    }

    problem
}
