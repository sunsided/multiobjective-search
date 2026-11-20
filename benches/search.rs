//! Benchmarks for every search on pseudo-random grid graphs, and for the
//! Pareto frontier that they share.

use std::fmt::Write;
use std::hint::black_box;

use criterion::Criterion;
use multiobjective_search::{
    Epsilon, Front, Graph, MvHeuristic, ReverseGraph, RoadGraph, SingleValued, TreeFront,
    WeightedSum, ZeroHeuristic, apex_mvh, apex_star, boa_star, boa_star_anytime,
    boa_star_anytime_basic, boa_star_anytime_ws, boa_star_eps, boba_star, emoa_star,
    emoa_star_tree, heuristic_fn, mv_heuristic_fn, namoa_star, namoa_star_dr, namoa_star_dr_mvh,
    namoa_star_dr_mvh_lazy, pp_a_star,
};

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_two_objectives(&mut criterion);
    bench_anytime(&mut criterion);
    bench_road(&mut criterion);
    bench_three_objectives(&mut criterion);
    bench_four_objectives(&mut criterion);
    bench_multi_valued(&mut criterion);
    bench_front(&mut criterion);
    bench_front_four(&mut criterion);
    criterion.final_summary();
}

/// A four-connected `side` x `side` grid whose edge costs are a hash of the
/// unordered cell pair, so every edge costs the same in both directions and
/// the grid is its own reverse.
struct Grid<const N: usize> {
    /// Cells per row and per column.
    side: u32,
}

impl<const N: usize> Grid<N> {
    /// The cost of the edge between cells `a` and `b`, each objective in
    /// `1..=10`.
    fn cost(a: u32, b: u32) -> [u32; N] {
        let (low, high) = (a.min(b), a.max(b));
        let mut x = u64::from(low) << 32 | u64::from(high);

        core::array::from_fn(|_| {
            // splitmix64 step
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            u32::try_from((z ^ (z >> 31)) % 10).expect("below 10") + 1
        })
    }

    /// The last cell, opposite the start corner `0`.
    fn goal(&self) -> u32 {
        self.side * self.side - 1
    }
}

impl<const N: usize> Graph for Grid<N> {
    type State = u32;
    type Weight = [u32; N];

    fn successors(&self, state: &u32, out: &mut Vec<(u32, [u32; N])>) {
        let (x, y) = (state % self.side, state / self.side);
        let mut push = |next: u32| out.push((next, Self::cost(*state, next)));

        if x > 0 {
            push(state - 1);
        }

        if x + 1 < self.side {
            push(state + 1);
        }

        if y > 0 {
            push(state - self.side);
        }

        if y + 1 < self.side {
            push(state + self.side);
        }
    }
}

impl<const N: usize> ReverseGraph for Grid<N> {
    fn predecessors(&self, state: &u32, out: &mut Vec<(u32, [u32; N])>) {
        self.successors(state, out);
    }
}

/// All algorithms on a two-objective grid, uninformed except BOBA*, which
/// computes its own heuristics.
fn bench_two_objectives(c: &mut Criterion) {
    let grid = Grid::<2> { side: 24 };
    let goal = grid.goal();
    let is_goal = |s: &u32| *s == goal;
    let mut group = c.benchmark_group("grid_24x24_2obj");

    group.bench_function("boa_star", |b| {
        b.iter(|| boa_star(&grid, ZeroHeuristic, black_box(0), is_goal));
    });
    group.bench_function("boa_star_eps_10pct", |b| {
        let eps = Epsilon::new(1, 10);
        b.iter(|| boa_star_eps(&grid, ZeroHeuristic, black_box(0), is_goal, eps));
    });
    group.bench_function("pp_a_star_10pct", |b| {
        let eps = [Epsilon::new(1, 10); 2];
        b.iter(|| pp_a_star(&grid, ZeroHeuristic, black_box(0), is_goal, eps));
    });
    group.bench_function("boba_star", |b| {
        b.iter(|| boba_star(&grid, black_box(0), goal));
    });
    group.bench_function("namoa_star", |b| {
        b.iter(|| namoa_star(&grid, ZeroHeuristic, black_box(0), is_goal));
    });
    group.bench_function("namoa_star_dr", |b| {
        b.iter(|| namoa_star_dr(&grid, ZeroHeuristic, black_box(0), is_goal));
    });
    group.bench_function("emoa_star", |b| {
        b.iter(|| emoa_star(&grid, ZeroHeuristic, black_box(0), is_goal));
    });
    group.finish();
}

/// The `N`-objective algorithms on a three-objective grid, uninformed.
fn bench_three_objectives(c: &mut Criterion) {
    let grid = Grid::<3> { side: 8 };
    let goal = grid.goal();
    let is_goal = |s: &u32| *s == goal;
    let mut group = c.benchmark_group("grid_8x8_3obj");

    group.bench_function("apex_star_10pct", |b| {
        let eps = [Epsilon::new(1, 10); 3];
        b.iter(|| apex_star(&grid, ZeroHeuristic, black_box(0), is_goal, eps));
    });
    group.bench_function("namoa_star", |b| {
        b.iter(|| namoa_star(&grid, ZeroHeuristic, black_box(0), is_goal));
    });
    group.bench_function("namoa_star_dr", |b| {
        b.iter(|| namoa_star_dr(&grid, ZeroHeuristic, black_box(0), is_goal));
    });
    group.bench_function("emoa_star", |b| {
        b.iter(|| emoa_star(&grid, ZeroHeuristic, black_box(0), is_goal));
    });
    group.finish();
}

/// NAMOA*dr and both EMOA* frontiers on a four-objective grid, uninformed:
/// the truncated frontiers are three-dimensional.
fn bench_four_objectives(c: &mut Criterion) {
    let grid = Grid::<4> { side: 10 };
    let goal = grid.goal();
    let is_goal = |s: &u32| *s == goal;
    let mut group = c.benchmark_group("grid_10x10_4obj");

    group.bench_function("namoa_star_dr", |b| {
        b.iter(|| namoa_star_dr(&grid, ZeroHeuristic, black_box(0), is_goal));
    });
    group.bench_function("emoa_star", |b| {
        b.iter(|| emoa_star(&grid, ZeroHeuristic, black_box(0), is_goal));
    });
    group.bench_function("emoa_star_tree", |b| {
        b.iter(|| emoa_star_tree(&grid, ZeroHeuristic, black_box(0), is_goal));
    });
    group.finish();
}

/// Single-objective Dijkstra distances from `goal` on a grid (its own
/// reverse) under the scalar edge cost `cost`.
fn distances(grid: &Grid<2>, goal: u32, cost: impl Fn([u32; 2]) -> u32) -> Vec<u32> {
    let mut dist = vec![u32::MAX; (grid.side * grid.side) as usize];
    let mut heap = std::collections::BinaryHeap::from([std::cmp::Reverse((0, goal))]);
    let mut edges = Vec::new();

    while let Some(std::cmp::Reverse((d, s))) = heap.pop() {
        if d >= dist[s as usize] {
            continue;
        }

        dist[s as usize] = d;
        grid.successors(&s, &mut edges);
        for (t, w) in edges.drain(..) {
            heap.push(std::cmp::Reverse((d + cost(w), t)));
        }
    }

    dist
}

/// A-BOA*ε and its variants to the exact front, against BOA*, on a
/// two-objective grid with the exact per-objective heuristic (as in the
/// paper), `d = 4` and `w = 1`.
fn bench_anytime(c: &mut Criterion) {
    let grid = Grid::<2> { side: 24 };
    let goal = grid.goal();
    let is_goal = |s: &u32| *s == goal;
    let (h1, h2) = (
        distances(&grid, goal, |w| w[0]),
        distances(&grid, goal, |w| w[1]),
    );
    let sum = distances(&grid, goal, |w| w[0] + w[1]);
    let h = |s: &u32| Some([h1[*s as usize], h2[*s as usize]]);
    let mut group = c.benchmark_group("grid_24x24_2obj_anytime");

    group.bench_function("boa_star", |b| {
        b.iter(|| boa_star(&grid, heuristic_fn(h), black_box(0), is_goal));
    });
    group.bench_function("a_boa_star_eps", |b| {
        b.iter(|| {
            boa_star_anytime(
                &grid,
                heuristic_fn(h),
                black_box(0),
                is_goal,
                4,
                |_, _| {},
                |_: &_| false,
            )
        });
    });
    group.bench_function("a_boa_star_eps_w", |b| {
        b.iter(|| {
            let weighted = WeightedSum {
                weight: 1,
                heuristic: heuristic_fn(|s: &u32| Some(sum[*s as usize])),
            };
            boa_star_anytime_ws(
                &grid,
                heuristic_fn(h),
                black_box(0),
                is_goal,
                4,
                weighted,
                |_, _| {},
                |_: &_| false,
            )
        });
    });
    group.bench_function("basic_a_boa_star_eps", |b| {
        b.iter(|| {
            boa_star_anytime_basic(
                &grid,
                heuristic_fn(h),
                black_box(0),
                is_goal,
                4,
                |_, _| {},
                |_: &_| false,
            )
        });
    });
    group.finish();
}

/// The exact multi-valued heuristic of a grid: per cell, the Pareto set of
/// costs to `goal`. The grid is its own reverse, so this is one NAMOA* run
/// from `goal` to every cell.
fn exact_mvh<const N: usize>(grid: &Grid<N>, goal: u32) -> Vec<Vec<[u32; N]>> {
    (0..grid.side * grid.side)
        .map(|cell| {
            let out = namoa_star(grid, ZeroHeuristic, goal, |s: &u32| *s == cell);
            out.solutions.iter().map(|s| s.cost).collect()
        })
        .collect()
}

/// A multi-valued heuristic that looks up `table[cell]`.
fn table(table: &[Vec<[u32; 3]>]) -> impl MvHeuristic<u32, [u32; 3]> + '_ {
    mv_heuristic_fn(|s: &u32, out: &mut Vec<[u32; 3]>| out.extend(&table[*s as usize]))
}

/// NAMOA*dr with the ideal-point heuristic against both MVH searches on a
/// three-objective grid: with the exact (consistent) MVH, and with every
/// estimate halved (admissible, inconsistent; lazy search only).
fn bench_multi_valued(c: &mut Criterion) {
    let grid = Grid::<3> { side: 6 };
    let goal = grid.goal();
    let is_goal = |s: &u32| *s == goal;
    let exact = exact_mvh(&grid, goal);
    let halved: Vec<Vec<[u32; 3]>> = exact
        .iter()
        .map(|set| set.iter().map(|v| v.map(|c| c / 2)).collect())
        .collect();
    let ideal: Vec<[u32; 3]> = exact
        .iter()
        .map(|set| core::array::from_fn(|i| set.iter().map(|v| v[i]).min().unwrap_or(0)))
        .collect();
    let mut group = c.benchmark_group("grid_6x6_3obj_mvh");

    group.bench_function("namoa_star_dr_ideal", |b| {
        let h = heuristic_fn(|s: &u32| Some(ideal[*s as usize]));
        b.iter(|| namoa_star_dr(&grid, h, black_box(0), is_goal));
    });
    group.bench_function("lazy_ideal", |b| {
        let h = SingleValued(heuristic_fn(|s: &u32| Some(ideal[*s as usize])));
        b.iter(|| namoa_star_dr_mvh_lazy(&grid, h, black_box(0), is_goal));
    });
    group.bench_function("lazy_exact", |b| {
        b.iter(|| namoa_star_dr_mvh_lazy(&grid, table(&exact), black_box(0), is_goal));
    });
    group.bench_function("lazy_halved", |b| {
        b.iter(|| namoa_star_dr_mvh_lazy(&grid, table(&halved), black_box(0), is_goal));
    });
    group.bench_function("eager_exact", |b| {
        b.iter(|| namoa_star_dr_mvh(&grid, table(&exact), black_box(0), is_goal));
    });

    // Wolff et al.'s constructions: backward A*pex (admissible, "A") and
    // its consistency fix ("C"), at two tolerances.
    for (name, eps) in [
        ("10pct", Epsilon::new(1, 10)),
        ("50pct", Epsilon::new(1, 2)),
    ] {
        let built = apex_mvh(&grid, [goal], [eps; 3]);
        let mut fixed = built.clone();
        fixed.make_consistent(&grid);
        println!(
            "apex_mvh {name}: {} estimates, {} after the fix",
            built.size(),
            fixed.size()
        );

        group.bench_function(format!("build_apex_{name}"), |b| {
            b.iter(|| apex_mvh(&grid, [black_box(goal)], [eps; 3]));
        });
        group.bench_function(format!("build_fixed_{name}"), |b| {
            b.iter(|| {
                let mut table = apex_mvh(&grid, [black_box(goal)], [eps; 3]);
                table.make_consistent(&grid);
                table
            });
        });
        group.bench_function(format!("lazy_apex_{name}"), |b| {
            b.iter(|| namoa_star_dr_mvh_lazy(&grid, &built, black_box(0), is_goal));
        });
        group.bench_function(format!("lazy_fixed_{name}"), |b| {
            b.iter(|| namoa_star_dr_mvh_lazy(&grid, &fixed, black_box(0), is_goal));
        });
        group.bench_function(format!("eager_fixed_{name}"), |b| {
            b.iter(|| namoa_star_dr_mvh(&grid, &fixed, black_box(0), is_goal));
        });
    }

    group.finish();
}

/// Inserting 1000 pseudo-random points into a frontier, two and three
/// objectives.
fn bench_front(c: &mut Criterion) {
    let points2: Vec<[u32; 2]> = (0..1000).map(|i| Grid::<2>::cost(i, i + 1)).collect();
    let points3: Vec<[u32; 3]> = (0..1000).map(|i| Grid::<3>::cost(i, i + 1)).collect();
    let mut group = c.benchmark_group("front_insert_1000");

    group.bench_function("two_objectives", |b| {
        b.iter(|| {
            let mut front = Front::new();
            for p in &points2 {
                front.insert(black_box(*p));
            }

            front
        });
    });
    group.bench_function("three_objectives", |b| {
        b.iter(|| {
            let mut front = Front::new();
            for p in &points3 {
                front.insert(black_box(*p));
            }

            front
        });
    });
    group.finish();
}

/// Inserting mutually non-dominated four-objective points (near-constant
/// coordinate sum), so the frontier grows to almost its full size: the
/// sorted `Vec` frontier against the balanced tree, at a size where the
/// `Vec` wins and one where the tree does.
fn bench_front_four(c: &mut Criterion) {
    for size in [1000, 16_000] {
        let points = antichain(size);
        let mut group = c.benchmark_group(format!("front_insert_{size}_4obj"));
        group.sample_size(10);

        group.bench_function("vec", |b| {
            b.iter(|| {
                let mut front = Front::new();
                for p in &points {
                    front.insert(black_box(*p));
                }

                front
            });
        });
        group.bench_function("tree", |b| {
            b.iter(|| {
                let mut front = TreeFront::new();
                for p in &points {
                    front.insert(black_box(*p));
                }

                front
            });
        });
        group.finish();
    }
}

/// `size` pseudo-random four-objective points whose coordinates sum to
/// 30000 plus a little noise, so few dominate each other.
fn antichain(size: u64) -> Vec<[u32; 4]> {
    (0..size)
        .map(|i| {
            let [a, b, c, noise] = [0, 1, 2, 3].map(|k| splitmix(i * 4 + k));
            let (a, b, c) = (a % 10_000, b % 10_000, c % 10_000);
            [a, b, c, 30_000 - a - b - c + noise % 50]
        })
        .collect()
}

/// One splitmix64 output for `x`, truncated to `u32`.
fn splitmix(x: u64) -> u32 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    u32::try_from((z ^ (z >> 31)) >> 32).expect("upper 32 bits")
}

/// A generated road network in 9th DIMACS format, `side` x `side`
/// intersections with two-way streets: distances of 50 to 149 units and
/// travel times of 3 to 12 per unit, every sixth row and column a fast
/// road. Generated, not an excerpt of the challenge data, whose licence is
/// unclear. Returns the distance and the time `.gr` file.
fn road_network(side: u32) -> (String, String) {
    let id = |x: u32, y: u32| y * side + x + 1;
    let mut arcs = Vec::new();

    for y in 0..side {
        for x in 0..side {
            if x + 1 < side {
                arcs.push((id(x, y), id(x + 1, y), y % 6 == 0));
            }

            if y + 1 < side {
                arcs.push((id(x, y), id(x, y + 1), x % 6 == 0));
            }
        }
    }

    let mut distance = format!(
        "c generated road network\np sp {} {}\n",
        side * side,
        arcs.len() * 2
    );
    let mut time = format!(
        "c generated road network\np sp {} {}\n",
        side * side,
        arcs.len() * 2
    );

    for (k, &(a, b, fast)) in arcs.iter().enumerate() {
        let seed = u64::try_from(k).expect("small");
        let length = 50 + splitmix(seed * 2) % 100;
        let per_unit = if fast {
            3
        } else {
            6 + splitmix(seed * 2 + 1) % 7
        };

        for (from, to) in [(a, b), (b, a)] {
            writeln!(distance, "a {from} {to} {length}").expect("String");
            writeln!(time, "a {from} {to} {}", length * per_unit).expect("String");
        }
    }

    (distance, time)
}

/// Loading and searching a generated 30 x 30 road network with the
/// distance and time objectives, corner to corner.
fn bench_road(c: &mut Criterion) {
    let (distance, time) = road_network(30);
    let graph = RoadGraph::from_dimacs(&distance, &time).expect("generated files are valid");
    let goal = graph.node_count();
    let is_goal = |s: &u32| *s == goal;
    let mut group = c.benchmark_group("dimacs_generated_30x30");
    group.sample_size(20);

    group.bench_function("load", |b| {
        b.iter(|| RoadGraph::from_dimacs(black_box(&distance), black_box(&time)));
    });
    group.bench_function("boa_star", |b| {
        b.iter(|| boa_star(&graph, ZeroHeuristic, black_box(1), is_goal));
    });
    group.bench_function("boba_star", |b| {
        b.iter(|| boba_star(&graph, black_box(1), goal));
    });
    #[cfg(feature = "parallel")]
    group.bench_function("boba_star_parallel", |b| {
        b.iter(|| multiobjective_search::boba_star_parallel(&graph, black_box(1), goal));
    });
    group.bench_function("pp_a_star_1pct", |b| {
        let eps = [Epsilon::new(1, 100); 2];
        b.iter(|| pp_a_star(&graph, ZeroHeuristic, black_box(1), is_goal, eps));
    });
    group.finish();
}
