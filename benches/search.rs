//! Benchmarks for every search on pseudo-random grid graphs, and for the
//! Pareto frontier that they share.

use std::hint::black_box;

use criterion::Criterion;
use multiobjective_search::{
    Front, Graph, ReverseGraph, TreeFront, ZeroHeuristic, boa_star, boba_star, emoa_star,
    emoa_star_tree, namoa_star, namoa_star_dr,
};

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_two_objectives(&mut criterion);
    bench_three_objectives(&mut criterion);
    bench_four_objectives(&mut criterion);
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
