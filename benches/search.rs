//! Benchmarks for every search on pseudo-random grid graphs, and for the
//! Pareto frontier that they share.

use std::hint::black_box;

use criterion::Criterion;
use multiobjective_search::{
    Front, Graph, ReverseGraph, ZeroHeuristic, boa_star, boba_star, emoa_star, namoa_star,
    namoa_star_dr,
};

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_two_objectives(&mut criterion);
    bench_three_objectives(&mut criterion);
    bench_front(&mut criterion);
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
