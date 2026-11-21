# multiobjective-search

[![crates.io](https://img.shields.io/crates/v/multiobjective-search.svg)](https://crates.io/crates/multiobjective-search)
[![docs.rs](https://img.shields.io/docsrs/multiobjective-search)](https://docs.rs/multiobjective-search)
[![CI](https://github.com/sunsided/multiobjective-search/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/multiobjective-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/multiobjective-search/graph/badge.svg)](https://codecov.io/gh/sunsided/multiobjective-search)
[![MSRV](https://img.shields.io/crates/msrv/multiobjective-search)](https://crates.io/crates/multiobjective-search)
[![License](https://img.shields.io/crates/l/multiobjective-search)](https://github.com/sunsided/multiobjective-search#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Multi-objective shortest paths: BOA*, BOBA*, EMOA* and NAMOA* over Pareto frontiers.

Given a graph whose edges carry cost vectors, the algorithms find one path for every
Pareto-optimal cost vector from a start state to a goal. The crate is `no_std` with `alloc`,
and builds on the cost and heuristic traits of
[`search-core`](https://github.com/sunsided/search-core).

## Installation

```sh
cargo add multiobjective-search
```

## Usage

Two routes from node 1 to node 3, one shorter and one faster; BOA\* returns both:

```rust
use multiobjective_search::{RoadGraph, ZeroHeuristic, boa_star};

let distance = "p sp 3 3\na 1 2 10\na 2 3 10\na 1 3 25\n";
let time = "p sp 3 3\na 1 2 5\na 2 3 5\na 1 3 6\n";
let graph = RoadGraph::from_dimacs(distance, time)?;

let out = boa_star(&graph, ZeroHeuristic, 1, |s| *s == 3);
let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
assert_eq!(costs, [[20, 10], [25, 6]]);
# Ok::<(), multiobjective_search::DimacsError>(())
```

## Features

- `std`: links `std`; no API of its own, required by `parallel`.
- `parallel`: multi-threaded BOBA\* (`boba_star_parallel`, `boba_star_with_parallel`),
  SIP-MOSP (`sip_mosp`) and parallel MOA\* (`parallel_moa_star`) on `std::thread::scope`.

## MSRV

The minimum supported Rust version is 1.86 (`rust-version` in `Cargo.toml`). It is a signal,
not a contract: raising it is a minor release, never a patch release.

## Scope and status

Implemented, cross-checked against each other and against a brute-force
reference on random graphs with two to four objectives:

- [x] BOA\*: bi-objective A* (`boa_star`)
- [x] BOA\* with an additive tolerance on the second objective (`boa_star_approx`)
- [x] BOA\*ε: BOA* with a multiplicative tolerance `1 + ε` on the second
      objective (`boa_star_eps`), rational `ε = num / den` (`Epsilon`)
- [x] A-BOA\*ε: anytime approximate bi-objective search (`boa_star_anytime`),
      with weighted-sum pruning (`boa_star_anytime_ws`) and the no-reuse
      baseline (`boa_star_anytime_basic`); reports each improved front and its
      exact factor (`ApproxFactor`) to a callback and stops on a budget
- [x] Iterative-PP-A\*: the anytime baseline over PP-A\* (`pp_a_star_iterative`)
- [x] A-A\*pex, anytime A\*pex for any number of objectives (`apex_star_anytime`,
      `AnytimeApex` variants reuse, reuse-enh, restart, hybrid), after Zhang et al.
      2024; `AnytimeFront` takes the objective count
- [x] PP-A\*: path pairs, tolerances on both objectives (`pp_a_star`); one-sided
      bounded test and `br` returned, so the tolerances never compound
- [x] A\*pex: approximate search for any number of objectives over
      apex-path pairs (`apex_star`), tolerance `1 + ε_i` per objective;
      checked against the paper's Algorithm 2, with its G (default) and L merge methods (`MergeRule`)
- [x] BOBA\*: bidirectional bi-objective A* (`boba_star`) with the paper's four
      cost-bounded preliminary searches (run one after another), or with
      caller-supplied heuristics (`boba_star_with`); with the paper's early
      solution update and heuristic tuning
- [x] Multi-threaded BOBA\* behind the `parallel` feature (`boba_star_parallel`,
      `boba_star_with_parallel`; `std::thread::scope`, no dependencies),
      cross-checked against the sequential front
- [x] EMOA\*: multi-objective A* with truncated dominance checks (`emoa_star`),
      sorted `Vec` instead of a balanced tree
- [x] NAMOA\*: classic multi-objective A* for admissible heuristics (`namoa_star`)
- [x] NAMOA\*dr: NAMOA* with dimensionality reduction (`namoa_star_dr`)
- [x] Multi-valued heuristics (`MvHeuristic`, `SingleValued`, `mv_heuristic_fn`)
      with NAMOA\*dr-mvh (`namoa_star_dr_mvh`, consistent MVHs) and the lazy
      L-NAMOA\*dr-mvh (`namoa_star_dr_mvh_lazy`, admissible MVHs), Wolff,
      Felner, Salzman 2026; MVHs built by backward A*pex (`apex_mvh`, `MvhTable`)
      and made consistent by a Bellman-Ford fix (`MvhTable::make_consistent`)
- [x] EMOA\* with the paper's balanced-tree frontiers (`emoa_star_tree`,
      `TreeFront`): a treap with per-subtree bounding boxes, cross-checked
      against the `Vec` frontier; slower than the `Vec` on the crate's
      benchmarks, so `emoa_star` keeps the `Vec`
- [x] SIP-MOSP: asynchronous parallel exact search with versioned frontier
      snapshots and indexed dominance pruning (`sip_mosp`, feature
      `parallel`), and its indexed frontier (`IndexedFront`, `FrontIndex`)
- [x] From the Craveirinha et al. 2024 survey: minsum–maxmin paths
      (`minsum_maxmin`), constrained shortest paths (`constrained_shortest_path`)
      and the classic ranking method (`ranking`); several bottlenecks
      (`minsum_maxmin_multi`), the cost/capacity ratio and quickest path,
      the two-phase method (`two_phase`, `supported_solutions`), knapsack and
      label-count encodings (`KnapsackGraph`, `LabelCount`), an interactive
      reference-point search (`ReferencePoint`), an FPTAS for the constrained
      path (`fptas_constrained`), and on graphs with edge identities
      (`EdgeGraph`) disjoint pairs (`disjoint_pair`), supported bicriteria
      pairs and dissimilar paths; the coverage table is in `docs/design.md`
- [x] Parallel MOA\* (`parallel_moa_star`, feature `parallel`): one MOA\* per
      cyclic objective order with shared multi-level upper bounds (Ahmadi et al.)
- [x] Objective-aggregation search (`aggregated_search`): solution objectives as
      monotone aggregations of hidden objectives (Peer et al.)
- [x] Benchmark-suite generators after Peer et al. 2026 (`random_grid`, `netmaker`,
      `guards_grid`, `road_extended`, `BenchmarkGraph`) and a `suite` bench group
- [x] From the Salzman et al. survey: multi-objective label correcting
      (`label_correcting`), the ideal-point heuristic (`IdealPoint`) and
      differential heuristics (`DifferentialHeuristic`)
- [x] Dynamic search (`DynamicGraph`, `DynamicSearch`): incremental repair after
      edge additions, cost changes and removals
- [x] Multi-objective minimum spanning trees (`pareto_spanning_trees`,
      `two_phase_spanning_trees`) and exact objective correlation
      (`objective_correlation`)
- [x] TPA (`tpa`, `tpa_paths`): exact anytime box splitting with
      augmented-Tchebycheff subproblems over any solver, and the hypervolume
      indicator (`hypervolume`), after Domínguez-Ríos et al.
- [x] Brute-force reference over simple paths (`brute_force`)
- [x] Pareto frontier for any number of objectives, `O(log n)` queries for two
      (`Front`, `BiFront`)
- [x] 9th DIMACS Challenge loader (`RoadGraph`: distance and time `.gr`
      files zipped into `[distance, time]` costs, `.co` coordinates), `no_std`
- [x] Criterion benchmarks on grid graphs and on a generated road network in
      DIMACS format (generated, since the challenge data's licence is unclear)

Open:

- [ ] Benchmarks on the real DIMACS road networks (download them and load
      with `RoadGraph::from_dimacs`; not bundled)

## Breaking changes

Adopting `search-core` (unreleased, after 0.0.0) changed the public API:

- `Cost` is now `search_core::Cost`, re-exported from the crate root. It has
  no blanket impl: it is implemented for the unsigned integers and
  `search_core::Decimal`. A user-defined cost type must implement `Cost`
  (with `ZERO`, `saturating_add` and `saturating_times`); signed integers no
  longer qualify.
- `Heuristic` is now `search_core::Heuristic`, re-exported, with
  `estimate(&mut self, ..)`. Closures are no longer heuristics by
  themselves: wrap them in `heuristic_fn` (also re-exported).
- `boa_star`, `boa_star_approx`, `emoa_star`, `namoa_star` and
  `namoa_star_dr` take the heuristic by value instead of `&H`. Pass
  `ZeroHeuristic` or `heuristic_fn(..)` directly, or `&mut h` to keep `h`.

`Dominates`, `Graph`, `ReverseGraph`, `ZeroHeuristic`, `Front`, `BiFront`,
`Output` and `Stats` are unchanged.

## Status

Pre-release. Version 0.0.0 reserves the name; the API is a draft and will change.
See [`docs/design.md`](https://github.com/sunsided/multiobjective-search/blob/main/docs/design.md) for the design.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/multiobjective-search/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/multiobjective-search/blob/main/LICENSE-MIT))

at your option. The license is `MIT OR Apache-2.0`; `REUSE.toml` records it for every file.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/multiobjective-search/blob/main/CONTRIBUTING.md). Local development expects
[`search-core`](https://github.com/sunsided/search-core) checked out next to this repository.
