# multiobjective-search

[![crates.io](https://img.shields.io/crates/v/multiobjective-search.svg)](https://crates.io/crates/multiobjective-search)
[![docs.rs](https://img.shields.io/docsrs/multiobjective-search)](https://docs.rs/multiobjective-search)
[![CI](https://github.com/sunsided/multiobjective-search/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/multiobjective-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/multiobjective-search/graph/badge.svg)](https://codecov.io/gh/sunsided/multiobjective-search)
[![MSRV](https://img.shields.io/crates/msrv/multiobjective-search)](https://crates.io/crates/multiobjective-search)
[![License](https://img.shields.io/crates/l/multiobjective-search)](https://github.com/sunsided/multiobjective-search#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Multi-objective shortest paths: BOA\*, BOBA\*, EMOA\*, NAMOA\* and their exact,
approximate, anytime and parallel relatives over Pareto frontiers.

Given a graph whose edges carry cost vectors (distance and travel time, say, or length,
risk and energy), the algorithms find one path for every Pareto-optimal cost vector from a
start state to a goal, instead of collapsing the objectives into one weighted sum up front.
Every exact search is cross-checked against the others and against a brute-force reference
on random graphs with two to four objectives. The crate is `no_std` with `alloc`, has no
dependencies besides [`search-core`](https://github.com/sunsided/search-core), whose cost
and heuristic traits it uses, and ships a loader for the 9th DIMACS Challenge road networks
along with the benchmark-suite generators of Peer et al.

## Who this is for

- Route planners that have to show the trade-off between two or more costs (fastest,
  shortest, cheapest, safest) rather than one "best" route, e.g. on road networks or grids.
- Robotics and logistics people who need a bounded-suboptimal front quickly: the
  ε-approximate (BOA\*ε, PP-A\*, A\*pex) and anytime searches (A-BOA\*ε, A-A\*pex, TPA)
  trade front quality for time with a guaranteed factor.
- Researchers comparing multi-objective search algorithms, who get the classic and recent
  algorithms behind one `Graph` interface, the standardized benchmark generators and a
  brute-force reference to test against.
- For a single objective, [heuristic-search](https://github.com/sunsided/heuristic-search)
  is the simpler and faster choice; for several agents that must not collide,
  [mapf-search](https://github.com/sunsided/mapf-search) (its MO-CBS uses this crate as the
  low level).

## Installation

```sh
cargo add multiobjective-search
```

Or in `Cargo.toml`:

```toml
[dependencies]
multiobjective-search = "0.1"
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

## Algorithms

Implemented, cross-checked against each other and against a brute-force
reference on random graphs with two to four objectives:

- BOA\*: bi-objective A* (`boa_star`)
- BOA\* with an additive tolerance on the second objective (`boa_star_approx`)
- BOA\*ε: BOA* with a multiplicative tolerance `1 + ε` on the second
  objective (`boa_star_eps`), rational `ε = num / den` (`Epsilon`)
- A-BOA\*ε: anytime approximate bi-objective search (`boa_star_anytime`),
  with weighted-sum pruning (`boa_star_anytime_ws`) and the no-reuse
  baseline (`boa_star_anytime_basic`); reports each improved front and its
  exact factor (`ApproxFactor`) to a callback and stops on a budget
- Iterative-PP-A\*: the anytime baseline over PP-A\* (`pp_a_star_iterative`)
- A-A\*pex, anytime A\*pex for any number of objectives (`apex_star_anytime`,
  `AnytimeApex` variants reuse, reuse-enh, restart, hybrid), after Zhang et al.
  2024; `AnytimeFront` takes the objective count
- PP-A\*: path pairs, tolerances on both objectives (`pp_a_star`); one-sided
  bounded test and `br` returned, so the tolerances never compound
- A\*pex: approximate search for any number of objectives over
  apex-path pairs (`apex_star`), tolerance `1 + ε_i` per objective;
  checked against the paper's Algorithm 2, with its G (default) and L merge methods (`MergeRule`)
- BOBA\*: bidirectional bi-objective A* (`boba_star`) with the paper's four
  cost-bounded preliminary searches (run one after another), or with
  caller-supplied heuristics (`boba_star_with`); with the paper's early
  solution update and heuristic tuning
- Multi-threaded BOBA\* behind the `parallel` feature (`boba_star_parallel`,
  `boba_star_with_parallel`; `std::thread::scope`, no dependencies),
  cross-checked against the sequential front
- EMOA\*: multi-objective A* with truncated dominance checks (`emoa_star`),
  sorted `Vec` instead of a balanced tree
- NAMOA\*: classic multi-objective A* for admissible heuristics (`namoa_star`)
- NAMOA\*dr: NAMOA* with dimensionality reduction (`namoa_star_dr`)
- Multi-valued heuristics (`MvHeuristic`, `SingleValued`, `mv_heuristic_fn`)
  with NAMOA\*dr-mvh (`namoa_star_dr_mvh`, consistent MVHs) and the lazy
  L-NAMOA\*dr-mvh (`namoa_star_dr_mvh_lazy`, admissible MVHs), Wolff,
  Felner, Salzman 2026; MVHs built by backward A*pex (`apex_mvh`, `MvhTable`)
  and made consistent by a Bellman-Ford fix (`MvhTable::make_consistent`)
- EMOA\* with the paper's balanced-tree frontiers (`emoa_star_tree`,
  `TreeFront`): a treap with per-subtree bounding boxes, cross-checked
  against the `Vec` frontier; slower than the `Vec` on the crate's
  benchmarks, so `emoa_star` keeps the `Vec`
- SIP-MOSP: asynchronous parallel exact search with versioned frontier
  snapshots and indexed dominance pruning (`sip_mosp`, feature
  `parallel`), and its indexed frontier (`IndexedFront`, `FrontIndex`)
- From the Craveirinha et al. 2024 survey: minsum–maxmin paths
  (`minsum_maxmin`), constrained shortest paths (`constrained_shortest_path`)
  and the classic ranking method (`ranking`); several bottlenecks
  (`minsum_maxmin_multi`), the cost/capacity ratio and quickest path,
  the two-phase method (`two_phase`, `supported_solutions`), knapsack and
  label-count encodings (`KnapsackGraph`, `LabelCount`), an interactive
  reference-point search (`ReferencePoint`), an FPTAS for the constrained
  path (`fptas_constrained`), and on graphs with edge identities
  (`EdgeGraph`) disjoint pairs (`disjoint_pair`), supported bicriteria
  pairs and dissimilar paths; the coverage table is in `docs/design.md`
- Parallel MOA\* (`parallel_moa_star`, feature `parallel`): one MOA\* per
  cyclic objective order with shared multi-level upper bounds (Ahmadi et al.)
- Objective-aggregation search (`aggregated_search`): solution objectives as
  monotone aggregations of hidden objectives (Peer et al.)
- Benchmark-suite generators after Peer et al. 2026 (`random_grid`, `netmaker`,
  `guards_grid`, `road_extended`, `BenchmarkGraph`) and a `suite` bench group
- From the Salzman et al. survey: multi-objective label correcting
  (`label_correcting`), the ideal-point heuristic (`IdealPoint`) and
  differential heuristics (`DifferentialHeuristic`)
- Dynamic search (`DynamicGraph`, `DynamicSearch`): incremental repair after
  edge additions, cost changes and removals
- Multi-objective minimum spanning trees (`pareto_spanning_trees`,
  `two_phase_spanning_trees`) and exact objective correlation
  (`objective_correlation`)
- TPA (`tpa`, `tpa_paths`): exact anytime box splitting with
  augmented-Tchebycheff subproblems over any solver, and the hypervolume
  indicator (`hypervolume`), after Domínguez-Ríos et al.
- Brute-force reference over simple paths (`brute_force`)
- Pareto frontier for any number of objectives, `O(log n)` queries for two
  (`Front`, `BiFront`)
- 9th DIMACS Challenge loader (`RoadGraph`: distance and time `.gr`
  files zipped into `[distance, time]` costs, `.co` coordinates), `no_std`
- Criterion benchmarks on grid graphs and on a generated road network in
  DIMACS format (generated, since the challenge data's licence is unclear)

Not yet implemented: benchmarks on the real DIMACS road networks. They can be downloaded
and loaded with `RoadGraph::from_dimacs`, but are not bundled.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/multiobjective-search/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/multiobjective-search/blob/main/LICENSE-MIT))

at your option. The license is `MIT OR Apache-2.0`; `REUSE.toml` records it for every file.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/multiobjective-search/blob/main/CONTRIBUTING.md).
