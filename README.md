# multiobjective-search

Multi-objective shortest paths: BOA*, BOBA*, EMOA* and NAMOA* over Pareto frontiers.

## Scope and status

Implemented, cross-checked against each other and against a brute-force
reference on random graphs with two to four objectives:

- [x] BOA\*: bi-objective A* (`boa_star`)
- [x] BOA\* with an additive tolerance on the second objective (`boa_star_approx`)
- [x] BOA\*ε: BOA* with a multiplicative tolerance `1 + ε` on the second
      objective (`boa_star_eps`), rational `ε = num / den` (`Epsilon`)
- [x] PP-A\*: path pairs, tolerances on both objectives (`pp_a_star`); one-sided
      bounded test and `br` returned, so the tolerances never compound
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
- [x] EMOA\* with the paper's balanced-tree frontiers (`emoa_star_tree`,
      `TreeFront`): a treap with per-subtree bounding boxes, cross-checked
      against the `Vec` frontier; slower than the `Vec` on the crate's
      benchmarks, so `emoa_star` keeps the `Vec`
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
See `docs/design.md` for the design.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
