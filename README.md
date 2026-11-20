# multiobjective-search

Multi-objective shortest paths: BOA*, BOBA*, EMOA* and NAMOA* over Pareto frontiers.

## Scope and status

Implemented, cross-checked against each other and against a brute-force
reference on random graphs with two to four objectives:

- [x] BOA\*: bi-objective A* (`boa_star`)
- [x] BOA\* with an additive tolerance on the second objective (`boa_star_approx`)
- [x] BOBA\*: bidirectional bi-objective A* (`boba_star`), without the paper's
      early solution update and heuristic tuning
- [x] EMOA\*: multi-objective A* with truncated dominance checks (`emoa_star`),
      sorted `Vec` instead of a balanced tree
- [x] NAMOA\*: classic multi-objective A* for admissible heuristics (`namoa_star`)
- [x] NAMOA\*dr: NAMOA* with dimensionality reduction (`namoa_star_dr`)
- [x] Brute-force reference over simple paths (`brute_force`)
- [x] Pareto frontier for any number of objectives, `O(log n)` queries for two
      (`Front`, `BiFront`)
- [x] Criterion benchmarks on grid graphs

Open:

- [ ] Multiplicative `(1 + ε)` approximation (BOA*ε, PP-A*)
- [ ] DIMACS road-network benchmarks

## Status

Pre-release. Version 0.0.0 reserves the name; the API is a draft and will change.
See `docs/design.md` for the design.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
