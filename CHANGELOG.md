# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/sunsided/multiobjective-search/compare/v0.1.0...v0.1.1) - 2026-11-22

### Changed

- Link the crate to its page on the planning and search site

## [0.1.0](https://github.com/sunsided/multiobjective-search/releases/tag/v0.1.0) - 2026-11-21

### Added

- Add the strict lint table, clippy.toml and rustfmt.toml
- Add CONTRIBUTING.md
- Add README badges, usage and license sections
- Add fuzz targets and dominance property tests
- Add REUSE licensing metadata
- Add cargo-deny, Dependabot and typos, and track Cargo.lock
- Add the Codecov configuration
- Add the Taskfile and CI workflows
- Add TPA, the exact anytime box-splitting method, and hypervolume
- Implement A-A*pex as published, with its four variants
- Add multi-objective minimum spanning trees
- Add exact objective-correlation detection
- Add dynamic multi-objective search with incremental repair
- Add label correcting, ideal-point and differential heuristics
- Add generators for the standardized MOS benchmark suite
- Add search over objective-aggregation functions
- Add parallel MOA* with multi-level upper bounds
- Add full disjoint-pair fronts, maximally disjoint pairs and ratio paths
- Add edge-identity graphs with disjoint and dissimilar paths
- Add an FPTAS for the constrained shortest path
- Add an interactive reference-point search
- Add the two-phase method for bi-objective shortest paths
- Add several bottlenecks, cost/capacity ratio and quickest paths
- Add minsum-maxmin, constrained and ranking paths from the survey
- Add SIP-MOSP, asynchronous parallel exact search, and IndexedFront
- Add Iterative-PP-A*, the anytime preprint's PP-A* baseline
- Add A-BOA*eps, anytime approximate bi-objective search
- Add multi-valued heuristics with NAMOA*dr-mvh and L-NAMOA*dr-mvh
- Add A*pex for any number of objectives
- Add multi-threaded BOBA* behind a parallel feature
- Add a 9th DIMACS Challenge road-network loader and bench
- Add BOA*eps and PP-A* with rational tolerances
- Add a balanced-tree Pareto frontier and emoa_star_tree
- Add hand-computed N-objective cases and widen approx coverage
- Add criterion benchmarks on grid graphs
- Add a public brute-force reference
- Add BOBA*: bidirectional bi-objective A*
- Add boa_star_approx: BOA* with an additive tolerance
- Add NAMOA*, NAMOA*dr and EMOA* for any number of objectives
- Add BOA* for two objectives
- Add cost vectors, dominance and a bi-objective Pareto frontier
- Add design document

### Changed

- Describe the released scope and audience in the README
- Stop tracking Cargo.lock
- Record search-core from the crates.io registry in the locks
- Depend on search-core from crates.io
- Split long tests and benchmarks into helpers
- Split Suurballe and label correcting into documented steps
- Render mermaid on docs.rs and doctest the README
- Require search-core 0.3
- Sum Dinkelbach weights in i128 when they fit, BigInt otherwise
- Use search-core BigInt for the Dinkelbach weights
- Require search-core 0.2
- Require search-core 0.1
- Accept search-core 0.0 and 0.1 during the 0.1.0 release
- Generalise AnytimeFront to N objectives and add anytime A*pex
- Deduplicate equal costs in the bottleneck test oracles
- Encode knapsacks and the minimum-label objective as paths
- Make the SIP-MOSP thresholds configurable
- Build multi-valued heuristics by backward A*pex
- Check NAMOA* against the JACM 2010 paper
- use the paper's greedy merge rule by default
- Document the A*pex check against the paper
- follow the paper's merge and solution rules
- early solution update and heuristic tuning
- Reject DIMACS files with different arc counts, soften a BOBA* panic
- Document the check of every algorithm against its paper
- cost-bounded preliminary searches and user heuristics
- Adopt search-core for Cost and Heuristic
- Update README, crate docs and design for the new algorithms
- Generalize the Pareto frontier to N objectives
- Update README status
- Cross-check BOA* against brute-force enumeration
- Reserve the crate name with a 0.0.0 placeholder

### Fixed

- Fix the DIMACS loader reserving the declared arc count
- Fix the MVH consistency definition and tighten the new tests
- Correct the A*pex check notes
- Correct paper citations, document shortcuts, raise MSRV to 1.86

### Removed

- Drop the sibling checkout note from the README
- Drop the u64 bounds of quickest_path and supported_disjoint_pairs
