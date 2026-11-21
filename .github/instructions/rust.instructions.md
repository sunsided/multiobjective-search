---
applyTo: "**/*.rs"
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function. No lumped code.
- One level of abstraction per function. Search loops are short drivers that call documented
  step functions (expand, prune, merge). If a block needs a "what" comment, extract it into a
  function named after that comment.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- No `unsafe`: the crate is `#![forbid(unsafe_code)]`.
- The crate is `#![no_std]` with `alloc`. `std` items sit behind the `std` feature, threads
  behind `parallel`. Flag any new `[dependencies]` entry, any `std` use outside those
  features, and any feature-gated item without its `#[cfg]` on the re-export in `lib.rs`.
- Cost types come from `search-core` (`Cost`, `ExactArith`). Scaled comparisons (`Epsilon`,
  weighted sums) use the saturating `Cost` methods or the exact `BigUint`/`BigInt` types, never
  division or floats.
- Every new search is cross-checked against `brute_force` (or another exact algorithm) by
  property tests under `tests/it/`, and parsers get a fuzz target under `fuzz/`.
- Public API changes go with a matching update to the README "Algorithms" list and the
  crate-level table in `src/lib.rs`. Sibling crates (`mapf-search`, `connector-routing`) depend
  on this one, so flag breaking changes.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
