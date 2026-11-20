# multiobjective-search: design

Status: draft, 2026-11-20, updated after milestones 1-4. Covers the plan
up to 0.1.0 and the shape of later milestones.

## Scope

Exact multi-objective shortest-path search over implicit graphs: find one
path for every Pareto-optimal cost vector from a start state to a goal set.

In scope:

- BOA* (bi-objective A*), the first algorithm and the baseline for the rest.
- BOBA* (bidirectional bi-objective A*).
- NAMOA* (classical multi-objective A* with per-state frontiers), for any
  number of objectives.
- EMOA* (NAMOA*-style search with faster frontier checks), for three or
  more objectives.
- Pareto-frontier data structures that the algorithms share.
- NAMOA*dr (NAMOA* with dimensionality reduction, Pulido et al. 2015).
- `boa_star_approx`: BOA* with an additive tolerance on the second
  objective (cheap, one changed comparison).
- `brute_force`: an exhaustive public reference for tests.

Out of scope:

- Multiplicative approximate or anytime variants (A*pex, BOA*ε, PP-A*).
  They need multiplication on costs, which `Cost` does not have. They may
  come later behind their own entry points.
- Constrained shortest paths (resource-constrained, budgeted). Different
  problem, different output.
- Any planning-specific model. The crate does not depend on `sas-plus`,
  `heuristic-search` or other ecosystem crates. Adapters live in the
  consuming crate (for example miniplan).

## Constraints

- `no_std` with `alloc`. Per-state bookkeeping uses `BTreeMap`, so states
  need `Ord`, not `Hash`.
- No runtime dependencies. A `hashbrown` feature for `Hash` states is the
  likely first one, added only when a consumer needs it (see open
  questions).
- `#![forbid(unsafe_code)]`, pedantic clippy, docs on every item.
- Integer costs. Algorithms compare costs with `Ord` and need exact
  equality for tie handling; floats do not satisfy `Ord`. Costs must be
  non-negative; the crate does not check this.

## Core types and traits

### Cost

```rust
pub trait Cost: Copy + Ord + Add<Output = Self> + Default {}
```

Blanket-implemented. `Default` is the zero. Unsigned integers are the
intended types; signed integers satisfy the bound but negative edges break
every algorithm here. Overflow follows the integer type (panic in debug).

A cost vector is a plain array `[C; N]`. No newtype: arrays already have
`Copy`, `Ord` (lexicographic, which BOA* uses for its open list) and
`Default`. Vector addition is a free function `add`.

### Dominance

`Dominates` (published in 0.0.0) stays. It is implemented for `[C; N]`:
`a` dominates `b` when `a[i] <= b[i]` for every `i` and `a != b`. Weak
dominance (`<=` everywhere, equality allowed) is what pruning uses; it is a
free function `weakly_dominates`.

### Graph

```rust
pub trait Graph {
    type State: Clone + Ord;
    type Weight;
    fn successors(&self, state: &Self::State, out: &mut Vec<(Self::State, Self::Weight)>);
}
```

- Implicit graph: successors are generated on demand, no node count needed.
- `out` is cleared by the caller and reused across expansions, so no
  allocation per expansion and no iterator lifetime juggling (GATs would
  work too but make adapters harder to write).
- `Weight` is the cost vector, `[C; 2]` for BOA*, `[C; N]` for NAMOA*.
- Predecessors for BOBA* come from a second trait, `ReverseGraph`, with the
  same shape. Unidirectional algorithms never require it.

### Heuristic

```rust
pub trait Heuristic<S, W> {
    fn estimate(&self, state: &S) -> Option<W>;
}
```

- `None` means "cannot reach a goal": the state is pruned. This avoids
  infinity sentinels and the overflow they cause in `g + h`.
- Blanket impl for closures `Fn(&S) -> Option<W>`, plus `ZeroHeuristic`.
- BOA* requires a consistent heuristic per objective, zero on every goal.

### Labels

A label is an arena node: state, `g` vector, parent index. Paths are
rebuilt from parent links once a solution is found. The open list stores
`(f, node index)` so the heap stays small and `Copy`.

### Pareto frontiers

- `BiFront<C>`: non-dominated 2D points, sorted by the first objective
  ascending and so by the second strictly descending. Insertion is a binary
  search plus removal of the dominated run that follows it. Equal points are
  rejected (the frontier keeps one representative per cost vector).
  Membership and dominance tests are `O(log n)`.
- `Front<C, N>`: the general frontier; `BiFront<C>` is `Front<C, 2>`.
  Sorted lexicographically. Only points lexicographically no greater than
  a probe can weakly dominate it, so a query binary-searches that prefix;
  for `N <= 2` the last point of the prefix decides (`O(log n)`), for
  `N > 2` the prefix is scanned. Insertion removes dominated points (a
  contiguous run for `N = 2`, a `retain` otherwise).
- `TruncatedFront<C, N>` (private): dominance over objectives `1..N` only,
  for lexicographic searches with consistent heuristics. Three objectives
  map onto a 2D `Front`; other `N` use a `Front<C, N>` with the first
  component zeroed. EMOA*'s balanced tree is replaced by the sorted `Vec`:
  same query cost, `O(n)` insertion shift (shortcut; revisit if profiles
  show insertion dominating).

NAMOA*, NAMOA*dr and EMOA* share one private engine
(`frontier_search`) with two switches: the closed archive (full `Front`
or `TruncatedFront`) and whether a per-state open frontier is kept
(NAMOA*, NAMOA*dr) or labels are filtered lazily at pop time (EMOA*).

BOBA* reuses BOA*'s `Search` twice: forward on the graph, backward on a
`Reversed` adapter that swaps the objectives, so the backward direction
is BOA* in `(f2, f1)` order. Each direction's last solution bounds the
other's primary objective; the run stops when either direction is done.

BOA* does not need a per-state frontier: with lexicographic expansion order
and a consistent heuristic, the minimum second cost seen at a state
(`g2_min`) is enough to decide dominance in `O(1)`. The frontier type serves
the result set, the test reference and the later algorithms.

## API sketch

```rust
let out = boa_star(&graph, &heuristic, start, |s| *s == goal);
for sol in &out.solutions {
    // sol.cost: [C; 2], sol.path: Vec<State> from start to goal
}
// out.stats: expanded, generated
```

- One free function per algorithm: `boa_star`, `boa_star_approx`,
  `boba_star`, `namoa_star`, `namoa_star_dr`, `emoa_star`, `brute_force`. A builder or config struct only once there are
  options (time limits, solution limits, ε).
- Goal is a predicate, so a goal set works. BOBA* needs a single goal state
  and gets its own signature.
- Solutions come sorted lexicographically by cost; for two objectives
  that means the first strictly ascending and the second strictly
  descending.
- Output is `Output { solutions, stats }`, shared by all algorithms.

## Algorithms and references

- **BOA\*:** C. Hernández, W. Yeoh, J. A. Baier, H. Zhang, L. Suazo,
  S. Koenig, O. Salzman. "A Simple and Fast Bi-Objective Search
  Algorithm." ICAPS 2020. Journal version: "Simple and efficient
  bi-objective search algorithms via fast dominance checks." Artificial
  Intelligence, 2023.
- **NAMOA\*:** L. Mandow, J. L. Pérez de la Cruz. "A New Approach to
  Multiobjective A* Search." IJCAI 2005. Also "Multiobjective A* search
  with consistent heuristics." Journal of the ACM, 2010.
- **BOBA\*:** S. Ahmadi, G. Tack, D. Harabor, P. Kilby. "Bi-Objective
  Search with Bi-Directional A*." ESA 2021.
- **NAMOA\*dr:** F.-J. Pulido, L. Mandow, J. L. Pérez de la Cruz.
  "Dimensionality reduction in multiobjective shortest path search."
  Computers & Operations Research, 2015.
- **BOA\*ε (multiplicative, not implemented):** B. Goldin, O. Salzman.
  "Approximate bi-objective search by efficient dominance checks." SoCS
  2021.
- **EMOA\*:** Z. Ren, R. Zhan, S. Rathinam, M. Likhachev, H. Choset.
  "Enhanced Multi-Objective A* Using Balanced Binary Search Trees." SoCS
  2022.

BOBA* was checked against the arXiv version (2105.11888v2). The other
citations are from memory; check venue and year before 0.1.0.

## Testing strategy

- **Reference implementation:** brute-force enumeration of all simple paths
  on small graphs, filtered with a naive `O(n²)` non-dominated filter that
  lives in the tests and shares no code with the crate. Pareto-optimal cost
  sets are reached by simple paths when costs are non-negative, so this is
  exact.
- **Property tests (proptest):** random graphs with up to 7 nodes, edge
  costs in `0..=4` (small ranges force ties and zero-cost cycles, which is
  where `>` versus `>=` bugs hide), self-loops and parallel edges allowed.
  Compare the BOA* cost set with the reference, with the zero heuristic and
  with an exact per-objective backward-Dijkstra heuristic.
- **Path validity:** every returned path starts at the start, ends at a
  goal, uses only real edges and sums to its reported cost.
- **Output shape:** first objective strictly increasing, second strictly
  decreasing.
- **Fixed cases:** start is a goal, goal unreachable, a hand-computed graph.
- **Frontier:** proptest `BiFront` against the naive filter.
- Every algorithm is cross-checked against the brute-force reference for
  two, three and four objectives (BOA* and BOBA*: two), with the zero and
  the exact heuristic; classic NAMOA* also with an admissible but
  inconsistent one (exact values zeroed at random states). For two
  objectives all algorithms are also compared with BOA*.
- `boa_star_approx` is checked for its stated guarantee: every returned
  cost is Pareto-optimal and every Pareto-optimal cost is covered within
  the slack.
- Benchmarks (later): DIMACS road networks with distance and time, the
  standard bi-objective benchmark set used by the BOA* paper.

## Milestones

Milestones 1 to 4 are done (NAMOA*dr, `boa_star_approx` and
`brute_force` came along); milestone 5 has grid benches but no DIMACS
loader yet.

1. **Frontier and BOA\***: `Cost`, `Dominates`, `Graph`,
   `Heuristic`, `BiFront`, `boa_star`, brute-force cross-check tests.
2. **NAMOA\*:** general `Front<C, N>`, per-state open and closed frontiers,
   `namoa_star` for any `N`. Cross-check against BOA* for `N = 2`.
3. **BOBA\*:** `ReverseGraph`, bidirectional search with the two
   lexicographic directions, heuristics from single-objective backward
   searches.
4. **EMOA\*:** truncated 3D frontier (sorted `Vec`, not a tree), `emoa_star`.
5. **Benchmarks and 0.1.0:** DIMACS loader in a dev tool or example,
   criterion benches, release.

## Cross-crate needs

None for milestone 1. Later:

- Consumers (miniplan, a future `heuristic-search`) implement `Graph` and
  `Heuristic` for their models. If `heuristic-search` defines a
  single-objective successor trait with the same shape, both may move to
  `search-core` (only once both exist, per architecture.md).
- BOBA* and the exact test heuristic need single-objective backward
  Dijkstra. Keep a small private one here rather than depend on
  `heuristic-search`.

## Open questions

- Should states require `Ord` (BTreeMap, no dependency) or `Hash` (needs
  `hashbrown` in `no_std`)? Draft uses `Ord`.
- Should `Graph` stay a callback-into-`Vec` interface, or return an
  iterator via a GAT?
- Is a goal predicate enough, or do consumers need a goal state for
  heuristics and BOBA*?
- Should the crate guard against negative costs (an `Unsigned` marker
  trait) instead of documenting the precondition?
- Should `Cost` gain multiplication (or a scaling trait) so that a
  multiplicative `(1 + ε)` approximation fits? Draft: no, additive only.
- Should BOBA* accept user heuristics, or bounded Dijkstra as in the
  paper? Draft: full Dijkstra, computed internally.
- Should `brute_force` stay public or move to the tests? Draft: public.
