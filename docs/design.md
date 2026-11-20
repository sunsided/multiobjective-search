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
- `boa_star_eps` (BOA*ε) and `pp_a_star` (PP-A*, Goldin and Salzman 2021):
  multiplicative tolerances `1 + ε` with a rational `Epsilon = num / den`,
  tested as `den · x <= (den + num) · y` through `Cost::saturating_times`.
  A saturated product falls back to the exact test, which only weakens the
  pruning. PP-A* deviates from the paper in two ways that keep the
  guarantee: its bounded test is one-sided (first objective) and it returns
  the `br` path, so the second tolerance is spent only by the solution
  pruning and approximations never compound; and a goal predicate replaces
  the single goal.
- `brute_force`: an exhaustive public reference for tests.

Out of scope:

- Anytime variants.
- Constrained shortest paths (resource-constrained, budgeted). Different
  problem, different output.
- Any planning-specific model. The crate depends only on `search-core`,
  not on `sas-plus`, `heuristic-search` or other ecosystem crates. Adapters live in the
  consuming crate (for example miniplan).

## Constraints

- `no_std` with `alloc`. Per-state bookkeeping uses `BTreeMap`, so states
  need `Ord`, not `Hash`.
- One runtime dependency, `search-core`, for `Cost` and `Heuristic`. A
  `hashbrown` feature for `Hash` states would be the next, added only when
  a consumer needs it (see open questions).
- `#![forbid(unsafe_code)]`, pedantic clippy, docs on every item.
- Integer costs. Algorithms compare costs with `Ord` and need exact
  equality for tie handling; floats do not satisfy `Ord`. Costs must be
  non-negative; the crate does not check this.

## Core types and traits

### Cost

`Cost` is `search_core::Cost`, re-exported from the crate root:
`Copy + Ord + Add<Output = Self>` with a `ZERO` constant, `saturating_add`
and `saturating_times`. It is implemented explicitly for the unsigned
integers and `search_core::Decimal` (plus `NonNegF64` behind search-core's
`float` feature), so negative costs are unrepresentable. A user-defined
cost type must implement `Cost` itself. Overflow follows the integer type
(panic in debug).

A cost vector is a plain array `[C; N]`. No newtype: arrays already have
`Copy` and `Ord` (lexicographic, which BOA* uses for its open list), and
`[C::ZERO; N]` is the zero. Vector addition is a free function `add`.
`Dominates`, `weakly_dominates` and `add` stay local: only this crate uses
Pareto dominance.

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

`Heuristic` is `search_core::Heuristic`, re-exported from the crate root
together with `heuristic_fn` and `HeuristicFn`:

```rust
pub trait Heuristic<S, C = u32> {
    fn estimate(&mut self, state: &S) -> Option<C>;
}
```

- `None` means "cannot reach a goal": the state is pruned. This avoids
  infinity sentinels and the overflow they cause in `g + h`.
- `&mut self` lets a heuristic keep a workspace. Algorithms take the
  heuristic by value and own it for the run; pass `&mut h` to keep it.
- Closures go through `heuristic_fn`; there is no blanket impl for `Fn`.
- `ZeroHeuristic` stays local: `search_core::Zero` covers scalar costs
  only, not cost vectors `[C; N]`.
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
  same query cost for three objectives (the paper's TOA* case), `O(n)`
  insertion shift, and a prefix scan instead of a pruned tree traversal
  for four or more.
- `TreeFront<C, N>` (public): the same frontier in a treap keyed
  lexicographically, each node holding its subtree's component-wise min
  and max. Queries skip subtrees whose min does not weakly dominate the
  probe (and, as in the paper's `Check`, right subtrees of keys at or
  above it); insertion removes dominated points in one pruned pass.
  `emoa_star_tree` runs EMOA* on it (private `TruncatedTree`). Measured
  slower than the `Vec` on four-objective grids and on near-antichain
  fronts of 1000 and 16000 points, where nearly every bounding box admits
  the probe, so `emoa_star` keeps the `Vec`. The paper's own complexity
  table gives no better worst case than a linear scan either.

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
  `boa_star_eps`, `pp_a_star`, `boba_star`, `boba_star_with`,
  `namoa_star`, `namoa_star_dr`, `emoa_star`, `emoa_star_tree`,
  `brute_force`. A builder or config struct only once there are
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
  Intelligence 314, 2023, 103807.
- **NAMOA\*:** L. Mandow, J. L. Pérez de la Cruz. "A New Approach to
  Multiobjective A* Search." IJCAI 2005. Also "Multiobjective A* search
  with consistent heuristics." Journal of the ACM 57(5), Article 27,
  2010.
- **BOBA\*:** S. Ahmadi, G. Tack, D. Harabor, P. Kilby. "Bi-Objective
  Search with Bi-Directional A*." ESA 2021.
- **NAMOA\*dr:** F.-J. Pulido, L. Mandow, J. L. Pérez de la Cruz.
  "Dimensionality reduction in multiobjective shortest path search."
  Computers & Operations Research, 2015.
- **BOA\*ε and PP-A\*:** B. Goldin,
  O. Salzman. "Approximate bi-criteria search by efficient representation
  of subsets of the Pareto-optimal frontier." ICAPS 2021 (arXiv
  2006.10302).
- **A\*pex:** H. Zhang, O. Salzman, T. K. S. Kumar, A. Felner, C.
  Hernández, S. Koenig. "A*pex: Efficient Approximate Multi-Objective
  Search on Graphs." ICAPS 2022. Checked against Algorithm 2 of the
  paper; see the table below.
- **NAMOA\*dr-mvh and L-NAMOA\*dr-mvh, multi-valued heuristics:** M.
  Wolff, A. Felner, O. Salzman. "Bridging Multi-Valued Heuristics and
  Dimensionality Reduction in Multi-Objective Search." arXiv 2606.20644,
  2026. Its Algorithm 1 also restates NAMOA*dr, used as a second source.
- **A-BOA\*ε:** "Anytime Approximate Bi-Objective Search" (anonymous
  preprint, crl.cs.technion.ac.il). Checked against its Algorithms 1 to 3
  and Section 5; see the table below.
- **SIP-MOSP:** X. Xu, N. Zhang, L. Qian, X. Liu, J. Wu, H. Tang.
  "Asynchronous Parallel Search for Exact Multi-Objective Shortest Paths
  with Versioned Frontier Snapshots and Indexed Dominance Pruning." arXiv
  2609.11944, 2026.
- **Parallel MOA\*:** S. Ahmadi, N. R. Sturtevant, A. Raith, D. Harabor,
  M. Jalili. "Parallelizing Multi-objective A* Search." arXiv 2503.10075.
- **EMOA\*:** Z. Ren, R. Zhan, S. Rathinam, M. Likhachev, H. Choset.
  "Enhanced Multi-Objective A* Using Balanced Binary Search Trees." SoCS
  2022.

### Verification against the papers

The algorithms were first written from memory, then checked line by line
against the papers' pseudocode: BOA* against the AIJ 2023 version
(Algorithm 3), NAMOA* against IJCAI 2005 and the JACM 2010 paper
(Table I, Sections 4 and 5), NAMOA*dr against the thesis (Tables 4.4 and 4.5,
Definition 4.5, eq. 4.9), A*pex against the ICAPS 2022 paper
(Algorithm 2), EMOA*
against arXiv 2202.08992 (Algorithms 1 to 3), PP-A* and BOA*ε against
arXiv 2006.10302v2, and BOBA* against arXiv 2105.11888v2 (Algorithms 1
and 2). Pulido et al. 2015 (C&OR) was not reachable (ScienceDirect
answers automated requests with a Cloudflare block); NAMOA*dr is checked
against the thesis and against the restatement in Wolff et al. 2026
(below). Only A*pex needed
fixes (below). The remaining differences elsewhere are deliberate and
leave the returned cost vectors unchanged:

| Algorithm | Paper | Here | Why the cost vectors do not change |
|---|---|---|---|
| all | equal cost vectors add a back pointer (all paths kept) | one path per cost vector | equal vectors are pruned by weak dominance; the output promises one path per vector |
| BOA* | single goal state, `g2min(s_goal)` | goal predicate, `g2` of the last solution | the last solution has the smallest `g2` and an `f1` no larger than any later label's |
| BOA* | tie-breaking unspecified | `(f1, f2)`, then arena index | determinism only |
| NAMOA* (JACM 2010, Table I step 4) | strict dominance `≺` for filtering by `COSTS` | weak dominance | verified; a label whose `f` equals a solution cost can only reach that cost or worse |
| NAMOA* (JACM 2010, step 5(b) i; IJCAI 2005) | a new non-dominated `g_m` removes the vectors it dominates from `G_op(m)` and `G_cl(m)` | removed from the open frontier at once; `G_cl` keeps them until a closed vector supersedes them | verified; anything a removed vector dominates, the vector that dominates it dominates too; memory only |
| NAMOA* (JACM 2010, step 4) | filtering by `COSTS` eager on every `F` in `OPEN`, empty triples removed | lazy, at selection and at generation | verified; a filtered triple is never selected, a lazily dropped one neither |
| NAMOA* (JACM 2010, step 3) | select any triple with an `f` non-dominated in `OPEN` | the lexicographically smallest `f` | verified; a lexicographic minimum is non-dominated, so this is one admissible choice |
| NAMOA* (JACM 2010, step 5) | skip successors that close a cycle in `SG`; a `g_m` equal to a known one only adds a back pointer | no cycle test; equal or dominated `g_m` pruned by weak dominance | verified; with non-negative costs a cycle's `g` is weakly dominated by the vector already known at its state |
| NAMOA* (JACM 2010, Section 2.2.3) | sets `H(n)` of estimates, `F` the non-dominated `g + h` | one estimate per state | verified; the multi-valued case is `namoa_star_dr_mvh_lazy` |
| NAMOA* (JACM 2010, Section 6) | optimal in path expansions among admissible algorithms with consistent heuristics (Theorem 6.4); no optimal algorithm with inconsistent ones (6.3) | no claim of optimal effort in the docs | verified; our weak-dominance pruning keeps one path per cost vector, so the expanded set can be smaller than the paper's and Theorem 6.4 is not restated |
| NAMOA* (JACM 2010, Theorem 4.3) | termination on infinite graphs needs costs bounded below by `δ > 0` | not checked; documented on `namoa_star` | verified; doc fixed (it only required non-negative costs) |
| NAMOA*dr | `t`-discard with case split on `v1` | weak dominance on objectives `1..N` | lexicographic order makes the first objective of closed labels and solutions no larger; equality in all objectives is the duplicate case above |
| EMOA* | AVL tree per frontier | sorted `Vec` (`emoa_star`), treap (`emoa_star_tree`) | same queries; the tree also prunes by subtree bounding boxes |
| PP-A* | two-sided bounded test, returns `tl` | one-sided (first objective), returns `br` | a different but proven covering; see `pp_a_star` |
| BOBA* | preliminary searches and directions in parallel threads | sequential by default; threads behind the `parallel` feature | same distances; the join drops boundary vectors a late bound leaves |
| BOBA* | early solution update, heuristic tuning | implemented; the join of the two directions also drops a dominated boundary solution | an early solution made before the other direction's bound arrives can be dominated by that direction's first |

### A*pex against Algorithm 2

The first draft was written from the paper's description and deviated in
three rules. The draft was sound (it merged less, so it returned more
solutions), but all three rules now follow the paper. The tests for the
`f`-based bound and for solution merging fail on the draft. The apex
update on pruning is only needed once solutions merge, so its test
passes on the draft and instead fails when the update is removed from
the current code.

| Paper rule | Here | Status |
|---|---|---|
| Open list ordered by lexicographic `f` (line 6) | min-heap on `(f, pair index)` | matches; index breaks ties for determinism |
| `is_dominated`: a solution whose representative's truncated `f` ε-dominates the pair's truncated `f` (line 20) | `is_covered_by_solution`, objectives `1..N` | matches; was a full-vector check, equivalent because objective 0 always holds (Lemma 1) |
| On that prune, the solution's apex becomes `min(f(AP), A')` (lines 21-22) | same | aligned (needed once solutions merge, footnote 3); test `pruning_lowers_the_solution_apex` fails without it |
| `is_dominated`: truncated `f` weakly dominated by `GTcl(s)` (line 24) | `TruncatedFront` of apexes per state; same `h` at a state, so apex and `f` compare alike | matches |
| `GTcl(s)` gets `Tr(f(AP))` before the goal test (line 9) | same | matches |
| Child: apex `A + c`, representative extended by the edge (line 14), checked by `is_dominated` before insertion (line 15) | same | matches |
| ε-bounded: `f(π) ⪯ε f(AP)` (Contribution 3) | `is_bounded`: `g(rep) + h ≤ (1 + ε)(apex + h)` | aligned (was `g(rep) ≤ (1 + ε) apex`, sound but merged less); test `merges_under_the_f_based_bound` |
| Merge: component-wise minimum apex, representative by method R, L or G; no merge if the choice is not bounded | `MergeRule::Greedy` (default): G, the larger slack `min_i (1 + ε_i − f_i(π)/f_i(AP)) / ε_i`, compared as the smaller worst excess `d_i (f_i(π) − f_i(AP)) / (n_i f_i(AP))` by exact cross-multiplication (`search_core::WideMul`); objectives where `f_i(π) = f_i(AP)` have the full slack 1 and are skipped, which also covers `ε_i = 0` and `f_i(AP) = 0`; the new pair's representative wins a tie. `MergeRule::ReverseLex`: L, first argument on a tie. R is not offered | matches (G and L). Both are cross-checked against brute force by the same proptests. The first draft took the first bounded of the two (new, then open). Bench `grid_8x8_3obj/apex_star_10pct`: L 290 µs, G 259 µs |
| `insert` merges into the first pair of the list it merges with, else adds (lines 27-35) | same, over the open pairs at the state | matches |
| Goal pair: `insert(AP, solutions)` (line 11) | `insert_solution` | aligned (goal pairs were appended unmerged, so the output was larger than the paper's); test `merges_goal_pairs_into_solutions` |
| Single goal state | goal predicate; solution pairs at different goals may merge | deliberate: solution pairs are never extended and `h = 0` at every goal, so the merged representative still covers both apexes |
| Returns every solution pair's representative (line 18) | drops dominated and duplicate representatives, sorts | deliberate: a dominating representative covers whatever the dropped one covered |
| Consistent heuristic | documented requirement | matches; the `f`-based bound relies on it to stay bounded under extension |

### NAMOA*dr against two sources

Pulido et al. 2015 itself was not reachable. `namoa_star_dr` is checked
against Pulido's thesis (Tables 4.4 and 4.5, Definition 4.5, eq. 4.9)
and against Algorithm 1 of Wolff, Felner, Salzman (arXiv 2606.20644),
which restates Pulido 2015. No deviation needed a fix.

| Rule | Source | Here | Status |
|---|---|---|---|
| Pop the lexicographically smallest `f` | thesis; Wolff Alg. 1 line 8 | min-heap on `(f, index)` | matches |
| At selection, discard if `G_cl^Tr(s)` weakly dominates `Tr(g)` | thesis (`t`-discard); Wolff line 9 | `TruncatedFront` per state | matches; Wolff writes `⪯`, the thesis's `t`-discard is strict on a first-objective tie, which only differs for duplicate vectors (row "NAMOA*dr" above) |
| At selection, discard if `G_cl^Tr(s_goal)` weakly dominates `Tr(f)` | thesis; Wolff line 11 | `TruncatedFront` of solution costs | matches; a goal predicate replaces `s_goal`, all goals share one archive |
| Add `Tr(g)` to `G_cl^Tr(s)` before the goal test; a goal label becomes a solution and is not expanded | thesis; Wolff lines 13 to 16 | same | matches |
| At generation, the same two checks on the child (`Tr(g')`, `Tr(f')`) | thesis; Wolff lines 21 and 23 | same, in `admits` | matches |
| No open frontier: only closed sets and solutions | Wolff Alg. 1 | per-state open frontier, full dominance, dominated open labels dropped (shared with NAMOA*) | deliberate: extra pruning, sound because a label weakly dominated by an open label at its state reaches nothing that label does not |
| Consistent heuristic | thesis; Wolff Property 1 | documented requirement | matches |

### Multi-valued heuristics against Wolff et al. 2026

| Paper rule | Here | Status |
|---|---|---|
| MVH: a set of non-dominated vectors per state; admissible and consistent as in Section 2.1 | `MvHeuristic`, sets need not be non-dominated (the lazy search sorts and dedups, the eager one tolerates redundancy); `SingleValued` wraps a `Heuristic` | deliberate: consistency is the monotone form of JACM 2010 Definition 5.7 (`∀h' ∈ H(s') ∃h ∈ H(s)`). Wolff et al. write the quantifiers the other way round, but their Table 1 only satisfies the monotone form (test `table_1_is_monotone_only`), as do exact Pareto sets. NAMOA*dr-mvh needs a path-consistent chain of estimates along every path, which either form gives |
| Naive DR with an MVH is unsound (Lemma 1, Figure 2, Tables 1 and 2) | unit test `naive_dimensionality_reduction_loses_a_solution` runs the lazy search with the order check off and loses `(16, 12)`; `tests/it/mvh.rs` runs the same graph through both searches and NAMOA* | matches |
| NAMOA*dr-mvh: one node per `(g, h)`, successors only for `h ⪯ c + h'` (Definition 1), closed sets per `(s, h)` (Theorem 1) | NAMOA*dr's engine on the graph of `(state, estimate)` pairs, whose single-valued heuristic is consistent by construction; a virtual root fans out to one label per start estimate | matches; the engine's open frontier is per pair |
| L-NAMOA*dr-mvh: one node per path, `ChooseH` binds the lexicographically smallest estimate not discarded by solutions (lines 26, 34 to 40) | same | matches |
| Lazy re-evaluation of a popped label whose `f` became dominated (lines 9 to 15) | re-bound and pushed again under the same arena index | matches |
| `LocalDomCheck`: truncated check, trusted only if `max g'_1 ≤ g_1`, else full check against `G_cl(s)` (lines 41 to 47) | same; the maximum is over every `g_1` ever closed at the state | deliberate: can only fall back more often |
| Global check against `Tr` of solutions | same (`TruncatedFront`) | matches; sound although labels leave the open list out of `f_1` order, because Pareto-optimal solutions are still found in lexicographic order (see `namoa_star_dr_mvh_lazy`) |
| Single goal | goal predicate | deliberate, as everywhere in the crate |
| A*pex-MVH: backward A*pex from the goal, the apexes per state (Section 6) | `apex_mvh` on a `ReverseGraph`: A*pex (greedy merges) on the reversed graph from the goal set, no goal, zero heuristic; the estimates are the full apexes of the pairs expanded at each state | matches; tested exact for `ε = 0` and admissible against exact Pareto sets otherwise |
| Consistency fix: while `c(s, s') + h' ≺ h` for some `h ∈ H(s)`, insert `c + h'` into `H(s)` and drop what it dominates, to a fix-point | `MvhTable::make_consistent`: insert `c + h'` whenever no estimate of `s` weakly dominates it, worklist over predecessors | deliberate: the paper's trigger alone does not give the monotone form NAMOA*dr-mvh needs (an uncovered but incomparable `c + h'` stays); tested monotone, admissible and `{0}` at goals |
| Querying the MVH | the eager variant asks the MVH on every generation | not cached: with table MVHs a cache would be the same map lookup and copy |

### A-BOA*ε against the anonymous preprint

`boa_star_anytime` (A-BOA*ε), `boa_star_anytime_ws` (A-BOA*ε-w) and
`boa_star_anytime_basic` (Basic-A-BOA*ε) share one engine.
`pp_a_star_iterative` is the paper's Iterative-PP-A* baseline.
`tests/it/anytime.rs` checks every reported front against brute force
for its factor, a strictly falling factor schedule, and the final front
against BOA*, with and without an expansion budget; dropping the kept
paths or weakening the weighted-sum test fails it.

| Paper rule | Here | Status |
|---|---|---|
| Seed `π_tl`, `π_br`: lexicographic optima `(c1, c2)` and `(c2, c1)` (Alg. 2, lines 1 and 2) | lexicographic A* with the given heuristic, counted in the stats | matches; equal costs give a one-solution front at once |
| `ILIST` of intervals, refine `argmax ε̂(I)` with `ε̂(I)/d` until all `ε̂ = 0` (lines 5 to 8) | same; ties go to the leftmost interval | matches |
| `ε̂(I) = max_π min(DF(π_tl, π), DF(π_br, π))` on `f` (Definition 2) | `ApproxFactor`, exact ratios of costs compared with `WideMul` | matches; real `ε` replaced by exact rationals |
| Alg. 3: `OPEN ← Π`, `g2_min` reset per call, `π_tmp ← π_tl` | same | matches |
| `is_dominated`: `f1 ≥ f1(π_br) ∨ f2 ≥ f2(π_tl) ∨` weighted-sum test, then BOA*ε's test, keeping paths pruned only by the tolerance in `Π_tmp` (lines 20 to 27) | same | matches |
| A solution closes the interval `⟨π_tmp, π, Π_tmp⟩` (lines 9 to 12); the rest goes to `⟨π_tmp, π_br, Π_tmp⟩` (line 18) | same | matches |
| Weighted sum: prune when `g1 + w g2 + h_w ≥ f1(π_br) + w f2(π_tmp)` (Section 5) | same, integer `w`; off when the right side saturates | matches |
| Basic-A-BOA*ε: rerun BOA*ε from scratch with `ε / d` until no kept path can matter | the same engine restarted over the whole front each iteration | matches |
| `d` real, `ε` real | `d: u32 >= 2`; `ε / d` multiplies the denominator; a saturated denominator gives `ε = 0` | deliberate: exact arithmetic; the zero fallback avoids stalling at the smallest ratio |
| `DF` with a zero `f` component | `ApproxFactor::Infinite`; an infinite interval is refined with `ε = 1` | deliberate: `∞ / d` would never shrink; any finite `ε` keeps Theorem 1 and lets Theorem 2's argument apply |
| Anytime: solutions can be read off `ILIST` at any time | `on_front` after seeding and whenever the overall factor falls; `Budget<Stats>` stops the search, folding the unexplored open paths into the current interval so the reported factor stays a valid bound | deliberate, follows `heuristic-search`'s anytime callbacks |
| Single goal | goal predicate; one `g2_min` over all goals | deliberate, as everywhere in the crate |
| Iterative-PP-A* baseline: PP-A* from scratch for `ε = 0.01, 0.001, 0.0001, 0` | `pp_a_star_iterative` with a caller's schedule (the paper's in the bench), same `on_front`, `Budget` and `AnytimeFront`; reports a run when its `ε` lowers the factor; the budget is checked between runs | matches. `ε` becomes an `ApproxFactor` scaled by the smallest positive solution cost, since `Cost` has no unit |

### SIP-MOSP against Xu et al. 2026

`sip_mosp` (feature `parallel`) and the reusable `IndexedFront`.
`tests/it/sip.rs` runs every property case with four worker and index
configurations, two or three times each, against brute force (two to
four objectives) and NAMOA* (larger graphs, and a layered graph whose
front of over 64 vectors exercises indexed snapshots and deltas).

| Paper rule | Here | Status |
|---|---|---|
| `R` search workers, `P` update workers owning disjoint vertex sets, one goal-update worker (Alg. 1) | same; owner by hash of the state (`G::State: Hash`), `std::thread::scope` | matches |
| Per-worker multi-producer incoming queue, private best-first queue | `Mutex<Vec>` plus `Condvar` inbox, private `BinaryHeap` | matches |
| Search worker: cyclic objective order `π_r`, expand, prune children by endpoint snapshot on `g` and goal snapshot on `f`, send to the owner (Alg. 2) | same; worker `r` orders by `f` rotated to start at objective `r mod N` | matches |
| Update worker: reuse a negative result if the carried version is current, else re-query; `F_v ← F_v \ {z : g ⪯ z} ∪ {g}`; publish; hand to the less loaded of two hashed search workers (Alg. 3) | same | matches; the canonical `Front::insert` also rejects a weakly dominated `g`, so a stale reuse cannot admit one |
| Snapshot `(B, Δ, I, k)`: sorted base, recent delta with its minimum, index over the base; base only below `θ_I`; consolidate when the delta is full (Section IV.B) | same; `SipConfig` sets `θ_I` (`index_threshold`, default 64), the delta capacity `max(delta_min, |F| / delta_divisor)` (defaults 8, 8) and `block_size` (16); the base is an `Arc` shared between versions | matches; the defaults are ours (the paper gives no values); tests also run with tiny thresholds |
| `SnapshotDominates`: delta guarded by its minimum, binary-searched lexicographic prefix, then block-minimum (BM) or segment-tree (ST) index (Algs. 4, 5) | `Snapshot::dominates` and `IndexedFront` with `FrontIndex::{None, Blocks, Tree}`, block size 16 | matches |
| Publication of immutable snapshots | per-owner `RwLock<HashMap<state, Arc<Snapshot>>>`; readers clone an `Arc` under a short read lock | deliberate: no atomic pointer swap without dependencies |
| Termination on stable global quiescence, checked by repeated observation | one atomic count of labels sent but not yet processed; the last `finish` stops all workers | deliberate: simpler, same condition (the paper avoids the counter for speed) |
| Lower bounds: exact per-objective Dijkstra | any admissible heuristic, cloned per search worker | deliberate |
| Single goal `t` | goal predicate; all goals share the goal worker | deliberate, as everywhere |
| Indexed pruning as a sequential frontier | `IndexedFront` is public; on a 16000-vector four-objective antichain the block index answers 2000 queries in 107 ms against 120 ms for `Front` (the tree index is slower), so the sequential searches keep `Front` | measured; not adopted |

Bench (20 hardware threads, machine under load): on the 10x10
three-objective grid NAMOA*dr takes 2.9 ms and SIP-MOSP 4.1 ms (4 + 4
workers, blocks) to 7.2 ms (2 + 2); on the 24x24 two-objective grid
parallel BOBA* takes 5.5 ms and SIP-MOSP 15 to 21 ms. These graphs are
far smaller than the paper's, where coordination is amortized.

### Objective aggregation against Peer et al.

H. Peer, E. Weiss, R. Alterovitz, O. Salzman. "Generalizing
Multi-Objective Search via Objective-Aggregation Functions." arXiv
2509.22085. `aggregated_search` and `Aggregation`; `tests/it/aggregate.rs`
checks additive and bottleneck hidden objectives, exact and approximate,
against brute force over the aggregated front.

| Paper rule | Here | Status |
|---|---|---|
| Hidden cost by a monotone extension `F_ext(g, c)`, solution cost `F_agg(c_ext)` monotone (Section IV.A) | `Aggregation { extend, aggregate }`, any `M` hidden and `K` solution objectives | matches; `F_ext` must also never decrease `g`, so `f = g + h` bounds completions |
| Open list in lexicographic `F_agg(f)` order | same | matches |
| Path dominance on the hidden `g` per vertex | full weak dominance against the labels expanded at the state | matches |
| Solution dominance on `F_agg(f)`; `ε` variant on the same test | same, `Epsilon` per solution objective | matches |
| Returns the aggregated front (Theorem 1) | plus a final Pareto filter of the aggregated costs, and the hidden cost of each solution | matches |
| Applications (POI coverage, shadows, road types) | not built in; expressed by the caller's `extend` and `aggregate` | deliberate |

### Benchmark suite of Peer et al.

H. Peer, C. Hernández, S. Koenig, A. Felner, O. Salzman. "Bridging the
Evaluation Gap: Standardized Benchmarks for Multi-Objective Search."
arXiv 2603.24084. The suite's data is public on GitHub but large, and no
licence is stated, so nothing is downloaded or bundled; `benchmarks.rs`
generates instances from the paper's definitions, and the `suite` bench
group runs the exact and approximate searches on them.

| Family (section) | Here | Status |
|---|---|---|
| Random-cost grids `Grid_2`, `Grid_3`: four-connected, costs uniform in `[1, 10]` per objective, auxiliary source and target on the left and right columns (4.3) | `random_grid` | matches the definition; seeds are ours, so the instances are not the released ones |
| NetMaker: Hamiltonian cycle with one cost from each of `[1, 333]`, `[334, 666]`, `[667, 1000]` in random order, local edges within the identifier window with costs in `[1, 99]`, out-degree from `amin + 1..=amax` (4.3) | `netmaker`, `netmaker_queries` (sources in the first, targets in the last tenth) | matches the definition; seeds are ours |
| GUARDS grids: eight-connected, length 10/14, exposure from the number of guards with line of sight, diagonal cost the maximum over destination and side cells (4.4) | `guards_grid` from a Moving AI map and guard cells, Bresenham line of sight | matches the definition; the maps and guard placements are inputs |
| DIMACS NY/FLA and DIMACS-Extended: elevation difference, average endpoint out-degree, hop count (4.2) | `RoadGraph` loads DIMACS; `road_extended` adds the three objectives (out-degree as the sum, twice the average, to stay integral; elevation is an input) and `BenchmarkGraph::project` picks variants | deliberate: integral out-degree objective |
| GER_10, robot motion-planning roadmaps (4.2, 4.5) | load with `BenchmarkGraph::from_arcs` | not generated: they need the suite's data |
| Reference Pareto sets, ε-dominance protocol | the property tests compare against brute force; `ApproxFactor`-style checks cover ε | the released reference sets are not used |

### Parallel MOA* against Ahmadi et al.

`parallel_moa_star` (feature `parallel`); `tests/it/parallel_moa.rs`
runs every case three times against brute force (two to four
objectives) and NAMOA* (larger graphs). Moving every solution cost down
a level regardless of the other threads' progress makes them fail.

| Paper rule | Here | Status |
|---|---|---|
| `k` threads, thread `i` a MOA* with objective `i` first in cyclic order (Alg. 2) | same, `std::thread::scope`, `N` threads | matches |
| Thread search: lexicographic order, lazy checks of `Tr(g)` against `G^Tr_1(s)` and `IsDominatedMD(f)` at pop and generation (Alg. 3) | same (LTMOA*-style order), one `TruncatedFront` per state | matches |
| `IsDominatedMD`: global scalar bounds, then each truncation level (Alg. 4) | same; levels hold rotated vectors with the truncated components zeroed | matches |
| `UpdateUpperBound` after each solution: move a level-`λ` vector down when its next objective is at most the latest solution cost of that objective's thread; a level-`k-1` scalar becomes a global bound (Alg. 5) | same; the latest solution is published under the thread's lock together with the solution | matches |
| Early termination of the thread led by objective `m` once its primary cost reaches the global bound `f_m` | same, `>=` | matches |
| Stop when the first thread ends; merge by removing duplicates | stop when the first thread ends; merge by a Pareto filter of the union | deliberate: also drops dominated solutions a thread may record after pruning by another thread's information |
| Shared bounds | one `Mutex` for the bounds, read once per pop into a per-thread copy | deliberate, to keep the lock out of the generation loop |
| Single goal | goal predicate | deliberate, as everywhere |

Bench (machine under load): 10x10 grid, three objectives, uninformed —
NAMOA*dr 1.04 ms, parallel MOA* 1.70 ms, SIP-MOSP 2.9 to 4.2 ms; 24x24
grid, two objectives — parallel BOBA* 2.7 ms, parallel MOA* 3.9 ms,
SIP-MOSP 8.5 to 11 ms.

### Gaps against the Craveirinha et al. 2024 survey

J. Craveirinha, J. Clímaco, R. Girão-Silva, M. Pascoal. "Multiobjective
Path Problems and Algorithms in Telecommunication Network Design:
Overview and Trends." Algorithms 17(6):222, 2024. Used as a map of
problem classes; what the crate lacked when it was read:

| Class (survey section) | Status here | Why |
|---|---|---|
| Minsum–maxmin: an additive cost and a bottleneck capacity (2.2.2; Martins 1984, Gandibleux et al. 2006) | **implemented**: `minsum_maxmin`, label setting with BOA*-style one-value-per-state pruning; min–max via `Reverse` | fits the `Graph` model with `Weight = (C, K)`; cross-checked against enumeration of simple paths |
| Constrained shortest path: minimum cost with a resource budget (2.2.3, relays and reliability thresholds as variants) | **implemented** for one budget: `constrained_shortest_path` | BOA*'s pruning plus `f2 <= budget`, first solution is optimal; cross-checked for every budget |
| Ranking methods: k-shortest loopless paths in one objective, keep the efficient ones (2.2.1, Algorithm 2; Clímaco and Martins 1982) | **implemented**: `ranking` | best-first enumeration of loopless paths up to the `(c2, c1)` optimum's first cost; cross-checked against brute force |
| Two-phase methods: supported points by weighted sums, then the duality gaps (2.2.1) | **implemented**: `supported_solutions` (phase 1, dichotomic weighted-sum A* compared exactly with `WideMul`, lexicographic ties) and `two_phase` (phase 2: BOA*'s label setting restricted to each triangle's box) | phase 1 is cross-checked against the hull vertices of the brute-force front, the whole method against brute force |
| Several maxmin objectives, or bottlenecks mixed with several sums (2.2.2; Pinto et al., Bornstein et al., Gandibleux et al.) | **implemented**: `minsum_maxmin_multi` (`N` sums, `M` bottlenecks) | label setting in lexicographic order with whole-label dominance per state, which keeps one path per non-dominated cost although efficient paths may run through weakly efficient sub-paths; cross-checked for (1 sum, 2 widths) and (2 sums, 1 width) |
| Hop count as an objective (2.2.2; Pascoal 2018) | covered | an additive objective with cost 1 per edge |
| Ratio objectives: cost/capacity (2.2.3(a)) | **implemented**: `min_cost_capacity_ratio` | the optimum is on the minsum–maxmin front; scanned with `WideMul` cross-multiplication; cross-checked |
| Linear fractional `Σc1 / Σc2` (2.2.3(c); Soroush 2008) | **implemented**: `min_ratio_path` | Dinkelbach's parametric search; the inner problem `min den·c1 − num·c2` has signed weights, summed exactly in a local 256-bit signed integer (not in search-core), by Bellman-Ford without negative cycles and by exhaustive simple-path search with them (NP-hard then); cross-checked against brute-force ratio minimisation |
| Cost/reliability `Σc / Πr` (2.2.3(b); Ahuja 1988) | **implemented**: `min_cost_reliability_ratio` | the optimum is on the (cost, reliability) Pareto front, found by label setting with exact multi-precision reliability products (a local bigint) and scanned with exact cross-multiplication; cross-checked against brute force |
| Quickest path, `sum delay + σ / min capacity` (2.2.3(d)) | **implemented**: `quickest_path` | a scan of the minsum–maxmin front (Pelegrín and Fernández) with the times compared as exact fractions through search-core's `ExactArith` (`(d1 c1 + σ) c2` against `(d2 c2 + σ) c1`), for every cost type; cross-checked |
| Disjoint path pairs (2.2.3(g); Suurballe, Bhandari) | **implemented** on the new `EdgeGraph` trait (an additive extension of `Graph` with edge identities): `disjoint_pair` for edge- or state-disjoint pairs (Suurballe with reduced costs, node splitting for states) | cross-checked against enumeration of path pairs for both kinds |
| Bicriteria disjoint path pairs (Clímaco and Pascoal 2009) | **implemented**: `supported_disjoint_pairs` (extreme supported pairs, polynomial: dichotomic weighted Suurballe with the weighted sums exact through `ExactArith`) and `disjoint_pair_front` (the full front, supported and non-supported: BOA* over the space of partial pairs, the first path then the second, each state carrying the edges or states used) | the full front is exponential, as for this NP-hard problem in general; cross-checked against pair enumeration for both kinds. Both work for every cost type with `ExactArith` |
| Maximally disjoint (relaxed) pairs, cost against shared edges (Pascoal and Clímaco 2020) | **implemented**: `maximally_disjoint_pairs` | the bi-objective two-unit flow (an edge carries one unit at its cost, or two at twice its cost and one shared edge) solved exactly as the whole ε-constraint family over the integral shared count at once: BOA* over partial pairs with objectives `(cost, shared)`; a penalty-parametric min-cost flow would only give the supported trade-offs; cross-checked against pair enumeration |
| k-shortest dissimilar paths (Moghanni et al. 2022) | **implemented** in its greedy form: `dissimilar_paths`, paths ranked by `(cost, edge sequence)` and kept when they share at most `max_shared` edges with every kept one | cross-checked against a greedy pass over all simple paths |
| Label-count objectives and knapsack-type paths (2.2.2, 2.2.3; Pascoal et al. 2013, Captivo et al. 2003) | **implemented** as encodings: `LabelCount` (the label set rides in the state, so the count is additive) and `KnapsackGraph` (a layered graph over items with capacities in the state, lost value as cost) | every search of the crate solves them; cross-checked against path and subset enumeration |
| Interactive and reference-point methods (INT class) | **implemented** as a library API: `ReferencePoint`, an iterator over the non-dominated solutions by weighted Chebyshev achievement `max_i w_i (c_i - r_i)` to a reference point, with `aim` to move the reference or the weights between "next solution" queries | best-first on the achievement of `f`, ties lexicographic so only non-dominated solutions come out; cross-checked against brute force, including a change of aim mid-session; the dialogue itself stays with the caller |
| Approximation schemes (2.3; Hansen, Warburton, Tsaggouris and Zaroliagis) | **implemented** for the constrained shortest path: `fptas_constrained` | Bellman-Ford rounds with one label per geometric box of first costs (ratio `1 + ε / (2 (n - 1))`, integer thresholds) and the smallest second cost per box: polynomially many labels, which the ε-searches do not bound; cross-checked against the exact `constrained_shortest_path` for random budgets and tolerances, exact at `ε = 0` |
| Uncertainty: stochastic, robust, fuzzy costs (2.4) | design note only (below) | different cost model |

### Design note: uncertain costs

The survey's Section 2.4 covers stochastic, robust and fuzzy edge costs.
None of them fits `Cost` as it stands, and none is implemented; this is
how they could enter the crate.

- **Scenario-robust costs** (min–max regret or min–max cost over a finite
  scenario set) are the easiest: a cost vector with one component per
  scenario is already an `N`-objective problem, and the robust optimum is
  Pareto-optimal for it. A scan of a front (as for `quickest_path`)
  picks the min–max solution; regret needs the per-scenario optima first.
- **Stochastic costs with independent distributions** need an order on
  distributions. First-order stochastic dominance is a partial order that
  plays the role of Pareto dominance, but it is not preserved by addition
  of independent variables in the way label setting needs, so pruning must
  be by bounds (mean and variance as two objectives is the common
  surrogate, and is additive under independence).
- **Fuzzy costs** (triangular or trapezoidal numbers) add component-wise;
  with an order on fuzzy numbers (the survey cites an order relation that
  reduces to comparing a few parameters) they become multi-objective
  vectors of those parameters, so the existing searches apply once the
  order is fixed.

A future `UncertainCost` trait would therefore sit beside `Cost`, with an
adapter that maps each uncertain cost to the vector the chosen criterion
compares; the searches would stay unchanged.

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
4. **EMOA\*:** truncated 3D frontier (sorted `Vec`), `emoa_star`; the
   balanced-tree variant `emoa_star_tree` on `TreeFront`.
5. **Benchmarks and 0.1.0:** DIMACS loader (`RoadGraph`, in the library,
   `no_std`, parsing from `&str`), criterion benches on a generated road
   network in DIMACS format, release.

## Cross-crate needs

None for milestone 1. Later:

- `apex_star` needs `search_core::WideMul` (search-core `draft/initial`,
  "Add WideMul, an exact cost-by-cost product") for the greedy merge rule.

- Consumers (miniplan, a future `heuristic-search`) implement `Graph` and
  `Heuristic` for their models. If `heuristic-search` defines a
  single-objective successor trait with the same shape, both may move to
  `search-core` (only once both exist, per architecture.md).
- BOBA* and the exact test heuristic need single-objective searches. Keep
  small private ones here (BOBA*'s cost-bounded lexicographic A*) rather
  than depend on `heuristic-search`.

## Open questions

- Should states require `Ord` (BTreeMap, no dependency) or `Hash` (needs
  `hashbrown` in `no_std`)? Draft uses `Ord`.
- Should `Graph` stay a callback-into-`Vec` interface, or return an
  iterator via a GAT?
- Is a goal predicate enough, or do consumers need a goal state for
  heuristics and BOBA*?
- Should the crate guard against negative costs (an `Unsigned` marker
  trait) instead of documenting the precondition?
- BOBA*'s preliminary searches run one after another. The paper runs
  them two at a time in threads; a `std` feature could add that.
- Should `brute_force` stay public or move to the tests? Draft: public.
