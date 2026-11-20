//! Generators and loaders for the standardized multi-objective search
//! benchmark suite of Peer, Hernández, Koenig, Felner, Salzman ("Bridging
//! the Evaluation Gap: Standardized Benchmarks for Multi-Objective
//! Search", arXiv 2603.24084).
//!
//! The suite's data (github.com/CRL-Technion/Multi-Objective-Search-Benchmarks)
//! is not bundled: its graphs are large and their licence is not stated.
//! Instead this module builds instances by the paper's definitions, from a
//! seed, and converts the inputs the paper builds on (DIMACS road arcs,
//! Moving AI maps with guards) into its objectives:
//!
//! - [`random_grid`]: the random-cost grids `Grid_d` (Section 4.3);
//! - [`netmaker`] and [`netmaker_queries`]: `NetMaker` graphs (Section 4.3);
//! - [`guards_grid`]: GUARDS-based bi-objective grids (Section 4.4);
//! - [`road_extended`]: the DIMACS-Extended objectives (Section 4.2).
//!
//! `GER_10` and the robot motion-planning roadmaps need data that only the
//! suite provides; [`BenchmarkGraph::from_arcs`] loads any such arc list.

use alloc::vec::Vec;

use crate::graph::{Graph, ReverseGraph};

/// A directed graph with `N` integer objectives per edge, over states
/// `0..nodes`, as the suite's instances are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkGraph<const N: usize> {
    /// Outgoing edges per state.
    out: Vec<Vec<(u32, [u32; N])>>,
    /// Incoming edges per state.
    into: Vec<Vec<(u32, [u32; N])>>,
}

/// A start–goal query.
pub type Query = (u32, u32);

impl<const N: usize> BenchmarkGraph<N> {
    /// A graph over `nodes` states with the arcs `(from, to, costs)`.
    ///
    /// # Panics
    ///
    /// Panics if an arc names a state `>= nodes`.
    #[must_use]
    pub fn from_arcs(nodes: u32, arcs: &[(u32, u32, [u32; N])]) -> Self {
        let n = usize::try_from(nodes).expect("node counts fit in usize");
        let mut graph = Self {
            out: alloc::vec![Vec::new(); n],
            into: alloc::vec![Vec::new(); n],
        };
        for &(from, to, cost) in arcs {
            graph.out[index(from)].push((to, cost));
            graph.into[index(to)].push((from, cost));
        }

        graph
    }

    /// The number of states.
    ///
    /// # Panics
    ///
    /// Never: the graph was built from a `u32` count.
    #[must_use]
    pub fn node_count(&self) -> u32 {
        u32::try_from(self.out.len()).expect("built from a u32 count")
    }

    /// The number of arcs.
    #[must_use]
    pub fn arc_count(&self) -> usize {
        self.out.iter().map(Vec::len).sum()
    }

    /// Every arc as `(from, to, costs)`.
    ///
    /// # Panics
    ///
    /// Never: the graph was built from a `u32` count.
    pub fn arcs(&self) -> impl Iterator<Item = (u32, u32, [u32; N])> + '_ {
        self.out.iter().enumerate().flat_map(|(u, edges)| {
            let u = u32::try_from(u).expect("built from a u32 count");
            edges.iter().map(move |&(v, c)| (u, v, c))
        })
    }

    /// The same graph with only the objectives `objectives`, in that order
    /// (for example the three-, four- and five-objective DIMACS-Extended
    /// variants of one five-objective graph).
    ///
    /// # Panics
    ///
    /// Panics if an objective index is `>= N`.
    #[must_use]
    pub fn project<const M: usize>(&self, objectives: [usize; M]) -> BenchmarkGraph<M> {
        let pick = |c: [u32; N]| objectives.map(|i| c[i]);
        let arcs: Vec<_> = self.arcs().map(|(u, v, c)| (u, v, pick(c))).collect();
        BenchmarkGraph::from_arcs(self.node_count(), &arcs)
    }
}

impl<const N: usize> Graph for BenchmarkGraph<N> {
    type State = u32;
    type Weight = [u32; N];

    fn successors(&self, state: &u32, out: &mut Vec<(u32, [u32; N])>) {
        out.extend_from_slice(&self.out[index(*state)]);
    }
}

impl<const N: usize> ReverseGraph for BenchmarkGraph<N> {
    fn predecessors(&self, state: &u32, out: &mut Vec<(u32, [u32; N])>) {
        out.extend_from_slice(&self.into[index(*state)]);
    }
}

/// `state` as an index.
fn index(state: u32) -> usize {
    usize::try_from(state).expect("u32 fits in usize")
}

/// A splitmix64 generator: deterministic from its seed, no dependencies.
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    /// The next 64 random bits.
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A uniform value in `low..=high`.
    fn range(&mut self, low: u32, high: u32) -> u32 {
        let span = u64::from(high - low) + 1;
        low + u32::try_from(self.next() % span).expect("below the span")
    }
}

/// A random-cost grid `Grid_N` (Section 4.3): a four-connected
/// `width × height` grid, every directed edge with `N` independent costs
/// uniform in `1..=10`, plus a source joined to every cell of the leftmost
/// column and a target joined from every cell of the rightmost one, with
/// costs from the same distribution. Cell `(x, y)` is state `y · width +
/// x`; the source and target are the last two states, and the query is
/// theirs. The paper uses widths and heights in `{300, 350, ..., 600}`.
///
/// # Panics
///
/// Panics if `width` or `height` is zero, or the grid does not fit `u32`
/// states.
#[must_use]
pub fn random_grid<const N: usize>(
    width: u32,
    height: u32,
    seed: u64,
) -> (BenchmarkGraph<N>, Query) {
    assert!(width > 0 && height > 0, "a grid needs at least one cell");
    let cells = width.checked_mul(height).expect("the grid fits u32 states");
    let (source, target) = (cells, cells + 1);
    let mut rng = Rng(seed);
    let mut cost = || core::array::from_fn(|_| rng.range(1, 10));
    let mut arcs = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let cell = y * width + x;
            if x + 1 < width {
                arcs.push((cell, cell + 1, cost()));
                arcs.push((cell + 1, cell, cost()));
            }

            if y + 1 < height {
                arcs.push((cell, cell + width, cost()));
                arcs.push((cell + width, cell, cost()));
            }
        }

        arcs.push((source, y * width, cost()));
        arcs.push((y * width + width - 1, target, cost()));
    }

    (
        BenchmarkGraph::from_arcs(cells + 2, &arcs),
        (source, target),
    )
}

/// A `NetMaker` graph (Section 4.3) over states `0..nodes` with three
/// objectives: a random directed Hamiltonian cycle whose edges get one cost
/// from each of `1..=333`, `334..=666` and `667..=1000` in random order, and
/// per state a random extra out-degree from `out_min + 1..=out_max` (the
/// cycle edge counts as one) filled with edges to states within the
/// locality window `[u - locality / 2, u + locality / 2]` of its identifier,
/// with costs uniform in `1..=99`; no self-loops or duplicates. The paper
/// uses `nodes` in `{10000, 20000, 30000}`, `out_min = 1`, `out_max = 10`
/// and `locality` in `{20, 50}`.
///
/// # Panics
///
/// Panics if `nodes < 2` or `out_min >= out_max`.
#[must_use]
pub fn netmaker(
    nodes: u32,
    out_min: u32,
    out_max: u32,
    locality: u32,
    seed: u64,
) -> BenchmarkGraph<3> {
    assert!(
        nodes >= 2 && out_min < out_max,
        "`NetMaker` needs two states and out_min < out_max"
    );
    let mut rng = Rng(seed);

    // A uniform permutation by Fisher-Yates.
    let mut order: Vec<u32> = (0..nodes).collect();
    for i in (1..order.len()).rev() {
        let j = index(rng.range(0, u32::try_from(i).expect("below nodes")));
        order.swap(i, j);
    }

    let mut out: Vec<Vec<(u32, [u32; 3])>> = alloc::vec![Vec::new(); index(nodes)];
    for k in 0..order.len() {
        let (u, v) = (order[k], order[(k + 1) % order.len()]);
        let mut bands = [(1, 333), (334, 666), (667, 1000)];
        for i in (1..3).rev() {
            let j = index(rng.range(0, u32::try_from(i).expect("below 3")));
            bands.swap(i, j);
        }

        out[index(u)].push((v, bands.map(|(low, high)| rng.range(low, high))));
    }

    for u in 0..nodes {
        let degree = rng.range(out_min + 1, out_max);
        let low = u.saturating_sub(locality / 2);
        let high = (u + locality / 2).min(nodes - 1);
        let mut free: Vec<u32> = (low..=high)
            .filter(|&v| v != u && out[index(u)].iter().all(|e| e.0 != v))
            .collect();

        while out[index(u)].len() < index(degree) && !free.is_empty() {
            let pick = index(rng.range(0, u32::try_from(free.len() - 1).expect("small")));
            let v = free.swap_remove(pick);
            out[index(u)].push((v, core::array::from_fn(|_| rng.range(1, 99))));
        }
    }

    let arcs: Vec<_> = out
        .iter()
        .enumerate()
        .flat_map(|(u, edges)| {
            edges
                .iter()
                .map(move |&(v, c)| (u32::try_from(u).expect("below nodes"), v, c))
        })
        .collect();
    BenchmarkGraph::from_arcs(nodes, &arcs)
}

/// `count` `NetMaker` queries (Section 4.3): sources uniform among the first
/// tenth of the identifiers, targets among the last tenth.
///
/// # Panics
///
/// Panics if `nodes < 10`.
#[must_use]
pub fn netmaker_queries(nodes: u32, count: usize, seed: u64) -> Vec<Query> {
    assert!(nodes >= 10, "queries need at least ten states");
    let tenth = nodes / 10;
    let mut rng = Rng(seed);
    (0..count)
        .map(|_| (rng.range(0, tenth - 1), rng.range(nodes - tenth, nodes - 1)))
        .collect()
}

/// A GUARDS-based bi-objective grid (Section 4.4) from a Moving AI map and
/// guard cells: an eight-connected directed graph over the passable cells,
/// objectives path length (10 per orthogonal, 14 per diagonal move) and
/// guard exposure. A cell's exposure is the number of guards that see it
/// (Bresenham line of sight through passable cells); an orthogonal move
/// costs the destination's exposure, a diagonal one the largest of the
/// destination and the two side cells, and is allowed only when both side
/// cells are passable. Cell `(x, y)` is state `y · width + x`; blocked cells
/// have no edges.
///
/// The map format is Moving AI's: a header with `height` and `width`
/// lines, a `map` line, then one row per line, `.`, `G` and `S` passable.
///
/// Returns `None` if the header or a row is malformed, or the map has more
/// cells than `u32` states.
///
/// # Panics
///
/// Never: cell indices are checked against the map size.
#[must_use]
pub fn guards_grid(map: &str, guards: &[(u32, u32)]) -> Option<BenchmarkGraph<2>> {
    let (width, height, open) = parse_map(map)?;
    let passable = |x: i64, y: i64| {
        x >= 0 && y >= 0 && x < i64::from(width) && y < i64::from(height) && open[cell(width, x, y)]
    };
    let exposure: Vec<u32> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| {
            let seen = guards
                .iter()
                .filter(|&&(gx, gy)| {
                    sees(
                        &passable,
                        (i64::from(gx), i64::from(gy)),
                        (i64::from(x), i64::from(y)),
                    )
                })
                .count();
            u32::try_from(seen).expect("guard counts fit")
        })
        .collect();

    let mut arcs = Vec::new();
    for y in 0..i64::from(height) {
        for x in 0..i64::from(width) {
            if !passable(x, y) {
                continue;
            }

            for (dx, dy) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ] {
                let (tx, ty) = (x + dx, y + dy);
                if !passable(tx, ty) {
                    continue;
                }

                let diagonal = dx != 0 && dy != 0;
                let risk = if diagonal {
                    if !passable(x + dx, y) || !passable(x, y + dy) {
                        continue;
                    }

                    exposure[cell(width, tx, ty)]
                        .max(exposure[cell(width, x + dx, y)])
                        .max(exposure[cell(width, x, y + dy)])
                } else {
                    exposure[cell(width, tx, ty)]
                };

                let id = |x: i64, y: i64| u32::try_from(cell(width, x, y)).expect("cells fit u32");
                arcs.push((id(x, y), id(tx, ty), [if diagonal { 14 } else { 10 }, risk]));
            }
        }
    }

    Some(BenchmarkGraph::from_arcs(width.checked_mul(height)?, &arcs))
}

/// The index of cell `(x, y)`.
fn cell(width: u32, x: i64, y: i64) -> usize {
    usize::try_from(y * i64::from(width) + x).expect("a cell on the map")
}

/// Parses a Moving AI map into its width, height and passability.
fn parse_map(map: &str) -> Option<(u32, u32, Vec<bool>)> {
    let mut lines = map.lines();
    let (mut width, mut height) = (None, None);
    for line in lines.by_ref() {
        let mut words = line.split_whitespace();
        match (words.next(), words.next()) {
            (Some("height"), Some(h)) => height = h.parse().ok(),
            (Some("width"), Some(w)) => width = w.parse().ok(),
            (Some("map"), None) => break,
            _ => {}
        }
    }

    let (width, height): (u32, u32) = (width?, height?);
    let mut open = Vec::new();
    for _ in 0..height {
        let row = lines.next()?;
        if row.chars().count() != index(width) {
            return None;
        }

        open.extend(row.chars().map(|c| matches!(c, '.' | 'G' | 'S')));
    }

    Some((width, height, open))
}

/// Returns `true` if `from` sees `to`: every cell on the Bresenham line
/// between them is passable.
fn sees(passable: &impl Fn(i64, i64) -> bool, from: (i64, i64), to: (i64, i64)) -> bool {
    let (mut x, mut y) = from;
    let (dx, dy) = ((to.0 - x).abs(), -(to.1 - y).abs());
    let (sx, sy) = (if x < to.0 { 1 } else { -1 }, if y < to.1 { 1 } else { -1 });
    let mut err = dx + dy;

    loop {
        if !passable(x, y) {
            return false;
        }

        if (x, y) == to {
            return true;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }

        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// The DIMACS-Extended objectives (Section 4.2, after Ahmadi et al. 2024)
/// on a road network given as arcs with `[distance, time]`: distance,
/// time, the absolute elevation difference of the endpoints, the sum of
/// the endpoints' out-degrees (twice their average, to stay integral), and
/// one hop. `elevation[v]` is the elevation of state `v`; the paper takes
/// it from external elevation data. Project with
/// [`BenchmarkGraph::project`] for the three- and four-objective variants.
///
/// # Panics
///
/// Panics if an arc names a state without an elevation.
#[must_use]
pub fn road_extended(
    nodes: u32,
    arcs: &[(u32, u32, [u32; 2])],
    elevation: &[i32],
) -> BenchmarkGraph<5> {
    let mut degree = alloc::vec![0_u32; index(nodes)];
    for &(u, _, _) in arcs {
        degree[index(u)] += 1;
    }

    let extended: Vec<_> = arcs
        .iter()
        .map(|&(u, v, [d, t])| {
            let rise = elevation[index(u)].abs_diff(elevation[index(v)]);
            (u, v, [d, t, rise, degree[index(u)] + degree[index(v)], 1])
        })
        .collect();

    BenchmarkGraph::from_arcs(nodes, &extended)
}
