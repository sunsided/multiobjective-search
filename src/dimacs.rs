//! A loader for road networks in the format of the 9th DIMACS
//! Implementation Challenge on shortest paths.
//!
//! The challenge distributes each road network as a distance graph
//! (`USA-road-d.*.gr`), a travel-time graph (`USA-road-t.*.gr`) with the same
//! arcs in the same order, and node coordinates (`USA-road-*.co`). The
//! bi-objective search literature (BOA*, BOBA*, PP-A*) uses distance and time
//! as its two objectives. [`RoadGraph`] zips the two `.gr` files into one
//! graph with `[distance, time]` edge costs.
//!
//! The format, line by line (`c` lines are comments everywhere):
//!
//! - `.gr`: one problem line `p sp <nodes> <arcs>`, then `<arcs>` arc lines
//!   `a <from> <to> <weight>`, nodes numbered `1..=<nodes>`;
//! - `.co`: one problem line `p aux sp co <nodes>`, then node lines
//!   `v <id> <x> <y>`, coordinates as signed integers (micro-degrees in the
//!   USA files).
//!
//! The loader parses from `&str`, so it works without `std`; read the files
//! with `std::fs::read_to_string` or decompress them first.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::str::FromStr;

use crate::graph::{Graph, ReverseGraph};

/// A road network with `[distance, time]` arc costs, loaded from DIMACS
/// files.
///
/// States are the DIMACS node ids, `1..=node_count()`. Arcs are stored in
/// compressed sparse rows in both directions, so [`Graph::successors`] and
/// [`ReverseGraph::predecessors`] are a slice copy each.
///
/// # Examples
///
/// ```
/// use multiobjective_search::{RoadGraph, boa_star, ZeroHeuristic};
///
/// let distance = "c tiny\np sp 3 3\na 1 2 10\na 2 3 10\na 1 3 25\n";
/// let time = "p sp 3 3\na 1 2 5\na 2 3 5\na 1 3 6\n";
/// let graph = RoadGraph::from_dimacs(distance, time)?;
/// assert_eq!(graph.node_count(), 3);
///
/// let out = boa_star(&graph, ZeroHeuristic, 1, |s| *s == 3);
/// let costs: Vec<_> = out.solutions.iter().map(|s| s.cost).collect();
/// assert_eq!(costs, [[20, 10], [25, 6]]);
/// # Ok::<(), multiobjective_search::DimacsError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoadGraph {
    /// Number of nodes; ids run from 1 to this.
    nodes: u32,
    /// Forward adjacency: arcs of node `v` are
    /// `out_arcs[out_start[v - 1]..out_start[v]]`. Length `nodes + 1`.
    out_start: Vec<usize>,
    /// Forward arcs as `(to, [distance, time])`, grouped by source.
    out_arcs: Vec<(u32, [u32; 2])>,
    /// Reverse adjacency offsets, as [`RoadGraph::out_start`].
    in_start: Vec<usize>,
    /// Reverse arcs as `(from, [distance, time])`, grouped by target.
    in_arcs: Vec<(u32, [u32; 2])>,
    /// Coordinates by node id minus one; empty until
    /// [`RoadGraph::with_coordinates`]. `None` for nodes without a `v` line.
    coordinates: Vec<Option<[i32; 2]>>,
}

/// An error while parsing a DIMACS file, with the 1-based line where it
/// occurred (the line after the last one for errors found at the end).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimacsError {
    /// 1-based line number in the file being parsed.
    pub line: usize,
    /// What went wrong.
    pub kind: DimacsErrorKind,
}

/// The kinds of [`DimacsError`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DimacsErrorKind {
    /// No problem line before the first data line or the end of the file.
    MissingProblemLine,
    /// A second problem line.
    DuplicateProblemLine,
    /// A line that does not have the expected fields or numbers.
    Malformed,
    /// A node id outside `1..=nodes`.
    NodeOutOfRange(u32),
    /// The file has a different number of arcs than its problem line says,
    /// or the time file a different number than the distance file (line 0).
    ArcCount {
        /// The count on the problem line.
        expected: usize,
        /// The number of arc lines.
        found: usize,
    },
    /// The distance and time files disagree on the node count, or the
    /// coordinate file on the graph's node count.
    NodeCount {
        /// The graph's node count.
        expected: u32,
        /// The count in the second file.
        found: u32,
    },
    /// The time file's arc on this line connects other nodes than the
    /// distance file's arc at the same position.
    ArcMismatch,
    /// A second `v` line for the same node.
    DuplicateNode(u32),
}

impl fmt::Display for DimacsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: ", self.line)?;

        match &self.kind {
            DimacsErrorKind::MissingProblemLine => f.write_str("missing problem line"),
            DimacsErrorKind::DuplicateProblemLine => f.write_str("second problem line"),
            DimacsErrorKind::Malformed => f.write_str("malformed line"),
            DimacsErrorKind::NodeOutOfRange(id) => write!(f, "node {id} out of range"),
            DimacsErrorKind::ArcCount { expected, found } => {
                write!(f, "expected {expected} arcs, found {found}")
            }
            DimacsErrorKind::NodeCount { expected, found } => {
                write!(f, "expected {expected} nodes, found {found}")
            }
            DimacsErrorKind::ArcMismatch => {
                f.write_str("arc differs from the distance file's arc at this position")
            }
            DimacsErrorKind::DuplicateNode(id) => write!(f, "node {id} listed twice"),
        }
    }
}

impl core::error::Error for DimacsError {}

impl RoadGraph {
    /// Parses a distance `.gr` file and a time `.gr` file with the same
    /// arcs in the same order.
    ///
    /// # Errors
    ///
    /// Returns a [`DimacsError`] for a malformed or inconsistent file:
    /// a missing or second problem line, a malformed line, a node id out of
    /// range, an arc count that differs from the problem line, or two files
    /// that differ in node count ([`DimacsErrorKind::NodeCount`], line 0),
    /// in arc count ([`DimacsErrorKind::ArcCount`], line 0) or in an arc's
    /// endpoints ([`DimacsErrorKind::ArcMismatch`], at the time
    /// file's line).
    pub fn from_dimacs(distance: &str, time: &str) -> Result<Self, DimacsError> {
        let distance = parse_gr(distance)?;
        let time = parse_gr(time)?;

        if distance.nodes != time.nodes {
            return Err(DimacsError {
                line: 0,
                kind: DimacsErrorKind::NodeCount {
                    expected: distance.nodes,
                    found: time.nodes,
                },
            });
        }

        if distance.arcs.len() != time.arcs.len() {
            return Err(DimacsError {
                line: 0,
                kind: DimacsErrorKind::ArcCount {
                    expected: distance.arcs.len(),
                    found: time.arcs.len(),
                },
            });
        }

        let mut arcs = Vec::with_capacity(distance.arcs.len());
        for (d, t) in distance.arcs.iter().zip(&time.arcs) {
            if (d.from, d.to) != (t.from, t.to) {
                return Err(DimacsError {
                    line: t.line,
                    kind: DimacsErrorKind::ArcMismatch,
                });
            }

            arcs.push((d.from, d.to, [d.weight, t.weight]));
        }

        Ok(Self::from_arcs(distance.nodes, &arcs))
    }

    /// Builds a graph over nodes `1..=nodes` from `(from, to, [distance,
    /// time])` arcs.
    ///
    /// # Panics
    ///
    /// Panics if an arc endpoint is outside `1..=nodes`.
    #[must_use]
    pub fn from_arcs(nodes: u32, arcs: &[(u32, u32, [u32; 2])]) -> Self {
        let (out_start, out_arcs) = compressed_rows(nodes, arcs.iter().map(|a| (a.0, a.1, a.2)));
        let (in_start, in_arcs) = compressed_rows(nodes, arcs.iter().map(|a| (a.1, a.0, a.2)));

        Self {
            nodes,
            out_start,
            out_arcs,
            in_start,
            in_arcs,
            coordinates: Vec::new(),
        }
    }

    /// Adds the node coordinates from a `.co` file.
    ///
    /// # Errors
    ///
    /// Returns a [`DimacsError`] for a missing or second problem line, a
    /// malformed line, a node count that differs from the graph's, a node id
    /// out of range or listed twice.
    pub fn with_coordinates(mut self, co: &str) -> Result<Self, DimacsError> {
        let mut coordinates = vec![None; self.nodes as usize];
        let mut header = false;
        let mut line_number = 0;

        for (index, line) in co.lines().enumerate() {
            line_number = index + 1;
            let error = |kind| DimacsError {
                line: line_number,
                kind,
            };
            let mut fields = line.split_whitespace();

            match fields.next() {
                None | Some("c") => {}
                Some("p") => {
                    if header {
                        return Err(error(DimacsErrorKind::DuplicateProblemLine));
                    }

                    let found = parse_co_header(fields).ok_or(error(DimacsErrorKind::Malformed))?;
                    if found != self.nodes {
                        return Err(error(DimacsErrorKind::NodeCount {
                            expected: self.nodes,
                            found,
                        }));
                    }

                    header = true;
                }
                Some("v") if header => {
                    let (id, x, y) =
                        parse_fields(fields).ok_or(error(DimacsErrorKind::Malformed))?;
                    let slot = node_slot(&mut coordinates, id)
                        .ok_or(error(DimacsErrorKind::NodeOutOfRange(id)))?;
                    if slot.replace([x, y]).is_some() {
                        return Err(error(DimacsErrorKind::DuplicateNode(id)));
                    }
                }
                Some("v") => return Err(error(DimacsErrorKind::MissingProblemLine)),
                Some(_) => return Err(error(DimacsErrorKind::Malformed)),
            }
        }

        if !header {
            return Err(DimacsError {
                line: line_number + 1,
                kind: DimacsErrorKind::MissingProblemLine,
            });
        }

        self.coordinates = coordinates;

        Ok(self)
    }

    /// Number of nodes; the node ids are `1..=node_count()`.
    #[must_use]
    pub fn node_count(&self) -> u32 {
        self.nodes
    }

    /// Number of arcs.
    #[must_use]
    pub fn arc_count(&self) -> usize {
        self.out_arcs.len()
    }

    /// The coordinates of `node` as `[x, y]`, or `None` if no coordinates
    /// were loaded for it.
    #[must_use]
    pub fn coordinates(&self, node: u32) -> Option<[i32; 2]> {
        let index = usize::try_from(node.checked_sub(1)?).ok()?;
        self.coordinates.get(index).copied().flatten()
    }

    /// The arcs of `node` in the compressed rows `start` and `arcs`, or an
    /// empty slice for an id out of range.
    fn row<'a>(start: &[usize], arcs: &'a [(u32, [u32; 2])], node: u32) -> &'a [(u32, [u32; 2])] {
        let Some(index) = node.checked_sub(1).map(|i| i as usize) else {
            return &[];
        };

        match (start.get(index), start.get(index + 1)) {
            (Some(&from), Some(&to)) => &arcs[from..to],
            _ => &[],
        }
    }
}

impl Graph for RoadGraph {
    type State = u32;
    type Weight = [u32; 2];

    fn successors(&self, state: &u32, out: &mut Vec<(u32, [u32; 2])>) {
        out.extend_from_slice(Self::row(&self.out_start, &self.out_arcs, *state));
    }
}

impl ReverseGraph for RoadGraph {
    fn predecessors(&self, state: &u32, out: &mut Vec<(u32, [u32; 2])>) {
        out.extend_from_slice(Self::row(&self.in_start, &self.in_arcs, *state));
    }
}

/// One parsed `.gr` file.
struct GrFile {
    /// Node count from the problem line.
    nodes: u32,
    /// Arcs in file order.
    arcs: Vec<Arc>,
}

/// One arc line of a `.gr` file.
struct Arc {
    /// 1-based line number, for error messages.
    line: usize,
    /// Source node id.
    from: u32,
    /// Target node id.
    to: u32,
    /// Arc weight.
    weight: u32,
}

/// Parses one `.gr` file.
fn parse_gr(text: &str) -> Result<GrFile, DimacsError> {
    let mut header: Option<(u32, usize)> = None;
    let mut arcs = Vec::new();
    let mut line_number = 0;

    for (index, line) in text.lines().enumerate() {
        line_number = index + 1;
        let error = |kind| DimacsError {
            line: line_number,
            kind,
        };
        let mut fields = line.split_whitespace();

        match fields.next() {
            None | Some("c") => {}
            Some("p") => {
                if header.is_some() {
                    return Err(error(DimacsErrorKind::DuplicateProblemLine));
                }

                let parsed = parse_gr_header(fields).ok_or(error(DimacsErrorKind::Malformed))?;
                arcs.reserve(parsed.1);
                header = Some(parsed);
            }
            Some("a") => {
                let (nodes, _) = header.ok_or(error(DimacsErrorKind::MissingProblemLine))?;
                let (from, to, weight): (u32, u32, u32) =
                    parse_fields(fields).ok_or(error(DimacsErrorKind::Malformed))?;

                for id in [from, to] {
                    if id == 0 || id > nodes {
                        return Err(error(DimacsErrorKind::NodeOutOfRange(id)));
                    }
                }

                arcs.push(Arc {
                    line: line_number,
                    from,
                    to,
                    weight,
                });
            }
            Some(_) => return Err(error(DimacsErrorKind::Malformed)),
        }
    }

    let end = |kind| DimacsError {
        line: line_number + 1,
        kind,
    };
    let (nodes, expected) = header.ok_or(end(DimacsErrorKind::MissingProblemLine))?;
    if arcs.len() != expected {
        return Err(end(DimacsErrorKind::ArcCount {
            expected,
            found: arcs.len(),
        }));
    }

    Ok(GrFile { nodes, arcs })
}

/// Parses the fields after `p` of a `.gr` problem line: `sp <nodes> <arcs>`.
fn parse_gr_header<'a>(mut fields: impl Iterator<Item = &'a str>) -> Option<(u32, usize)> {
    if fields.next()? != "sp" {
        return None;
    }

    let (nodes, arcs) = parse_fields(fields)?;
    Some((nodes, arcs))
}

/// Parses the fields after `p` of a `.co` problem line:
/// `aux sp co <nodes>`.
fn parse_co_header<'a>(mut fields: impl Iterator<Item = &'a str>) -> Option<u32> {
    for expected in ["aux", "sp", "co"] {
        if fields.next()? != expected {
            return None;
        }
    }

    let (nodes,) = parse_fields(fields)?;
    Some(nodes)
}

/// Parses exactly the fields of a tuple of numbers, rejecting extra fields.
fn parse_fields<'a, T: Fields>(mut fields: impl Iterator<Item = &'a str>) -> Option<T> {
    let parsed = T::parse(&mut fields)?;
    fields.next().is_none().then_some(parsed)
}

/// A tuple of whitespace-separated numbers.
trait Fields: Sized {
    /// Parses the tuple's fields from the front of `fields`.
    fn parse<'a>(fields: &mut impl Iterator<Item = &'a str>) -> Option<Self>;
}

/// Parses the next field as a number.
fn next<'a, T: FromStr>(fields: &mut impl Iterator<Item = &'a str>) -> Option<T> {
    fields.next()?.parse().ok()
}

impl<A: FromStr> Fields for (A,) {
    fn parse<'a>(fields: &mut impl Iterator<Item = &'a str>) -> Option<Self> {
        Some((next(fields)?,))
    }
}

impl<A: FromStr, B: FromStr> Fields for (A, B) {
    fn parse<'a>(fields: &mut impl Iterator<Item = &'a str>) -> Option<Self> {
        Some((next(fields)?, next(fields)?))
    }
}

impl<A: FromStr, B: FromStr, C: FromStr> Fields for (A, B, C) {
    fn parse<'a>(fields: &mut impl Iterator<Item = &'a str>) -> Option<Self> {
        Some((next(fields)?, next(fields)?, next(fields)?))
    }
}

/// The coordinate slot of node `id`, or `None` if it is out of range.
fn node_slot(coordinates: &mut [Option<[i32; 2]>], id: u32) -> Option<&mut Option<[i32; 2]>> {
    let index = usize::try_from(id.checked_sub(1)?).ok()?;
    coordinates.get_mut(index)
}

/// Groups `(row, column, weight)` triples by row into compressed sparse rows
/// over rows `1..=nodes`, keeping the input order within a row.
///
/// # Panics
///
/// Panics if a row is outside `1..=nodes`.
fn compressed_rows(
    nodes: u32,
    entries: impl Iterator<Item = (u32, u32, [u32; 2])> + Clone,
) -> (Vec<usize>, Vec<(u32, [u32; 2])>) {
    let rows = nodes as usize;
    let row = |id: u32| {
        assert!(
            (1..=nodes).contains(&id),
            "node {id} out of range 1..={nodes}"
        );
        id as usize - 1
    };

    let mut start = vec![0; rows + 1];
    for (from, _, _) in entries.clone() {
        start[row(from) + 1] += 1;
    }

    for i in 0..rows {
        start[i + 1] += start[i];
    }

    let mut next = start.clone();
    let mut arcs = vec![(0, [0, 0]); start[rows]];
    for (from, to, weight) in entries {
        let slot = &mut next[row(from)];
        arcs[*slot] = (to, weight);
        *slot += 1;
    }

    (start, arcs)
}

#[cfg(test)]
mod tests {
    //! Unit tests for the DIMACS loader.

    use alloc::vec::Vec;

    use super::{DimacsErrorKind, RoadGraph};
    use crate::graph::{Graph, ReverseGraph};

    /// A distance file with comments and blank lines.
    const DISTANCE: &str = "c 9th DIMACS style\n\np sp 3 3\nc arcs\na 1 2 10\na 2 3 10\na 1 3 25\n";

    /// The matching time file.
    const TIME: &str = "p sp 3 3\na 1 2 5\na 2 3 5\na 1 3 6\n";

    /// Successors and predecessors mirror the arcs.
    #[test]
    fn adjacency_in_both_directions() {
        let graph = RoadGraph::from_dimacs(DISTANCE, TIME).expect("valid");
        let mut out = Vec::new();

        graph.successors(&1, &mut out);
        assert_eq!(out, [(2, [10, 5]), (3, [25, 6])]);

        out.clear();
        graph.predecessors(&3, &mut out);
        assert_eq!(out, [(2, [10, 5]), (1, [25, 6])]);

        out.clear();
        graph.successors(&0, &mut out);
        graph.successors(&4, &mut out);
        assert_eq!(out, []);
        assert_eq!(graph.arc_count(), 3);
    }

    /// Coordinates load by node id.
    #[test]
    fn coordinates() {
        let co = "c coords\np aux sp co 3\nv 1 -73530767 41085396\nv 3 10 -20\n";
        let graph = RoadGraph::from_dimacs(DISTANCE, TIME)
            .and_then(|g| g.with_coordinates(co))
            .expect("valid");

        assert_eq!(graph.coordinates(1), Some([-73_530_767, 41_085_396]));
        assert_eq!(graph.coordinates(2), None);
        assert_eq!(graph.coordinates(3), Some([10, -20]));
        assert_eq!(graph.coordinates(0), None);
    }

    /// Every error kind is reported with its line.
    #[test]
    fn errors() {
        let kind = |d: &str, t: &str| RoadGraph::from_dimacs(d, t).map(|_| ()).unwrap_err();

        let e = kind("a 1 2 3\n", TIME);
        assert_eq!((e.line, e.kind), (1, DimacsErrorKind::MissingProblemLine));
        let e = kind("p sp 3 1\np sp 3 1\n", TIME);
        assert_eq!((e.line, e.kind), (2, DimacsErrorKind::DuplicateProblemLine));
        let e = kind("p sp 3 1\na 1 x 3\n", TIME);
        assert_eq!((e.line, e.kind), (2, DimacsErrorKind::Malformed));
        let e = kind("p sp 3 1\na 1 4 3\n", TIME);
        assert_eq!((e.line, e.kind), (2, DimacsErrorKind::NodeOutOfRange(4)));
        let e = kind("p sp 3 2\na 1 2 3\n", TIME);
        assert_eq!(
            e.kind,
            DimacsErrorKind::ArcCount {
                expected: 2,
                found: 1
            }
        );
        let e = kind("p sp 4 3\na 1 2 10\na 2 3 10\na 1 3 25\n", TIME);
        assert_eq!(
            e.kind,
            DimacsErrorKind::NodeCount {
                expected: 4,
                found: 3
            }
        );
        let e = kind(DISTANCE, "p sp 3 2\na 1 2 5\na 2 3 5\n");
        assert_eq!(
            (e.line, e.kind),
            (
                0,
                DimacsErrorKind::ArcCount {
                    expected: 3,
                    found: 2
                }
            )
        );
        let e = kind(DISTANCE, "p sp 3 3\na 1 2 5\na 3 2 5\na 1 3 6\n");
        assert_eq!((e.line, e.kind), (3, DimacsErrorKind::ArcMismatch));

        let graph = RoadGraph::from_dimacs(DISTANCE, TIME).expect("valid");
        let co = |text: &str| {
            graph
                .clone()
                .with_coordinates(text)
                .map(|_| ())
                .unwrap_err()
        };
        assert_eq!(co("v 1 2 3\n").kind, DimacsErrorKind::MissingProblemLine);
        assert_eq!(
            co("p aux sp co 2\n").kind,
            DimacsErrorKind::NodeCount {
                expected: 3,
                found: 2
            }
        );
        assert_eq!(
            co("p aux sp co 3\nv 1 2 3\nv 1 2 3\n").kind,
            DimacsErrorKind::DuplicateNode(1)
        );
        assert_eq!(
            co("p aux sp co 3\nv 9 2 3\n").kind,
            DimacsErrorKind::NodeOutOfRange(9)
        );
        assert_eq!(
            co("p aux sp co 3\nv 1 2\n").kind,
            DimacsErrorKind::Malformed
        );
    }

    /// Error messages name the line.
    #[test]
    fn display() {
        let e = RoadGraph::from_dimacs("p sp 3 1\na 1 4 3\n", TIME).unwrap_err();
        assert_eq!(alloc::format!("{e}"), "line 2: node 4 out of range");
    }
}
