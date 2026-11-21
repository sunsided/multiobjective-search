//! Multi-objective minimum spanning trees: the exact Pareto set by branch
//! and bound, and the two-phase method for two objectives.
//!
//! A problem the survey of Salzman et al. (arXiv 2510.25504, Section 4)
//! lists among the uses of the multi-objective search toolbox (Hamacher and
//! Ruhe 1994; Ramos et al. 1998 for the two-phase approach).

use alloc::vec::Vec;
use core::cmp::Ordering;

use search_core::{Cost, ExactArith, ExactNumber};

use crate::cost::{add, weakly_dominates};

/// An undirected edge `(u, v, cost)` between nodes `0..nodes`.
pub type TreeEdge<C, const N: usize> = (usize, usize, [C; N]);

/// A spanning tree as its edge indices (ascending) and its cost.
pub type Tree<C, const N: usize> = (Vec<usize>, [C; N]);

/// The Pareto-optimal spanning trees of an undirected graph, one per
/// non-dominated cost, sorted by cost; empty if the graph is disconnected.
///
/// Branch and bound over the edges in order: each edge is taken (if it
/// joins two components) or left out (if the rest can still connect the
/// graph). A partial tree is pruned when a known tree weakly dominates its
/// lower bound, its cost plus, per objective, the cheapest completion
/// (Kruskal on the remaining edges). Exact, and exponential in the worst
/// case, as the number of Pareto-optimal trees can be.
///
/// # Examples
///
/// ```
/// use multiobjective_search::pareto_spanning_trees;
///
/// // A triangle: any two edges span it.
/// let edges = [(0, 1, [1_u32, 5]), (1, 2, [2, 2]), (0, 2, [5, 1])];
/// let trees = pareto_spanning_trees(3, &edges);
/// let costs: Vec<_> = trees.iter().map(|t| t.1).collect();
/// assert_eq!(costs, [[3, 7], [6, 6], [7, 3]]);
/// ```
#[must_use]
pub fn pareto_spanning_trees<C: Cost, const N: usize>(
    nodes: usize,
    edges: &[TreeEdge<C, N>],
) -> Vec<Tree<C, N>> {
    let mut front = Vec::new();
    search(nodes, edges, &mut front, &|_| true);
    sorted(front)
}

/// The two-phase method for bi-objective spanning trees (Ramos et al.
/// 1998): phase 1 finds the extreme supported trees by dichotomic search
/// with weighted-sum Kruskal (weights from neighbouring trees, compared
/// exactly through [`ExactArith`], ties lexicographic); phase 2 finds the
/// rest by branch and bound restricted to each triangle between
/// neighbours. Same result as [`pareto_spanning_trees`].
///
/// # Examples
///
/// ```
/// use multiobjective_search::{pareto_spanning_trees, two_phase_spanning_trees};
///
/// let edges = [(0, 1, [1_u32, 5]), (1, 2, [2, 2]), (0, 2, [5, 1]), (1, 3, [1, 1]), (2, 3, [3, 4])];
/// assert_eq!(two_phase_spanning_trees(4, &edges), pareto_spanning_trees(4, &edges));
/// ```
#[must_use]
pub fn two_phase_spanning_trees<C: Cost + ExactArith>(
    nodes: usize,
    edges: &[TreeEdge<C, 2>],
) -> Vec<Tree<C, 2>> {
    let supported = supported_spanning_trees(nodes, edges);
    let mut all = supported.clone();
    for pair in supported.windows(2) {
        let (a, b) = (pair[0].1, pair[1].1);
        let mut front = pair.to_vec();
        search(nodes, edges, &mut front, &|lb: &[C; 2]| {
            lb[0] < b[0] && lb[1] < a[1]
        });
        all.extend(front.into_iter().filter(|t| t.1[0] > a[0] && t.1[0] < b[0]));
    }

    sorted(all)
}

/// Phase 1 of [`two_phase_spanning_trees`]: the extreme supported trees,
/// sorted by cost; empty if the graph is disconnected.
///
/// # Panics
///
/// Never: a graph connected in one edge order is connected in every.
#[must_use]
pub fn supported_spanning_trees<C: Cost + ExactArith>(
    nodes: usize,
    edges: &[TreeEdge<C, 2>],
) -> Vec<Tree<C, 2>> {
    let lex = |a: &[C; 2], b: &[C; 2]| a.cmp(b);
    let Some(left) = kruskal(nodes, edges, |x, y| lex(&x.2, &y.2)) else {
        return Vec::new();
    };
    let right = kruskal(nodes, edges, |x, y| {
        lex(&[x.2[1], x.2[0]], &[y.2[1], y.2[0]])
    })
    .expect("connected");
    if left.1 == right.1 {
        return alloc::vec![left];
    }

    let value =
        |w: &[C::Exact; 2], c: &[C; 2]| w[0].clone() * c[0].exact() + w[1].clone() * c[1].exact();
    let mut found = alloc::vec![left];
    let mut pending = alloc::vec![right];
    while let Some(b) = pending.pop() {
        let a = found.last().expect("starts with the left end").1;
        let w = [
            a[1].exact().abs_diff(&b.1[1].exact()),
            b.1[0].exact().abs_diff(&a[0].exact()),
        ];
        let p = kruskal(nodes, edges, |x, y| {
            value(&w, &x.2).cmp(&value(&w, &y.2)).then(lex(&x.2, &y.2))
        })
        .expect("connected");
        if value(&w, &p.1) < value(&w, &a) {
            pending.push(b);
            pending.push(p);
        } else {
            found.push(b);
        }
    }

    found
}

/// Kruskal with edges in `order`; `None` if the graph is disconnected.
fn kruskal<C: Cost, const N: usize>(
    nodes: usize,
    edges: &[TreeEdge<C, N>],
    order: impl Fn(&TreeEdge<C, N>, &TreeEdge<C, N>) -> Ordering,
) -> Option<Tree<C, N>> {
    let mut sorted: Vec<usize> = (0..edges.len()).collect();
    sorted.sort_by(|&x, &y| order(&edges[x], &edges[y]).then(x.cmp(&y)));
    let mut sets = Sets::new(nodes);
    let mut tree = Vec::new();
    let mut cost = [C::ZERO; N];
    for e in sorted {
        if sets.union(edges[e].0, edges[e].1) {
            tree.push(e);
            cost = add(cost, edges[e].2);
        }
    }

    (tree.len() + 1 >= nodes).then(|| {
        tree.sort_unstable();
        (tree, cost)
    })
}

/// Branch and bound into `front`, keeping only partial trees whose lower
/// bound `inside` accepts.
fn search<C: Cost, const N: usize>(
    nodes: usize,
    edges: &[TreeEdge<C, N>],
    front: &mut Vec<Tree<C, N>>,
    inside: &dyn Fn(&[C; N]) -> bool,
) {
    let mut chosen = Vec::new();
    descend(
        nodes,
        edges,
        0,
        Sets::new(nodes),
        &mut chosen,
        [C::ZERO; N],
        front,
        inside,
    );
}

/// One branching step at edge `k`.
#[expect(clippy::too_many_arguments, reason = "the branch-and-bound state")]
fn descend<C: Cost, const N: usize>(
    nodes: usize,
    edges: &[TreeEdge<C, N>],
    k: usize,
    sets: Sets,
    chosen: &mut Vec<usize>,
    cost: [C; N],
    front: &mut Vec<Tree<C, N>>,
    inside: &dyn Fn(&[C; N]) -> bool,
) {
    if chosen.len() + 1 >= nodes {
        if !front.iter().any(|t| weakly_dominates(&t.1, &cost)) {
            front.retain(|t| !weakly_dominates(&cost, &t.1));
            front.push((chosen.clone(), cost));
        }

        return;
    }

    let Some(bound) = lower_bound(nodes, edges, k, &sets, cost) else {
        return;
    };
    if !inside(&bound) || front.iter().any(|t| weakly_dominates(&t.1, &bound)) {
        return;
    }

    let (u, v, w) = edges[k];
    let mut with = sets.clone();
    if with.union(u, v) {
        chosen.push(k);
        descend(
            nodes,
            edges,
            k + 1,
            with,
            chosen,
            add(cost, w),
            front,
            inside,
        );
        chosen.pop();
    }

    descend(nodes, edges, k + 1, sets, chosen, cost, front, inside);
}

/// `cost` plus, per objective, the cheapest way to finish the tree with
/// edges `k..`; `None` if they cannot.
fn lower_bound<C: Cost, const N: usize>(
    nodes: usize,
    edges: &[TreeEdge<C, N>],
    k: usize,
    sets: &Sets,
    cost: [C; N],
) -> Option<[C; N]> {
    let mut bound = cost;
    for i in 0..N {
        let mut rest: Vec<usize> = (k..edges.len()).collect();
        rest.sort_by_key(|&e| edges[e].2[i]);
        let mut s = sets.clone();
        for e in rest {
            if s.union(edges[e].0, edges[e].1) {
                bound[i] = bound[i] + edges[e].2[i];
            }
        }

        if s.components(nodes) > 1 {
            return None;
        }
    }

    Some(bound)
}

/// Sorts trees by cost.
fn sorted<C: Cost, const N: usize>(mut trees: Vec<Tree<C, N>>) -> Vec<Tree<C, N>> {
    trees.sort_by_key(|t| t.1);
    trees.dedup_by_key(|t| t.1);
    trees
}

/// A union-find over nodes `0..n`.
#[derive(Debug, Clone)]
struct Sets(Vec<usize>);

impl Sets {
    /// `n` singletons.
    fn new(n: usize) -> Self {
        Self((0..n).collect())
    }

    /// The representative of `x`.
    fn find(&self, mut x: usize) -> usize {
        while self.0[x] != x {
            x = self.0[x];
        }

        x
    }

    /// Joins the sets of `a` and `b`; `false` if already joined.
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }

        self.0[ra] = rb;
        true
    }

    /// The number of sets among the first `n` nodes.
    fn components(&self, n: usize) -> usize {
        (0..n).filter(|&x| self.find(x) == x).count()
    }
}
