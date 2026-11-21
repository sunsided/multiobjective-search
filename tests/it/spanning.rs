//! Multi-objective spanning tree tests against enumeration of all edge
//! subsets.

use multiobjective_search::{
    pareto_spanning_trees, supported_spanning_trees, two_phase_spanning_trees,
};
use proptest::prelude::*;

/// Every spanning tree's cost by enumerating `nodes - 1` edge subsets.
fn all_trees<const N: usize>(nodes: usize, edges: &[(usize, usize, [u32; N])]) -> Vec<[u32; N]> {
    let mut out = Vec::new();
    for mask in 0..1_u32 << edges.len() {
        if mask.count_ones() as usize + 1 != nodes.max(1) {
            continue;
        }

        let mut parent: Vec<usize> = (0..nodes).collect();
        let find = |p: &Vec<usize>, mut x: usize| {
            while p[x] != x {
                x = p[x];
            }
            x
        };
        let mut ok = true;
        let mut cost = [0; N];
        for (i, &(u, v, c)) in edges.iter().enumerate() {
            if mask >> i & 1 == 1 {
                let (a, b) = (find(&parent, u), find(&parent, v));
                if a == b {
                    ok = false;
                }
                parent[a] = b;
                for k in 0..N {
                    cost[k] += c[k];
                }
            }
        }

        if ok {
            out.push(cost);
        }
    }

    out
}

/// The Pareto front of `points`, sorted, without duplicates.
fn pareto<const N: usize>(points: &[[u32; N]]) -> Vec<[u32; N]> {
    let mut front: Vec<_> = points
        .iter()
        .copied()
        .filter(|a| {
            !points
                .iter()
                .any(|b| b != a && (0..N).all(|k| b[k] <= a[k]))
        })
        .collect();
    front.sort_unstable();
    front.dedup();
    front
}

/// Lower-left hull vertices of a sorted 2D front.
fn hull(front: &[[u32; 2]]) -> Vec<[u32; 2]> {
    let cross = |o: [u32; 2], a: [u32; 2], b: [u32; 2]| {
        (i64::from(a[0]) - i64::from(o[0])) * (i64::from(b[1]) - i64::from(o[1]))
            - (i64::from(a[1]) - i64::from(o[1])) * (i64::from(b[0]) - i64::from(o[0]))
    };
    let mut h: Vec<[u32; 2]> = Vec::new();
    for &p in front {
        while h.len() >= 2 && cross(h[h.len() - 2], h[h.len() - 1], p) <= 0 {
            h.pop();
        }
        h.push(p);
    }
    h
}

/// Random graphs over up to five nodes and nine edges.
fn graphs<const N: usize>() -> impl Strategy<Value = (usize, Vec<(usize, usize, [u32; N])>)> {
    (1..=5_usize).prop_flat_map(|n| {
        let edge = (0..n, 0..n, prop::array::uniform(0..=6_u32));
        (Just(n), prop::collection::vec(edge, 0..=9))
    })
}

proptest! {
    /// Branch and bound, the two-phase method and its first phase against
    /// enumeration.
    #[test]
    fn spanning_trees_match_enumeration((n, edges) in graphs::<2>(), (m, edges3) in graphs::<3>()) {
        let front = pareto(&all_trees(n, &edges));
        let costs = |t: &[(Vec<usize>, [u32; 2])]| t.iter().map(|x| x.1).collect::<Vec<_>>();
        let exact = pareto_spanning_trees(n, &edges);
        prop_assert_eq!(costs(&exact), front.clone());
        prop_assert_eq!(costs(&two_phase_spanning_trees(n, &edges)), front.clone());
        prop_assert_eq!(costs(&supported_spanning_trees(n, &edges)), hull(&front));
        for (tree, c) in &exact {
            let sum = tree.iter().fold([0, 0], |s, &e| [s[0] + edges[e].2[0], s[1] + edges[e].2[1]]);
            prop_assert_eq!(sum, *c);
            prop_assert_eq!(tree.len() + 1, n);
        }

        let front3 = pareto(&all_trees(m, &edges3));
        let got: Vec<_> = pareto_spanning_trees(m, &edges3).iter().map(|x| x.1).collect();
        prop_assert_eq!(got, front3);
    }
}
