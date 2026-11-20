//! Objectives derived from the minsum–maxmin front: the minimal
//! cost/capacity ratio and the quickest path.

use core::cmp::Ordering;

use search_core::{Cost, Heuristic, WideMul};

use crate::bottleneck::{Bottleneck, minsum_maxmin};
use crate::graph::Graph;
use crate::output::Solution;

/// A path with its cost and bottleneck capacity.
pub type WidePath<S, C> = Solution<S, (C, Bottleneck<C>)>;

/// The path from `start` to a goal with the smallest ratio of its cost (the
/// sum of the edges' first components) to its bottleneck capacity (the
/// smallest second component); `None` if no goal is reachable.
///
/// Martins 1984 (surveyed by Craveirinha et al., Algorithms 17(6):222,
/// 2024, Section 2.2.3(a)): a path that another beats in both cost and
/// capacity has no smaller ratio, so the optimum lies on the
/// [`minsum_maxmin`] front, which is scanned with exact cross-multiplied
/// comparisons. A zero capacity makes the ratio infinite; the empty path
/// (start is a goal) has ratio zero. Ties go to the cheaper path.
///
/// Requirements as for [`minsum_maxmin`].
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, heuristic_fn, min_cost_capacity_ratio};
///
/// /// Two routes from 0 to 1: (cost, capacity).
/// struct Links;
///
/// impl Graph for Links {
///     type State = u8;
///     type Weight = (u32, u32);
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, (u32, u32))>) {
///         if *state == 0 {
///             out.extend([(1, (2, 4)), (1, (5, 20))]);
///         }
///     }
/// }
///
/// let best = min_cost_capacity_ratio(&Links, heuristic_fn(|_: &u8| Some(0)), 0, |s| *s == 1);
/// assert_eq!(best.map(|s| s.cost), Some((5, Some(20)))); // 5/20 < 2/4
/// ```
pub fn min_cost_capacity_ratio<G, H, C, P>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
) -> Option<WidePath<G::State, C>>
where
    G: Graph<Weight = (C, C)>,
    H: Heuristic<G::State, C>,
    C: Cost + WideMul,
    P: Fn(&G::State) -> bool,
{
    let front = minsum_maxmin(graph, heuristic, start, is_goal).solutions;
    front
        .into_iter()
        .min_by(|a, b| ratio_order(a.cost, b.cost).then(a.cost.0.cmp(&b.cost.0)))
}

/// Orders `sum / width` ratios exactly: `None` width is unbounded (ratio
/// 0), zero width with a positive sum is infinite.
fn ratio_order<C: Cost + WideMul>(a: (C, Bottleneck<C>), b: (C, Bottleneck<C>)) -> Ordering {
    match (a.1, b.1) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(wa), Some(wb)) => a.0.wide_mul(wb).cmp(&b.0.wide_mul(wa)),
    }
}

/// The quickest path for sending `sigma` units from `start` to a goal: the
/// smallest transmission time `delay + sigma / capacity`, with the delay the
/// sum of the edges' first components and the capacity the smallest second
/// component; `None` if no goal is reachable.
///
/// The quickest path problem (Moore 1976; Chen and Chin 1990; surveyed by
/// Craveirinha et al., Section 2.2.3(d)). Its optimum lies on the
/// [`minsum_maxmin`] front (Pelegrín and Fernández 1998), which is scanned
/// comparing `(d1 · c1 + σ) · c2` with `(d2 · c2 + σ) · c1` exactly in 256-bit
/// arithmetic, hence the `Into<u64>` bound. A zero capacity makes the time
/// infinite unless `sigma` is zero; ties go to the smaller delay.
///
/// Requirements as for [`minsum_maxmin`].
///
/// # Examples
///
/// ```
/// use multiobjective_search::{Graph, heuristic_fn, quickest_path};
///
/// /// A short thin link and a long fat one from 0 to 1: (delay, capacity).
/// struct Links;
///
/// impl Graph for Links {
///     type State = u8;
///     type Weight = (u32, u32);
///
///     fn successors(&self, state: &u8, out: &mut Vec<(u8, (u32, u32))>) {
///         if *state == 0 {
///             out.extend([(1, (1, 1)), (1, (10, 100))]);
///         }
///     }
/// }
///
/// let zero = heuristic_fn(|_: &u8| Some(0));
/// // 5 units: 1 + 5/1 = 6 beats 10 + 5/100.
/// assert_eq!(quickest_path(&Links, zero, 0, |s| *s == 1, 5).map(|s| s.cost.0), Some(1));
/// // 1000 units: 10 + 10 = 20 beats 1 + 1000.
/// assert_eq!(quickest_path(&Links, zero, 0, |s| *s == 1, 1000).map(|s| s.cost.0), Some(10));
/// ```
pub fn quickest_path<G, H, C, P>(
    graph: &G,
    heuristic: H,
    start: G::State,
    is_goal: P,
    sigma: C,
) -> Option<WidePath<G::State, C>>
where
    G: Graph<Weight = (C, C)>,
    H: Heuristic<G::State, C>,
    C: Cost + Into<u64>,
    P: Fn(&G::State) -> bool,
{
    let sigma: u64 = sigma.into();
    let front = minsum_maxmin(graph, heuristic, start, is_goal).solutions;
    front.into_iter().min_by(|a, b| {
        let (da, db) = (a.cost.0.into(), b.cost.0.into());
        let (ca, cb) = (a.cost.1.map(Into::into), b.cost.1.map(Into::into));
        time_order(da, ca, db, cb, sigma).then(da.cmp(&db))
    })
}

/// Orders `d + σ / c` exactly; `None` capacity is unbounded.
fn time_order(da: u64, ca: Option<u64>, db: u64, cb: Option<u64>, sigma: u64) -> Ordering {
    let infinite = |c: Option<u64>| sigma > 0 && c == Some(0);
    match (infinite(ca), infinite(cb)) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        (false, false) => {}
    }

    // With σ = 0, or an unbounded capacity, the time is the delay alone.
    let (ca, cb) = match (ca.filter(|_| sigma > 0), cb.filter(|_| sigma > 0)) {
        (None, None) => return da.cmp(&db),
        (Some(ca), None) => {
            return (u128::from(da) * u128::from(ca) + u128::from(sigma))
                .cmp(&(u128::from(db) * u128::from(ca)));
        }
        (None, Some(cb)) => {
            return (u128::from(da) * u128::from(cb))
                .cmp(&(u128::from(db) * u128::from(cb) + u128::from(sigma)));
        }
        (Some(ca), Some(cb)) => (ca, cb),
    };

    // (da·ca + σ)/ca vs (db·cb + σ)/cb, cross-multiplied.
    let left = mul(u128::from(da) * u128::from(ca) + u128::from(sigma), cb);
    let right = mul(u128::from(db) * u128::from(cb) + u128::from(sigma), ca);
    left.cmp(&right)
}

/// The exact product of `a` and `b` as `(high, low)` 128-bit halves.
fn mul(a: u128, b: u64) -> (u128, u128) {
    let (hi, lo) = (a >> 64, a & u128::from(u64::MAX));
    let (hi, lo) = (hi * u128::from(b), lo * u128::from(b));
    let (low, carry) = (hi << 64).overflowing_add(lo);

    (hi >> 64)
        .checked_add(u128::from(carry))
        .map_or((u128::MAX, u128::MAX), |high| (high, low))
}

#[cfg(test)]
mod tests {
    //! Unit tests for the exact time comparison.

    use core::cmp::Ordering;

    use super::{mul, time_order};

    /// The 192-bit product matches small cases and the extreme.
    #[test]
    fn product_is_exact() {
        assert_eq!(mul(3, 5), (0, 15));
        assert_eq!(mul(u128::MAX, u64::MAX).0, u128::from(u64::MAX) - 1);
    }

    /// `1 + 5/1 = 6 < 10 + 5/100`, and the reverse for 1000 units.
    #[test]
    fn times_compare() {
        assert_eq!(time_order(1, Some(1), 10, Some(100), 5), Ordering::Less);
        assert_eq!(
            time_order(1, Some(1), 10, Some(100), 1000),
            Ordering::Greater
        );
        assert_eq!(time_order(0, Some(0), 1, None, 1), Ordering::Greater);
        assert_eq!(time_order(4, Some(0), 3, Some(7), 0), Ordering::Greater);
    }
}
