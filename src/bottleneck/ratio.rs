//! Objectives derived from the minsum–maxmin front: the minimal
//! cost/capacity ratio and the quickest path.

use core::cmp::Ordering;

use search_core::{Cost, ExactArith, ExactNumber, Heuristic, WideMul};

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
/// comparing `(d1 · c1 + σ) · c2` with `(d2 · c2 + σ) · c1` exactly through
/// [`ExactArith`]. A zero capacity makes the time
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
    C: Cost + ExactArith,
    P: Fn(&G::State) -> bool,
{
    let front = minsum_maxmin(graph, heuristic, start, is_goal).solutions;
    front
        .into_iter()
        .min_by(|a, b| time_order(a.cost, b.cost, sigma).then(a.cost.0.cmp(&b.cost.0)))
}

/// Orders `d + σ / c` exactly; `None` capacity is unbounded, a zero one
/// infinite unless `σ` is zero.
fn time_order<C: Cost + ExactArith>(
    a: (C, Bottleneck<C>),
    b: (C, Bottleneck<C>),
    sigma: C,
) -> Ordering {
    let infinite = |c: Bottleneck<C>| sigma > C::ZERO && c == Some(C::ZERO);
    match (infinite(a.1), infinite(b.1)) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        (false, false) => {}
    }

    // The time as a fraction: d / 1, or (d · c + σ) / c.
    let time = |(d, c): (C, Bottleneck<C>)| match c.filter(|_| sigma > C::ZERO) {
        None => (d.exact(), C::Exact::one()),
        Some(c) => (d.exact() * c.exact() + sigma.exact(), c.exact()),
    };
    let ((na, da), (nb, db)) = (time(a), time(b));

    (na * db).cmp(&(nb * da))
}

#[cfg(test)]
mod tests {
    //! Unit tests for the exact time comparison.

    use core::cmp::Ordering;

    use super::time_order;

    /// `1 + 5/1 = 6 < 10 + 5/100`, and the reverse for 1000 units; zero
    /// capacities are infinite only with data to send; extreme values stay
    /// exact.
    #[test]
    fn times_compare() {
        assert_eq!(
            time_order((1_u64, Some(1)), (10, Some(100)), 5),
            Ordering::Less
        );
        assert_eq!(
            time_order((1_u64, Some(1)), (10, Some(100)), 1000),
            Ordering::Greater
        );
        assert_eq!(
            time_order((0_u64, Some(0)), (1, None), 1),
            Ordering::Greater
        );
        assert_eq!(
            time_order((4_u64, Some(0)), (3, Some(7)), 0),
            Ordering::Greater
        );
        let max = u128::MAX;
        assert_eq!(
            time_order((max, Some(max)), (max - 1, Some(max)), max),
            Ordering::Greater
        );
    }
}
