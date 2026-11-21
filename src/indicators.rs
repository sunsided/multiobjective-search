//! Quality indicators for approximate and partial Pareto fronts.

use alloc::vec::Vec;

use search_core::{BigUint, ExactArith, ExactNumber};

/// The hypervolume of `points`: the volume of the region they dominate
/// below `reference`, exactly (Zitzler and Thiele; the convergence-
/// diversity indicator Domínguez-Ríos et al. use to compare anytime
/// fronts). Points not strictly below `reference` in every objective add
/// nothing.
///
/// Computed by slicing along the last objective, which is exponential in
/// the number of objectives but exact; meant for evaluation, not for the
/// inner loop of a search.
///
/// # Examples
///
/// ```
/// use multiobjective_search::hypervolume;
/// use search_core::BigUint;
///
/// // Two points below (10, 10): [2,10)×[8,10) and [6,10)×[3,10) overlap in [6,10)×[8,10).
/// let hv = hypervolume(&[[2_u32, 8], [6, 3]], [10, 10]);
/// assert_eq!(hv, BigUint::from(16 + 28 - 8));
/// ```
#[must_use]
pub fn hypervolume<C: ExactArith<Exact = BigUint>, const N: usize>(
    points: &[[C; N]],
    reference: [C; N],
) -> BigUint {
    let inside: Vec<Vec<BigUint>> = points
        .iter()
        .filter(|p| (0..N).all(|i| p[i] < reference[i]))
        .map(|p| p.iter().map(|c| c.exact()).collect())
        .collect();
    let reference: Vec<BigUint> = reference.iter().map(|c| c.exact()).collect();

    slices(inside, &reference)
}

/// The hypervolume of `points` (all below `reference`) in `reference.len()`
/// dimensions.
fn slices(mut points: Vec<Vec<BigUint>>, reference: &[BigUint]) -> BigUint {
    let d = reference.len();
    if points.is_empty() || d == 0 {
        return if points.is_empty() {
            BigUint::zero()
        } else {
            BigUint::one()
        };
    }

    points.sort_by(|a, b| a[d - 1].cmp(&b[d - 1]));
    let mut total = BigUint::zero();
    for k in 0..points.len() {
        let next = points.get(k + 1).map_or(&reference[d - 1], |p| &p[d - 1]);
        let height = next.abs_diff(&points[k][d - 1]);
        if height == BigUint::zero() {
            continue;
        }

        let base: Vec<Vec<BigUint>> = points[..=k].iter().map(|p| p[..d - 1].to_vec()).collect();
        total += slices(base, &reference[..d - 1]) * height;
    }

    total
}
