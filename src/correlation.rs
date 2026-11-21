//! Edge-level correlation between objectives, the measure the benchmark
//! suite of Peer et al. (arXiv 2603.24084, Section 2) uses to describe how
//! redundant two objectives are, after Halle et al. 2025.

use search_core::{BigUint, ExactArith, ExactNumber};

/// The Pearson correlation of two objectives over a set of edges, kept
/// exact: its sign and its square as a fraction.
///
/// `ρ = cov / sqrt(var_i · var_j)`, so `ρ² = cov² / (var_i · var_j)`, all
/// computed from the integer sums `n Σxy - Σx Σy`, `n Σx² - (Σx)²`
/// without rounding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Correlation {
    /// `true` for `ρ < 0`.
    pub negative: bool,
    /// `cov²` (times `n²`).
    pub numerator: BigUint,
    /// `var_i · var_j` (times `n²`); zero when an objective is constant,
    /// where the correlation is undefined.
    pub denominator: BigUint,
}

impl Correlation {
    /// Returns `true` if `|ρ| >= num / den`; `false` when undefined.
    #[must_use]
    pub fn at_least(&self, num: u64, den: u64) -> bool {
        if self.denominator == BigUint::zero() {
            return false;
        }

        let (num, den) = (
            BigUint::from(u128::from(num)),
            BigUint::from(u128::from(den)),
        );
        self.numerator.clone() * den.clone() * den >= num.clone() * num * self.denominator.clone()
    }
}

/// The correlation of objectives `a` and `b` over the cost vectors of
/// `edges`, for integer and decimal costs.
///
/// Correlation detection only: the survey of Salzman et al. and the
/// benchmark paper do not define how correlated objectives would be
/// merged, so no merging step is offered.
///
/// # Examples
///
/// ```
/// use multiobjective_search::objective_correlation;
///
/// let edges = [[1_u32, 2, 5], [2, 4, 4], [3, 6, 1]];
/// let r = objective_correlation(edges.iter().copied(), 0, 1);
/// assert!(!r.negative && r.at_least(1, 1)); // perfectly correlated
/// let r = objective_correlation(edges.iter().copied(), 0, 2);
/// assert!(r.negative && r.at_least(9, 10));
/// ```
///
/// # Panics
///
/// Panics if `a` or `b` is out of range.
pub fn objective_correlation<C: ExactArith<Exact = BigUint>, const N: usize>(
    edges: impl IntoIterator<Item = [C; N]>,
    a: usize,
    b: usize,
) -> Correlation {
    let zero = BigUint::zero;
    let (mut count, mut sx, mut sy, mut sxy, mut sxx, mut syy) =
        (0_u128, zero(), zero(), zero(), zero(), zero());
    for cost in edges {
        let (first, second) = (cost[a].exact(), cost[b].exact());
        count += 1;
        sxy = sxy + first.clone() * second.clone();
        sxx = sxx + first.clone() * first.clone();
        syy = syy + second.clone() * second.clone();
        sx = sx + first;
        sy = sy + second;
    }

    let n = BigUint::from(count);
    let (products, means) = (n.clone() * sxy, sx.clone() * sy.clone());
    let cov = products.abs_diff(&means);
    let var_x = (n.clone() * sxx).abs_diff(&(sx.clone() * sx));
    let var_y = (n * syy).abs_diff(&(sy.clone() * sy));

    Correlation {
        negative: products < means,
        numerator: cov.clone() * cov,
        denominator: var_x * var_y,
    }
}
