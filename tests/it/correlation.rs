//! Correlation tests against a floating-point Pearson coefficient.

use multiobjective_search::{objective_correlation, random_grid};
use proptest::prelude::*;

/// Pearson's ρ in `f64`, `None` when an objective is constant.
fn pearson(edges: &[[u32; 3]], i: usize, j: usize) -> Option<f64> {
    let n = f64::from(u32::try_from(edges.len()).unwrap());
    let mean = |k: usize| edges.iter().map(|e| f64::from(e[k])).sum::<f64>() / n;
    let (mx, my) = (mean(i), mean(j));
    let cov: f64 = edges
        .iter()
        .map(|e| (f64::from(e[i]) - mx) * (f64::from(e[j]) - my))
        .sum();
    let var = |k: usize, m: f64| {
        edges
            .iter()
            .map(|e| (f64::from(e[k]) - m).powi(2))
            .sum::<f64>()
    };
    let d = (var(i, mx) * var(j, my)).sqrt();
    (d > 0.0).then(|| cov / d)
}

proptest! {
    /// Sign and thresholds agree with the floating-point coefficient.
    #[test]
    fn matches_pearson(edges in prop::collection::vec(prop::array::uniform3(0..20_u32), 1..40), t in 1..10_u64) {
        let exact = objective_correlation(edges.iter().copied(), 0, 1);
        match pearson(&edges, 0, 1) {
            None => prop_assert!(!exact.at_least(0, 1)),
            Some(rho) => {
                if rho.abs() > 1e-9 {
                    prop_assert_eq!(exact.negative, rho < 0.0);
                }
                let threshold = f64::from(u32::try_from(t).unwrap()) / 10.0;
                if (rho.abs() - threshold).abs() > 1e-9 {
                    prop_assert_eq!(exact.at_least(t, 10), rho.abs() >= threshold);
                }
            }
        }
    }
}

/// Random grids have nearly independent objectives (the paper reports
/// `|ρ| < 0.01` on its sizes).
#[test]
fn random_grids_are_uncorrelated() {
    let (grid, _) = random_grid::<3>(60, 60, 7);
    let r = objective_correlation(grid.arcs().map(|a| a.2), 0, 1);
    assert!(!r.at_least(1, 20));
}
