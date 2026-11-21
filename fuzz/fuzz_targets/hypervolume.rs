//! Fuzzes [`hypervolume`] in three objectives against cell counting on a
//! small grid: with integer coordinates below `REFERENCE`, the hypervolume
//! is the number of unit cells weakly dominated by some point.

#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;
use multiobjective_search::hypervolume;

/// The reference point's coordinate in every objective.
const REFERENCE: u32 = 12;

/// Counts the unit cells `[c, c + 1)^3` below the reference that some point
/// weakly dominates.
fn dominated_cells(points: &[[u32; 3]]) -> u128 {
    let mut count = 0;
    for x in 0..REFERENCE {
        for y in 0..REFERENCE {
            for z in 0..REFERENCE {
                let cell = [x, y, z];
                if points.iter().any(|p| (0..3).all(|i| p[i] <= cell[i])) {
                    count += 1;
                }
            }
        }
    }

    count
}

fuzz_target!(|raw: Vec<[u8; 3]>| {
    // Coordinates up to REFERENCE + 3, so some points lie outside and add nothing.
    let points: Vec<[u32; 3]> = raw
        .iter()
        .take(32)
        .map(|p| p.map(|c| u32::from(c) % (REFERENCE + 4)))
        .collect();

    let expected = dominated_cells(&points);
    assert!(
        hypervolume(&points, [REFERENCE; 3]) == expected.into(),
        "hypervolume of {points:?}"
    );
});
