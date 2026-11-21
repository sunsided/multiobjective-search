//! Fuzzes [`Epsilon::within`] against an exact `u128` oracle for `u32` and
//! `u64` costs: it never accepts a pair outside `1 + ε`, and it is exact
//! whenever `(num + den) · y` fits the cost type.

#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;
use multiobjective_search::Epsilon;

/// Checks one comparison `x <= (1 + num / den) · y` on a cost type with
/// maximum `max`.
fn check(epsilon: Epsilon, x: u128, y: u128, within: bool, max: u128) {
    let den = u128::from(epsilon.den());
    let factor = den + u128::from(epsilon.num());
    let exact = den * x <= factor * y;

    // Sound: the saturating fallback only ever tightens the test to x <= y.
    assert!(
        !within || exact,
        "accepted {x} > (1 + ε) · {y} for {epsilon:?}"
    );

    // Complete while neither product saturates.
    if factor * y < max && den * x < max {
        assert_eq!(within, exact, "x = {x}, y = {y}, {epsilon:?}");
    }
}

fuzz_target!(|input: (u32, u32, u32, u32, u64, u64)| {
    let (num, den, x32, y32, x, y) = input;
    let Some(den) = den.checked_add(1) else {
        return;
    };

    if num.checked_add(den).is_none() {
        return;
    }

    let epsilon = Epsilon::new(num, den);

    let within = epsilon.within(x32, y32);
    check(epsilon, x32.into(), y32.into(), within, u32::MAX.into());

    let within = epsilon.within(x, y);
    check(epsilon, x.into(), y.into(), within, u64::MAX.into());
});
