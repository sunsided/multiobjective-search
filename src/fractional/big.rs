//! A small unsigned multi-precision integer for exact reliability
//! products.

use alloc::vec::Vec;
use core::cmp::Ordering;

/// An unsigned integer as 32-bit limbs, least significant first, without
/// most significant zero limbs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Big(Vec<u32>);

impl From<u64> for Big {
    fn from(value: u64) -> Self {
        let low = u32::try_from(value & u64::from(u32::MAX)).expect("masked");
        let high = u32::try_from(value >> 32).expect("shifted");
        let mut limbs = alloc::vec![low, high];
        trim(&mut limbs);
        Self(limbs)
    }
}

impl Big {
    /// `self · other`.
    pub(super) fn times(&self, other: &Self) -> Self {
        let mut out = alloc::vec![0_u32; self.0.len() + other.0.len()];
        for (i, &a) in self.0.iter().enumerate() {
            let mut carry = 0_u64;
            for (j, &b) in other.0.iter().enumerate() {
                let t = u64::from(a) * u64::from(b) + u64::from(out[i + j]) + carry;
                out[i + j] = u32::try_from(t & u64::from(u32::MAX)).expect("masked");
                carry = t >> 32;
            }

            out[i + other.0.len()] = u32::try_from(carry).expect("below 2^32");
        }

        trim(&mut out);
        Self(out)
    }

    /// The limbs, least significant first.
    pub(super) fn limbs(self) -> Vec<u32> {
        self.0
    }
}

impl PartialOrd for Big {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Big {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}

/// Drops the most significant zero limbs.
fn trim(limbs: &mut Vec<u32>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`Big`].

    use super::Big;

    /// Products match `u128` and compare numerically.
    #[test]
    fn products() {
        let a = Big::from(u64::MAX);
        let square = a.times(&a);
        let expected = u128::from(u64::MAX) * u128::from(u64::MAX);
        let value = square
            .clone()
            .limbs()
            .iter()
            .rev()
            .fold(0_u128, |v, &l| (v << 32) | u128::from(l));
        assert_eq!(value, expected);
        assert!(square > a);
        assert_eq!(Big::from(0).times(&a), Big::from(0));
    }
}
