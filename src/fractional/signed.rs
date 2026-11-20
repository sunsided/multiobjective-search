//! A small signed 256-bit integer for exact Dinkelbach weights.

use core::cmp::Ordering;

/// A signed integer in two's complement over four 64-bit limbs, least
/// significant first. Enough for `den · c1 - num · c2` summed along a path
/// with 64-bit costs and 128-bit `num` and `den`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Signed([u64; 4]);

impl Signed {
    /// Zero.
    pub(super) const ZERO: Self = Self([0; 4]);

    /// The non-negative product `a · b`.
    pub(super) fn scaled(a: u128, b: u64) -> Self {
        let mask = u128::from(u64::MAX);
        let (lo, hi) = ((a & mask) * u128::from(b), (a >> 64) * u128::from(b));
        let mid = (lo >> 64) + (hi & mask);
        let top = (hi >> 64) + (mid >> 64);
        let limb = |v: u128| u64::try_from(v & mask).expect("masked");

        Self([limb(lo), limb(mid), limb(top), limb(top >> 64)])
    }

    /// `self + other`, wrapping (no overflow in this crate's use).
    pub(super) fn plus(self, other: Self) -> Self {
        let mut out = [0_u64; 4];
        let mut carry = false;
        for (i, slot) in out.iter_mut().enumerate() {
            let (s, c1) = self.0[i].overflowing_add(other.0[i]);
            let (s, c2) = s.overflowing_add(u64::from(carry));
            *slot = s;
            carry = c1 || c2;
        }

        Self(out)
    }

    /// `self - other`.
    pub(super) fn minus(self, other: Self) -> Self {
        let negated = Self(other.0.map(|l| !l)).plus(Self([1, 0, 0, 0]));
        self.plus(negated)
    }

    /// `true` if negative.
    const fn negative(self) -> bool {
        self.0[3] >> 63 == 1
    }
}

impl PartialOrd for Signed {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Signed {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative(), other.negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            // Same sign: two's complement compares as unsigned.
            _ => self.0.iter().rev().cmp(other.0.iter().rev()),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`Signed`].

    use super::Signed;

    /// Sums and differences order like integers.
    #[test]
    fn arithmetic() {
        let three = Signed::scaled(3, 1);
        let five = Signed::scaled(1, 5);
        assert!(three.minus(five) < Signed::ZERO);
        assert_eq!(three.minus(five).plus(five), three);
        assert!(Signed::scaled(u128::MAX, u64::MAX) > Signed::scaled(u128::MAX, 1));
        assert!(Signed::ZERO.minus(Signed::scaled(u128::MAX, u64::MAX)) < Signed::ZERO.minus(five));
    }
}
