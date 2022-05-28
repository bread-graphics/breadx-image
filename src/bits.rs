// MIT/Apache2 License

/// A series of bits that can be manipulated like a collection.
///
/// Notes:
///
/// - By default, bits are stored in LSB order. The normalize()
///   function can be used to swap them into MSB order.
/// - This means that conversion functions should use to_le_bytes().
pub(crate) struct Bitfield<T: ?Sized>(pub(crate) T);

impl<T: AsRef<[u8]> + ?Sized> Bitfield<T> {
    /// Get the bit at the given index.
    pub(crate) fn bit(&self, index: usize) -> bool {
        let (byte_index, bit_index) = bb_index(index);
        let byte = self.0.as_ref().get(byte_index).expect("out of range");
        let bit = 1 << bit_index;

        (byte & bit) != 0
    }

    /// Get the length of this bitfield.
    pub(crate) fn len(&self) -> usize {
        self.0.as_ref().len() * 8
    }
}

impl<T: AsRef<[u8]> + AsMut<[u8]> + ?Sized> Bitfield<T> {
    /// Set the bit at the given index.
    pub(crate) fn set_bit(&mut self, index: usize, val: bool) {
        let (byte_index, bit_index) = bb_index(index);
        let byte = self.0.as_mut().get_mut(byte_index).expect("out of range");
        let bit = 1 << bit_index;

        if val {
            *byte |= bit;
        } else {
            *byte &= !bit;
        }
    }

    /// XOR the bit at the given position with the given value.
    pub(crate) fn xor_bit(&mut self, index: usize, val: bool) {
        let (byte_index, bit_index) = bb_index(index);
        let byte = self.0.as_mut().get_mut(byte_index).expect("out of range");
        let bit = 1 << bit_index;

        if val {
            *byte ^= bit;
        }
    }

    /// Copy bits from one to the other.
    pub(crate) fn copy_from(
        &mut self,
        other: &Bitfield<impl AsRef<[u8]>>,
        mut self_start: usize,
        mut other_start: usize,
        mut len: usize,
    ) {
        // we may just have to preform a naive copy
        if len < 8 || self_start % 8 != other_start % 8 {
            self.naive_copy(other, self_start, other_start, len);
            return;
        }

        // the following algorithm is much faster, especially for
        // the common case where data is aligned on a byte boundary
        // wins of up to 50% have been recorded

        // preform a naive copy until we arrive at the
        // boundary of the byte
        let initial_len = distance_to_byte_boundary(self_start);
        self.naive_copy(other, self_start, other_start, initial_len);
        self_start += initial_len;
        other_start += initial_len;
        len -= initial_len;

        // use copy_from_slice() to preform an optimized copy
        let byte_len = len / 8;
        let self_byte_start = self_start / 8;
        let other_byte_start = other_start / 8;
        let self_range = self_byte_start..(self_byte_start + byte_len);
        let other_range = other_byte_start..(other_byte_start + byte_len);

        self.0.as_mut()[self_range].copy_from_slice(&other.0.as_ref()[other_range]);

        // copy the remaining bits
        let remaining_len = len % 8;
        let self_start = self_start + byte_len * 8;
        let other_start = other_start + byte_len * 8;
        self.naive_copy(other, self_start, other_start, remaining_len);
    }

    fn naive_copy(
        &mut self,
        other: &Bitfield<impl AsRef<[u8]>>,
        self_start: usize,
        other_start: usize,
        len: usize,
    ) {
        for i in 0..len {
            let self_index = self_start + i;
            let other_index = other_start + i;

            self.set_bit(self_index, other.bit(other_index));
        }
    }

    /// Swap two bits in the value.
    pub(crate) fn swap_bits(&mut self, index1: usize, index2: usize) {
        if index1 == index2 {
            return;
        }

        let bit1 = self.bit(index1);
        let bit2 = self.bit(index2);

        // xor the two bits
        let xor_bit = bit1 ^ bit2;

        // set the bits
        self.xor_bit(index1, xor_bit);
        self.xor_bit(index2, xor_bit);
    }

    /// Swap an entire range of bits.
    ///
    /// It is assumed that the ranges do not overlap.
    pub(crate) fn swap_range(&mut self, start1: usize, start2: usize, len: usize) {
        // assert that the two ranges do not overlap
        assert!(start1 + len <= start2);

        for i in 0..len {
            let index1 = start1 + i;
            let index2 = start2 + i;

            self.swap_bits(index1, index2);
        }
    }

    /// Swap all bits in the given range so that the bits are in MSB order.
    pub(crate) fn msbify(&mut self, start: usize, len: usize) {
        // TODO: this function uses a very naive implementation
        // could be sped up significantly
        let end = start + len;

        for i in 0..len / 2 {
            let index = end - i - 1;

            self.swap_bits(start + i, index);
        }
    }
}

/// Given an index into this bitfield's bits, get the index of
/// the byte that contains that bit as well as the index of the
/// bit within.
fn bb_index(index: usize) -> (usize, usize) {
    let byte_index = index / 8;
    let bit_index = index % 8;
    (byte_index, bit_index)
}

/// Is an index at a byte boundary?
fn distance_to_byte_boundary(index: usize) -> usize {
    // get the number of bytes we need to add for index to be
    // a multiple of 8
    let remainder = index % 8;
    if remainder == 0 {
        0
    } else {
        8 - remainder
    }
}

#[allow(clippy::needless_range_loop)]
#[cfg(test)]
mod test {
    use super::Bitfield;
    use core::cmp;

    #[test]
    fn get_bit() {
        // get the bits we're calculating
        let number = fastrand::u64(..);
        let mut bits = [false; 64];

        for i in 0..64 {
            bits[i] = (number & (1 << i)) != 0;
        }

        // create a bitfield from the number
        let bitfield = Bitfield(number.to_le_bytes());

        // check that the bits are correct
        for i in 0..64 {
            assert_eq!(bits[i], bitfield.bit(i));
        }
    }

    #[test]
    fn set_bit() {
        let number = fastrand::u64(..);
        let mut bits = [false; 64];
        let mut changed_bits = [None; 64];

        // set bits as well as the bits to change
        for i in 0..64 {
            bits[i] = (number & (1 << i)) != 0;
            if fastrand::bool() {
                changed_bits[i] = Some(fastrand::bool());
            }
        }

        let mut bitfield = Bitfield(number.to_le_bytes());
        for (i, changed_bit) in (&changed_bits)
            .iter()
            .enumerate()
            .flat_map(|(i, b)| b.map(move |b| (i, b)))
        {
            bitfield.set_bit(i, changed_bit);
        }

        for i in 0..64 {
            let mut bit = bits[i];
            if let Some(changed) = changed_bits[i] {
                bit = changed;
            }

            assert_eq!(bit, bitfield.bit(i));
        }
    }

    #[test]
    fn copy_from() {
        let number1 = fastrand::u64(..);
        let number2 = fastrand::u64(..);

        let mut bits1 = [false; 64];
        let mut bits2 = [false; 64];

        for i in 0..64 {
            bits1[i] = (number1 & (1 << i)) != 0;
            bits2[i] = (number2 & (1 << i)) != 0;
        }

        // copy some random amount of bytes from number2 to number1
        let n1start = fastrand::usize(0..63);
        let n2start = fastrand::usize(0..63);
        let len = fastrand::usize(0..cmp::min(62 - n1start, 62 - n2start));

        let mut bitfield1 = Bitfield(number1.to_le_bytes());
        let bitfield2 = Bitfield(number2.to_le_bytes());

        // copy range
        bitfield1.copy_from(&bitfield2, n1start, n2start, len);

        // check
        let modrange = n1start..n1start + len;
        for i in 0..64 {
            if modrange.contains(&i) {
                assert_eq!(bits2[n2start + (i - n1start)], bitfield1.bit(i));
            } else {
                assert_eq!(bits1[i], bitfield1.bit(i));
            }
        }
    }

    #[test]
    fn swap_bits() {
        let number1 = fastrand::u64(..);
        let mut bits = [false; 64];

        // take two indices to swap around
        let mut index1;
        let mut index2;
        loop {
            index1 = fastrand::usize(0..64);
            index2 = fastrand::usize(0..64);
            if index1 != index2 {
                break;
            }
        }

        for i in 0..64 {
            bits[i] = (number1 & (1 << i)) != 0;
        }

        bits.swap(index1, index2);

        // use the bitfield
        let mut bitfield = Bitfield(number1.to_le_bytes());
        bitfield.swap_bits(index1, index2);

        for i in 0..64 {
            assert_eq!(bits[i], bitfield.bit(i));
        }
    }

    #[test]
    fn swap_range() {
        let number = fastrand::u64(..);
        let mut bits = [false; 64];

        // generate two ranges to swap
        let len = fastrand::usize(1..31);
        let start1 = fastrand::usize(0..32 - len);
        let start2 = fastrand::usize(32..64 - len);

        for i in 0..64 {
            bits[i] = (number & (1 << i)) != 0;
        }

        for i in 0..len {
            let index1 = start1 + i;
            let index2 = start2 + i;
            bits.swap(index1, index2);
        }

        // use the bitfield
        let mut bitfield = Bitfield(number.to_le_bytes());
        bitfield.swap_range(start1, start2, len);

        for i in 0..64 {
            assert_eq!(bits[i], bitfield.bit(i));
        }
    }

    #[test]
    fn msbify() {
        let number = fastrand::u64(..);
        let mut bits = [false; 64];

        for i in 0..64 {
            bits[i] = (number & (1 << i)) != 0;
        }

        bits.reverse();

        let mut bitfield = Bitfield(number.to_le_bytes());
        bitfield.msbify(0, 64);

        for i in 0..64 {
            assert_eq!(bits[i], bitfield.bit(i));
        }
    }
}
