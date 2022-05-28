// MIT/Apache2 License

/// Reverse a slice of bytes in place.
/// 
/// This reversed the bits of all of the given bytes.
pub(crate) fn reverse_bytes(bytes: &mut [u8]) {
    bytes.iter_mut().for_each(|b| *b = REVERSED_BYTES[*b as usize]);
}

/// Get the `n` lowest bits of the `u32`.
pub(crate) fn low_bits(n: u32, bits: usize) -> u32 {
    n & LOW_BYTE_BITMASKS[bits]
}

/// An array of bytes whose bits are reversed.
/// 
/// The byte at index `i` is equivalent to `i` with its bits reversed.
const REVERSED_BYTES: [u8; 256] = {
    let mut reversed_bytes = [0u8; 256];

    // can't use for loops in const context yet
    let mut i = 0;
    while i < 256 {
        reversed_bytes[i] = reverse_bits(i as u8);
        i += 1;
    }

    reversed_bytes
};

/// An array of the bitmasks used to get the `n` lower bits of a byte.
const LOW_BYTE_BITMASKS: [u32; 33] = {
    let mut low_byte_bitmasks = [0u32; 33];
    let mut current = 0u32;
    let mut i = 0;

    while i < 33 {
        low_byte_bitmasks[i] = current;
        current = (current << 1) | 1;
        i += 1;
    }

    low_byte_bitmasks
};

/// Reverse the bits of a byte.
const fn reverse_bits(b: u8) -> u8 {
    let mut reversed = 0u8;

    let mut i = 0;
    while i < 8 {
        reversed |= ((b >> i) & 1) << (7 - i);
        i += 1;
    }

    reversed
}