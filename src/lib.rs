// MIT/Apache2 License

//! A library for sending images over [`breadx`].

#![no_std]

extern crate alloc;

mod bits;
use bits::Bitfield;

mod byte_tables;
use byte_tables::{reverse_bytes, low_bits};

use breadx::{protocol::xproto::{ImageFormat, ImageOrder, PutImageRequest}, display::DisplayBase};

/*

Personal notes:

 */

/// The format that an image can have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Format {
    /// The image is in XY format. 
    /// 
    /// The image is serialized as a series of planes of pixels, 
    /// where one can determine the value by adding up the values 
    /// taken from the planes.
    Xy {
        /// The direct format of the image. 
        /// 
        /// This determines whether it is in bitmap format or
        /// pixmap format.
        format: XyFormatType,
        /// The quantum of the image.
        /// 
        /// Each scanline can be broken down into a series of values
        /// of this type. For instance, if the quantum was 32, you 
        /// could represent the scanline as a slice of `u32`s, although
        /// the padding may actually differ.
        quantum: u8,
        /// The bit order to be used in the image.
        /// 
        /// This determines whether the leftmost bit is either the least or most
        /// significant bit.
        bit_order: ImageOrder,
    },
    /// The image in in Z format.
    /// 
    /// The image is serialized such that each pixel exists in its entire bit
    /// pattern.
    Z {
        /// The depth of the image.
        depth: u8,
        /// The bits per pixel for this image.
        /// 
        /// This value is always greater than or equal to the depth. If it is
        /// larger, the least significant bits hold the data.
        bits_per_pixel: u8,
    }
}

/// Whether or not an XY image format is in bitmap or pixmap format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum XyFormatType {
    /// The image is in bitmap format.
    Bitmap,
    /// The image is in pixmap format, with the given depth.
    Pixmap {
        /// The depth of the image.
        depth: u8,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Image<Storage: ?Sized> {
    /// The width of the image.
    pub width: u16,
    /// The height of the image.
    pub height: u16,
    /// The format used in the image.
    pub image_format: Format,
    /// The left padding used in the image.
    pub left_pad: u8,
    /// The byte ordering used in the image.
    /// 
    /// This applies to the scanline units used in the Z pixmap format,
    /// or to the quantums used in the XY format.
    pub byte_order: ImageOrder,
    /// The scanline pad for each scanline, in bits.
    pub scanline_pad: u8,
    /// The storage system used to store bytes.
    pub storage: Storage,
}

impl<Storage: ?Sized> AsRef<Storage> for Image<Storage> {
    fn as_ref(&self) -> &Storage {
        &self.storage
    }
}

impl<Storage: ?Sized> AsMut<Storage> for Image<Storage> {
    fn as_mut(&mut self) -> &mut Storage {
        &mut self.storage
    }
}

impl<Storage: AsRef<[u8]>> Image<Storage> {
    /// Create a new `Image` from a `Storage` and its individual parameters.
    pub fn new(
        storage: Storage,
        width: u16,
        height: u16,
        format: Format,
        byte_order: ImageOrder,
        scanline_pad: u8,
    ) -> Self {
        Self {
            width,
            height,
            scanline_pad,
            image_format: format,
            left_pad: 0,
            byte_order,
            storage
        }
    }

    /// Create a new `Image`, but use a specific `Display` to set certain important values.
    pub fn with_display(
        storage: Storage,
        width: u16,
        height: u16,
        image_format: ImageFormat,
        depth: u8,
        display: &impl DisplayBase,
    ) -> breadx::Result<Self> {
        let setup = display.setup();
        let byte_order = setup.image_byte_order;
        let (format, scanline_pad) = match image_format {
            ImageFormat::XY_BITMAP | ImageFormat::XY_PIXMAP => (Format::Xy {
                format: match image_format {
                    ImageFormat::XY_BITMAP => XyFormatType::Bitmap,
                    ImageFormat::XY_PIXMAP => XyFormatType::Pixmap { depth },
                    _ => unreachable!(),
                },
                quantum: setup.bitmap_format_scanline_unit,
                bit_order: byte_order,
            }, setup.bitmap_format_scanline_pad),
            ImageFormat::Z_PIXMAP => {
                let (bpp, scanline_pad) = setup.pixmap_formats.iter().find(|format| {
                    format.depth == depth
                }).map(|format| (format.bits_per_pixel, format.scanline_pad)).unwrap_or_else(|| {
                    let bpp = match depth {
                        i if i < 4 => 4,
                        i if i < 8 => 8,
                        i if i < 16 => 16,
                        _ => 32,
                    };

                    (bpp, bpp)
                });

                (Format::Z { depth, bits_per_pixel: bpp }, scanline_pad)
            }
            _ => panic!("Unsupported image format"),
        };

        Ok(Self::new(
            storage,
            width,
            height,
            format,
            byte_order,
            scanline_pad
        ))
    }

    /// Convert the image back into its `Storage`.
    pub fn into_storage(self) -> Storage {
        self.storage
    }
}

impl<Storage: ?Sized> Image<Storage> {
    /// The number of bits in a scanline.
    fn bits_per_scanline(&self) -> usize {
        let unpadded_scanline = match self.image_format {
            Format::Xy { .. } => self.width as usize + self.left_pad as usize,
            Format::Z { bits_per_pixel, .. } => self.width as usize * bits_per_pixel as usize,
        };

        // pad scanline to scanline_pad
        let scanline_pad = self.scanline_pad as usize;
        pad_to(unpadded_scanline, scanline_pad)
    }

    /// The size of the plane to skip when x-y indexing.
    fn plane_size(&self) -> usize {
        self.bits_per_scanline() * self.height as usize
    }

    fn is_msb(&self) -> bool {
        matches!(self.byte_order, ImageOrder::MSB_FIRST) ||
            matches!(self.image_format, Format::Xy { bit_order: ImageOrder::MSB_FIRST, .. })
    }

    /// Get the depth of this image.
    pub fn depth(&self) -> u8 {
        match self.image_format {
            Format::Xy { format: XyFormatType::Pixmap { depth }, .. } => depth,
            Format::Z { depth, .. } => depth,
            Format::Xy { format: XyFormatType::Bitmap, .. } => 1,
        }
    }

    fn quantum(&self) -> usize {
        match self.image_format {
            Format::Xy { quantum, .. } => quantum as usize,
            _ => 8,
        }
    }

    fn quantum_xy_index(&self, x: usize, y: usize) -> usize {
        let quantum = self.quantum();

        (y * self.bits_per_scanline() / 8) + ((x + self.left_pad as usize) / quantum) * (quantum / 8)
    }

    fn bits_per_pixel(&self) -> usize {
        match self.image_format {
            Format::Z { bits_per_pixel, .. } => bits_per_pixel as usize,
            _ => unreachable!(),
        }
    }

    fn z_index(&self, x: usize, y: usize) -> usize {
        let bits_per_pixel = self.bits_per_pixel();
        let bits_per_scanline = self.bits_per_scanline();

        (y * bits_per_scanline / 8) + (x * bits_per_pixel / 8)
    }

    /// Given a set of bits from an XY setup, normalize the bits.
    fn normalize_bits_xy(&self, bits: &mut [u8]) {
        let bitmap_format = match self.image_format {
            Format::Xy { ref bit_order, .. } => *bit_order,
            _ => self.byte_order,
        };
        
        if matches!((self.byte_order, bitmap_format), (ImageOrder::LSB_FIRST, ImageOrder::LSB_FIRST)) {
            // no normalization needed
            return;
        }

        let quantum = self.quantum();

        match quantum {
            16 => {
                // swap first two bytes
                bits.swap(0, 1);
            }
            32 => {
                // swap other bytes
                bits.swap(0, 3);
                bits.swap(1, 2);
            }
            8 => {},
            q => unreachable!("invalid quantum: {}", q),
        }

        if matches!(bitmap_format, ImageOrder::LSB_FIRST) {
            reverse_bytes(&mut bits[0..quantum / 8])
        }
    }

    fn normalize_bits_z(&self, bits: &mut [u8]) {
        if matches!(self.byte_order, ImageOrder::LSB_FIRST) {
            return;
        }

        match self.bits_per_pixel() {
            4 => {
                // swap first two nibbles
                bits[0] = ((bits[0] >> 4) & 0x0F) | ((bits[0] << 4) & 0xF0);
            }
            8 => {},
            16 => {
                // swap first two bytes
                bits.swap(0, 1);
            }
            32 => {
                // swap other bytes
                bits.swap(0, 3);
                bits.swap(1, 2);
            }
            bpp => tracing::error!("invalid bits per pixel: {}", bpp),
        }
    }
}

impl<Storage: AsRef<[u8]> + ?Sized> Image<Storage> {
    /// Get a `Bitfield` representing the image as a whole.
    fn bitfield(&self) -> Bitfield<&[u8]> {
        Bitfield(self.storage.as_ref())
    } 

    /// Iterate over the regions of the image consisting of planes.
    fn planes(&self) -> impl Iterator<Item = &[u8]> + '_ {
        self.storage.as_ref().chunks(self.plane_size()).take(self.depth().into())
    }

    /// Assuming this is an X-Y format image, return the value of the pixel
    /// at the given coordinates.
    fn pixel_xy(&self, x: usize, y: usize) -> u32 {
        let bit_index = (x + self.left_pad as usize) % self.quantum();
        let quantum_bytes = self.quantum() / 8;
        let addr = self.quantum_xy_index(x, y);
        let mut result = 0;

        // iterate over planes
        for plane in self.planes() {
            // copy the intended bytes into a buffer
            let mut pixel = [0u8; 4];
            let data_slice = &plane[addr..addr + quantum_bytes];
            pixel[..quantum_bytes].copy_from_slice(data_slice);

            // normalize the bits for MSB
            self.normalize_bits_xy(&mut pixel);

            // shift the bits into the result
            let bit = u32::from(pixel[bit_index / 8]) >> (bit_index % 8);
            result = result << 1 | (bit & 1);
        }

        result
    }

    /// Assumming this is a Z-format image, return the value of the pixel
    /// at the given coordinates.
    fn pixel_z(&self, x: usize, y: usize) -> u32 {
        let mut pixel = [0u8; 4];
        let addr = self.z_index(x, y);
        let quantum_bytes = (self.bits_per_pixel() + 7) / 8;

        // copy the intended bytes into a buffer
        let data = &self.storage.as_ref()[addr..addr + quantum_bytes];
        pixel[..quantum_bytes].copy_from_slice(data);

        // format bytes into a u32
        let mut res = 0u32;
        for i in (0..4).rev() {
            res = res << 8 | u32::from(pixel[i]);
        }

        // flatten if needed
        if self.bits_per_pixel() == 4 {
            if x & 1 == 0 {
                res &= 0x0F;
            } else {
                res >>= 4;
            }
        }

        if self.bits_per_pixel() != self.depth() as usize {
            // only get the needed bits
            res = low_bits(res, self.depth() as usize);
        }

        res
    }

    /// Get the value of the pixel at the given coordinates.
    pub fn pixel(&self, x: usize, y: usize) -> u32 {
        // TODO: 1-depth Z pixmaps are 
        match self.image_format {
            Format::Xy { .. } | Format::Z { depth: 1, .. } => self.pixel_xy(x, y),
            Format::Z { .. } => self.pixel_z(x, y),
        }
    }
}

impl<Storage: AsMut<[u8]> + ?Sized> Image<Storage> {
    /// Assuming this is an X-Y format image, set the value of a pixel at the
    /// given coordinates.
    fn set_pixel_xy(&mut self, x: usize, y: usize, pixel: u32) {
        let bit_index = (x + self.left_pad as usize) % self.quantum();
        let quantum_bytes = self.quantum() / 8;
        let addr = self.quantum_xy_index(x, y);
        let mut result = 0;
        let plane_size = self.plane_size();

        for (i, planestart) in (0..self.depth() as usize).map(|i| i * plane_size).rev().enumerate() {
            // copy the current bytes into a buffer
            let mut buffer = [0u8; 4];
            let base = addr + planestart;
            let data_slice = &mut self.storage.as_mut()[base..base + quantum_bytes];
            buffer[..quantum_bytes].copy_from_slice(data_slice);

            // normalize the bits for MSB
            self.normalize_bits_xy(&mut buffer);

            // create bitfields for the target and source,
            // then preform a bitwise copy
            let mut buffer_bits = Bitfield(&mut buffer);
            let pixel_bits = Bitfield(pixel.to_ne_bytes());
            buffer_bits.set_bit(bit_index, pixel_bits.bit(i));

            // substitute buffer back into image
            self.normalize_bits_xy(&mut buffer);
            let data_slice = &mut self.storage.as_mut()[base..base + quantum_bytes];
            data_slice.copy_from_slice(&buffer[..quantum_bytes]);
        }
    }

    /// Assuming this is a Z format image, set the value of a pixel at the
    /// given coordinates.
    fn set_pixel_z(&mut self, x: usize, y: usize, pixel: u32) {
        let mut pixel = [0u8; 4];
        let addr = self.z_index(x, y);
        let quantum_bytes = (self.bits_per_pixel() + 7) / 8;
        let bpp = self.bits_per_pixel();

        // copy into a buffer
        let mut buffer = [0u8; 4];
        let data_slice = &mut self.storage.as_mut()[addr..addr + quantum_bytes];
        buffer[..quantum_bytes].copy_from_slice(data_slice);

        // normalize the bits for MSB
        self.normalize_bits_z(&mut buffer);

        // preform a bitwise copy
        let src = Bitfield(&pixel);
        let mut dest = Bitfield(&mut buffer);
        dest.copy_from(&src, 0, (x * bpp) % 8, bpp);

        // normalize and insert into image
        self.normalize_bits_z(&mut buffer);
        let data_slice = &mut self.storage.as_mut()[addr..addr + quantum_bytes];
        data_slice.copy_from_slice(&buffer[..quantum_bytes]);
    }

    /// Set the value of a pixel at the given coordinates.
    pub fn set_pixel(&mut self, x: usize, y: usize, pixel: u32) {
        match self.image_format {
            Format::Xy { .. } | Format::Z { depth: 1, .. } => self.set_pixel_xy(x, y, pixel),
            Format::Z { .. } => self.set_pixel_z(x, y, pixel),
        }
    }
}

/// A wrapper around a `u32` that implements `AsRef<[u8]>` and `AsMut<[u8]>`.
struct Pixel {
    val: [u8; 4],
}

impl Pixel {
    fn new(val: u32) -> Pixel {
        Pixel {
            val: val.to_le_bytes(),
        }
    }
}

impl From<Pixel> for u32 {
    fn from(pixel: Pixel) -> Self {
        u32::from_le_bytes(pixel.val)
    }
}

impl AsRef<[u8]> for Pixel {
    fn as_ref(&self) -> &[u8] {
        &self.val
    }
}

impl AsMut<[u8]> for Pixel {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.val
    }
}

struct Quantum([u8; 4]);

fn pad_to(val: usize, pad: usize) -> usize {
    val + (pad - (val % pad)) % pad
}

#[allow(clippy::needless_range_loop)]
#[cfg(test)]
mod tests {
    use alloc::vec;
    use breadx::protocol::xproto::{ImageFormat, ImageOrder};

    use super::Image;

    const TEST_WIDTH: usize = 35;
    const TEST_HEIGHT: usize = 35;

    fn test_permutation(format: Format, scanline_pad: u8, byte_order: ImageOrder) {
        // generate a random 35x35 image
        let mut image = [[0u32; TEST_WIDTH]; TEST_HEIGHT];
        for y in 0..TEST_HEIGHT {
            for x in 0..TEST_WIDTH {
                let limit = if depth == 32 {
                    u32::MAX
                } else {
                    2_u32.pow(depth as u32)
                };
                image[y][x] = fastrand::u32(..limit);
            }
        }

        // create an image to test with
        let storage = vec![0u8; TEST_WIDTH * TEST_HEIGHT * (depth as usize)];
        let mut ximage = Image::new(
            storage,
            TEST_WIDTH as _,
            TEST_HEIGHT as _,
            format,
            byte_order,
            scanline_pad,
        );

        // populate with pixels
        for y in 0..TEST_HEIGHT {
            for x in 0..TEST_WIDTH {
                ximage.set_pixel(x, y, image[y][x]);
            }
        }

        // read pixels from image and compare to original
        for y in 0..TEST_HEIGHT {
            for x in 0..TEST_WIDTH {
                assert_eq!(
                    image[y][x],
                    ximage.pixel(x, y),
                    "Pixel at ({}, {}) not equal for format {:?} and depth {}",
                    x,
                    y,
                    format,
                    depth
                );
            }
        }
    }

    #[test]
    fn test_get_and_set_pixel() {
        for format in [
            ImageFormat::Z_PIXMAP,
            ImageFormat::XY_PIXMAP,
            ImageFormat::XY_BITMAP,
        ]
        .iter()
        {
            for depth in (1..=32).rev() {
                // x-y bitmaps only ever have a depth of 1
                if *format == ImageFormat::XY_BITMAP && depth != 1 {
                    continue;
                }

                for byte_order in [ImageOrder::LSB_FIRST, ImageOrder::MSB_FIRST].iter() {
                    test_permutation(*format, depth, *byte_order);
                }
            }
        }
    }
}
