// MIT/Apache2 License

//! A library for dealing with the X11 image format.

#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod bits;
use alloc::{vec, vec::Vec, boxed::Box};
use bits::Bitfield;

mod byte_tables;
use byte_tables::{low_bits, reverse_bytes};

mod ext;
pub use ext::DisplayExt;

mod subimage;

use breadx::{
    protocol::xproto::{ImageFormat, ImageOrder, PutImageRequest, Drawable, Gcontext, GetImageReply, Setup},
};
use core::{
    convert::{TryFrom, TryInto},
    fmt,
    num::TryFromIntError,
};

#[cfg(feature = "std")]
use std::error::Error;

/*

Personal notes:

There are two main image formats: XY and Z

 - Z is how you'd normally see an image: a series of scanlines,
   padded to a given scanline pad, where each scanline is a series
   of pixels made of a set of bits. Each pixel can be either 1, 4,
   8, 16, or 32 bits. We ensure this on a type-level by using
   the `BitsPerPixel` enum type.

   If the image has the MSB byte order, it needs to be normalized.
   For BPP of 8 and above, the bytes can be reversed. For BPP of
   4, the nibbles need to be reversed. For BPP of 1, normalization
   is the exact same as in an XY image.

   Note that the depth of a Z pixmap image may not be equal to the
   BPP, but must be less than or equal to it. We truncate it using
   a byte table otherwise.

 - XY is the other image format. It consists of a series of "planes",
   which is made up of scanlines containing quantums. A quantum is either
   8 bits, 16 bits or 32 bits. We ensure this on a type-level by using
   the `Quantum` enum type. Each quantum contains several bits each
   representing a single bit of the pixel. The planes are ordered from
   the most significant bit to the least significant bit, and there are
   a number of planes equal to the depth. To reconstruct a pixel, find
   each bit from the planes and put them together.

   Note that there are two "sub-types" of this image type. XY pixmaps
   operate as above, while XY bitmaps only have a depth of 1. XY bitmaps
   are useful because they can be colored arbitrarily.

   If the image has an MSB byte order or the server requires an MSB bit order,
   the image needs to be normalized. If the byte order is not equal to the
   bit order, the bytes can be reversed as above. If the bit order is MSB,
   the bits of each byte need to be reversed. This is done by using a table
   that should be computed at compile time. Thanks, const eval!

 - The `left_pad` (*Family Guy style flashback occurs*) value should be zero for
   all Z images, and the value at which the scanline starts for XY images. This
   could probably be used to help crop XY images in the future.

 - A Z pixmap with a depth of 1 is fundamentally identical in operation to an
   XY bitmap. Therefore, operations on 1-depth Z pixmaps should degenerate into
   operations on XY bitmaps.

 - Setting a pixel in any image can be done by reading the corresponding bytes,
   preforming bit manipulations, and then substituting the new bytes back into
   the image.

 - To crop an image, create another image of a smaller size, and then get/set
   pixels. There may be a more optimized way of doing this using ZPixmaps.

 - The `Image` struct should be able to be easily transformed into a PutImageRequest
   struct so that way it can be sent across the wire. It should also be able to be
   easily deserialized from a GetImageReply.

 - A "subimage" of an `Image` may be able to be sent in a PutImageRequest. In the
   worst case, this would degenerate into a `crop()` followed by the regular send.
   However, we may be able to crop each individual scanline in order to format them
   as a PutImageRequest. Depends on whether the quanums match up with the byte
   boundaries.

   We can use the RawRequest API to manually set up each scanline as its own
   I/O slice in these cases.

 */

/// The valid values for the bits per pixel of an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum BitsPerPixel {
    /// 1-bit per pixel.
    One = 1,
    /// 4-bits per pixel.
    Four = 4,
    /// 8-bits per pixel.
    Eight = 8,
    /// 16-bits per pixel.
    Sixteen = 16,
    /// 32-bits per pixel.
    ThirtyTwo = 32,
}

impl BitsPerPixel {
    pub fn roundup_from_depth(depth: u8) -> BitsPerPixel {
        match depth {
            i if i <= 1 => BitsPerPixel::One,
            i if i <= 4 => BitsPerPixel::Four,
            i if i <= 8 => BitsPerPixel::Eight,
            i if i <= 16 => BitsPerPixel::Sixteen,
            _ => BitsPerPixel::ThirtyTwo,
        }
    }
}

impl TryFrom<u8> for BitsPerPixel {
    type Error = InvalidNumber;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(BitsPerPixel::One),
            4 => Ok(BitsPerPixel::Four),
            8 => Ok(BitsPerPixel::Eight),
            16 => Ok(BitsPerPixel::Sixteen),
            32 => Ok(BitsPerPixel::ThirtyTwo),
            value => Err(InvalidNumber::out_of_range(value, &[1, 4, 8, 16, 32])),
        }
    }
}

/// The valid values for the quantum of an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Quantum {
    /// 8-bits per pixel.
    Eight = 8,
    /// 16-bits per pixel.
    Sixteen = 16,
    /// 32-bits per pixel.
    ThirtyTwo = 32,
}

impl TryFrom<u8> for Quantum {
    type Error = InvalidNumber;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(Quantum::Eight),
            16 => Ok(Quantum::Sixteen),
            32 => Ok(Quantum::ThirtyTwo),
            value => Err(InvalidNumber::out_of_range(value, &[8, 16, 32])),
        }
    }
}

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
        quantum: Quantum,
        /// The bit order to be used in the image.
        ///
        /// This determines whether the leftmost bit is either the least or most
        /// significant bit.
        bit_order: ImageOrder,
        /// Leftmost padding to apply.
        left_pad: u8,
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
        bits_per_pixel: BitsPerPixel,
    },
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
    },
}

impl Format {
    pub fn depth(&self) -> u8 {
        match self {
            Format::Xy {
                format: XyFormatType::Pixmap { depth },
                ..
            } => *depth,
            Format::Xy { .. } => 1,
            Format::Z { depth, .. } => *depth,
        }
    }

    pub fn format(&self) -> ImageFormat {
        match self {
            Format::Xy {
                format: XyFormatType::Bitmap,
                ..
            } => ImageFormat::XY_BITMAP,
            Format::Xy {
                format: XyFormatType::Pixmap { .. },
                ..
            } => ImageFormat::XY_PIXMAP,
            Format::Z { .. } => ImageFormat::Z_PIXMAP,
        }
    }
}

/// Computes the number of bytes necessary to store the image with the given parameters.
pub fn storage_bytes(
    width: u16,
    height: u16,
    depth: u8,
    bits_per_pixel: Option<BitsPerPixel>,
    format: ImageFormat,
    scanline_pad: u8,
) -> usize {
    let bytes_per_line = match format {
        ImageFormat::Z_PIXMAP => {
            let bpp =
                bits_per_pixel.unwrap_or_else(|| BitsPerPixel::roundup_from_depth(depth)) as usize;
            pad_to(pad_to(bpp * width as usize, 8) / 8, scanline_pad.into())
        }
        ImageFormat::XY_PIXMAP | ImageFormat::XY_BITMAP => {
            pad_to(width.into(), scanline_pad.into())
        }
        _ => panic!("Unsupported image format: {:?}", format),
    };

    let plane_len = bytes_per_line * height as usize;

    if matches!(format, ImageFormat::Z_PIXMAP) {
        plane_len
    } else {
        plane_len * depth as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Image<Storage: ?Sized> {
    /// The width of the image.
    width: u16,
    /// The height of the image.
    height: u16,
    /// The format used in the image.
    image_format: Format,
    /// The byte ordering used in the image.
    ///
    /// This applies to the scanline units used in the Z pixmap format,
    /// or to the quantums used in the XY format.
    byte_order: ImageOrder,
    /// The scanline pad for each scanline, in bits.
    scanline_pad: u8,
    /// The storage system used to store bytes.
    storage: Storage,
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
            byte_order,
            storage,
        }
    }

    /// Create a new `Image`, but use a specific `Display` to set certain important values.
    pub fn with_display(
        storage: Storage,
        width: u16,
        height: u16,
        image_format: ImageFormat,
        depth: u8,
        setup: &Setup,
    ) -> breadx::Result<Self> {
        let byte_order = setup.image_byte_order;
        let (format, scanline_pad) = match image_format {
            ImageFormat::XY_BITMAP | ImageFormat::XY_PIXMAP => (
                Format::Xy {
                    format: match image_format {
                        ImageFormat::XY_BITMAP => XyFormatType::Bitmap,
                        ImageFormat::XY_PIXMAP => XyFormatType::Pixmap { depth },
                        _ => unreachable!(),
                    },
                    quantum: setup
                        .bitmap_format_scanline_unit
                        .try_into()
                        .expect("invalid quantum"),
                    bit_order: byte_order,
                    left_pad: 0,
                },
                setup.bitmap_format_scanline_pad,
            ),
            ImageFormat::Z_PIXMAP => {
                let (bpp, scanline_pad) = setup
                    .pixmap_formats
                    .iter()
                    .find(|format| format.depth == depth)
                    .map(|format| {
                        (
                            format.bits_per_pixel.try_into().expect("invalid bpp"),
                            format.scanline_pad,
                        )
                    })
                    .unwrap_or_else(|| {
                        let bpp = BitsPerPixel::roundup_from_depth(depth);
                        (bpp, setup.bitmap_format_scanline_pad)
                    });

                (
                    Format::Z {
                        depth,
                        bits_per_pixel: bpp,
                    },
                    scanline_pad,
                )
            }
            _ => panic!("Unsupported image format"),
        };

        Ok(Self::new(
            storage,
            width,
            height,
            format,
            byte_order,
            scanline_pad,
        ))
    }

    /// Convert the image back into its `Storage`.
    pub fn into_storage(self) -> Storage {
        self.storage
    }
    
    /// Map the storage for this image.
    /// 
    /// This function assumes that nothing else about the image is changed.
    pub fn map_storage<R, F>(self, f: F) -> Image<R>
    where
        F: FnOnce(Storage) -> R,
    {
        let Self { width, height, image_format, byte_order, scanline_pad, storage } = self;

        Image {
            width,
            height,
            image_format,
            byte_order,
            scanline_pad,
            storage: f(storage),
        }
    }
}

impl<Storage: ?Sized> Image<Storage> {
    /// The number of bits in a scanline.
    fn bits_per_scanline(&self) -> usize {
        let unpadded_scanline = match self.image_format {
            Format::Xy { ref left_pad, .. } => self.width as usize + *left_pad as usize,
            Format::Z {
                ref bits_per_pixel, ..
            } => self.width as usize * *bits_per_pixel as usize,
        };

        // pad scanline to scanline_pad
        let scanline_pad = self.scanline_pad as usize;
        pad_to(unpadded_scanline, scanline_pad)
    }

    fn bytes_per_scanline(&self) -> usize {
        self.bits_per_scanline() / 8
    }

    /// The size of the plane to skip when x-y indexing.
    fn plane_size(&self) -> usize {
        self.bits_per_scanline() * self.height as usize
    }

    /// Get the depth of this image.
    pub fn depth(&self) -> u8 {
        match self.image_format {
            Format::Xy {
                format: XyFormatType::Pixmap { depth },
                ..
            } => depth,
            Format::Z { depth, .. } => depth,
            Format::Xy {
                format: XyFormatType::Bitmap,
                ..
            } => 1,
        }
    }

    fn quantum(&self) -> usize {
        match self.image_format {
            Format::Xy { quantum, .. } => quantum as usize,
            _ => 8,
        }
    }

    fn left_pad(&self) -> usize {
        match self.image_format {
            Format::Xy { left_pad, .. } => left_pad as usize,
            _ => 0,
        }
    }

    /// Determine the index of where the quantum starts for an XY image.
    fn quantum_xy_index(&self, x: usize, y: usize) -> usize {
        let quantum = self.quantum();
        let left_pad = self.left_pad();

        (y * self.bytes_per_scanline()) + ((x + left_pad) / quantum) * (quantum / 8)
    }

    fn bits_per_pixel(&self) -> usize {
        match self.image_format {
            Format::Z { bits_per_pixel, .. } => bits_per_pixel as usize,
            _ => unreachable!(),
        }
    }

    fn z_index(&self, x: usize, y: usize) -> usize {
        let bits_per_pixel = self.bits_per_pixel();

        (y * self.bytes_per_scanline()) + ((x * bits_per_pixel) / 8)
    }

    /// Given a set of bits from an XY setup, normalize the bits.
    fn normalize_bits_xy(&self, bits: &mut [u8]) {
        let (bitmap_format, quantum) = match self.image_format {
            Format::Xy {
                ref bit_order,
                ref quantum,
                ..
            } => (*bit_order, *quantum),
            Format::Z { .. } => (self.byte_order, Quantum::try_from(self.scanline_pad).unwrap()),
        };

        if matches!(
            (self.byte_order, bitmap_format),
            (ImageOrder::LSB_FIRST, ImageOrder::LSB_FIRST)
        ) {
            // no normalization needed
            return;
        }

        if self.byte_order != bitmap_format {
            match quantum {
                Quantum::Sixteen => {
                    // swap first two bytes
                    debug_assert_eq!(bits.len(), 2);
                    bits.swap(0, 1);
                }
                Quantum::ThirtyTwo => {
                    // swap other bytes
                    bits.swap(0, 3);
                    bits.swap(1, 2);
                }
                _ => {}
            }
        }

        if matches!(bitmap_format, ImageOrder::MSB_FIRST) {
            let quantum = quantum as usize;
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
            8 => {}
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

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut Storage {
        &mut self.storage
    }

    pub fn width(&self) -> usize {
        self.width.into()
    }

    pub fn height(&self) -> usize {
        self.height.into()
    }

    pub fn format(&self) -> &Format {
        &self.image_format
    }
}

impl<Storage: AsRef<[u8]> + ?Sized> Image<Storage> {
    pub fn borrow(&self) -> Image<&[u8]> {
        Image { 
            width: self.width,
            height: self.height, 
            image_format: self.image_format, 
            byte_order: self.byte_order, 
            scanline_pad: self.scanline_pad, 
            storage: self.storage.as_ref(),
        }
    }

    /// Iterate over the regions of the image consisting of planes.
    fn planes(&self) -> impl Iterator<Item = &[u8]> + '_ {
        self.storage
            .as_ref()
            .chunks(self.plane_size())
            .take(self.depth().into())
    }

    /// Assuming this is an X-Y format image, return the value of the pixel
    /// at the given coordinates.
    fn pixel_xy(&self, x: usize, y: usize) -> u32 {
        let bit_index = (x + self.left_pad()) % self.quantum();
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
            self.normalize_bits_xy(&mut pixel[..quantum_bytes]);

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
        self.normalize_bits_z(&mut pixel[..quantum_bytes]);

        // format bytes into a u32
        let mut res = u32::from_le_bytes(pixel);

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

    /// Create a [`PutImageRequest`] for this image.
    pub fn put_image_request(
        &self,
        drawable: impl Into<Drawable>,
        gc: impl Into<Gcontext>,
        dst_x: i16,
        dst_y: i16,
    ) -> PutImageRequest<'_> {
        let drawable = drawable.into();
        let gc = gc.into();

        PutImageRequest {
            format: self.image_format.format(),
            drawable,
            gc,
            width: self.width,
            height: self.height,
            dst_x,
            dst_y,
            left_pad: match self.image_format {
                Format::Xy { ref left_pad, .. } => *left_pad,
                _ => 0,
            },
            depth: self.depth(),
            data: self.storage.as_ref().into(),
        }
    }

    /// Crop this image down to the given dimensions.
    /// 
    /// ## Performance
    /// 
    /// This function creates a new image and then manually copies all of the
    /// pixels from this image to that one, using the `pixel` and `set_pixel`
    /// functions. This usually leads to poor performance.
    pub fn crop(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Image<Box<[u8]>> {
        // create a new empty image
        let bpp = match self.image_format {
            Format::Xy { .. } => None,
            Format::Z { ref bits_per_pixel, .. } => Some(*bits_per_pixel),
        };
        let len = storage_bytes(width as _, height as _, self.depth(), bpp, self.format().format(), self.scanline_pad);
        let storage = vec![0u8; len].into_boxed_slice();
        let mut image = Image::new(
            storage,
            width as _,
            height as _,
            self.image_format,
            self.byte_order,
            self.scanline_pad,
        );

        // TODO: do we need to consider left_pad?

        // begin a pixelwise copy
        for dest_y in 0..height {
            for dest_x in 0..width {
                let src_x = x + dest_x;
                let src_y = y + dest_y;

                let src_pixel = self.pixel(src_x, src_y);
                image.set_pixel(dest_x, dest_y, src_pixel);
            }
        }

        // we're done
        image
    }
}

impl<Storage: AsMut<[u8]> + ?Sized> Image<Storage> {
    pub fn borrow_mut(&mut self) -> Image<&mut [u8]> {
        Image {
            storage: self.storage.as_mut(),
            width: self.width,
            height: self.height,
            image_format: self.image_format,
            byte_order: self.byte_order,
            scanline_pad: self.scanline_pad,
        }
    }

    /// Assuming this is an X-Y format image, set the value of a pixel at the
    /// given coordinates.
    fn set_pixel_xy(&mut self, x: usize, y: usize, pixel: u32) {
        let bit_index = (x + self.left_pad()) % self.quantum();
        let quantum_bytes = self.quantum() / 8;
        let addr = self.quantum_xy_index(x, y);
        let plane_size = self.plane_size();

        for (i, planestart) in (0..self.depth() as usize)
            .map(|i| i * plane_size)
            .rev()
            .enumerate()
        {
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
        let addr = self.z_index(x, y);
        let bpp = self.bits_per_pixel();
        let quantum_bytes = (bpp + 7) / 8;

        // copy into a buffer
        let mut buffer = [0u8; 4];
        let data_slice = &mut self.storage.as_mut()[addr..addr + quantum_bytes];
        buffer[..quantum_bytes].copy_from_slice(data_slice);

        // normalize the bits for MSB
        self.normalize_bits_z(&mut buffer[..quantum_bytes]);

        // preform a bitwise copy
        let src = Bitfield(pixel.to_le_bytes());
        let mut dest = Bitfield(&mut buffer);
        let dest_addr = (x * bpp) % 8;
        dest.copy_from(&src, dest_addr, 0, bpp);

        // normalize and insert into image
        self.normalize_bits_z(&mut buffer[..quantum_bytes]);
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

impl Image<Vec<u8>> {
    /// Create a new image based on a [`GetImageReply`].
    /// 
    /// [`GetImageReply`]: breadx::protocol::xproto::GetImageReply
    pub fn from_get_image_reply(reply: GetImageReply, width: u16, height: u16, format: ImageFormat, setup: &Setup) -> breadx::Result<Self> {
        let GetImageReply { depth, data, .. } = reply;

        Self::with_display(data, width, height, format, depth, setup)
    }
}

fn pad_to(val: usize, pad: usize) -> usize {
    val + (pad - (val % pad)) % pad
}

/// An invalid number was provided.
#[derive(Debug, Clone, Copy)]
pub struct InvalidNumber(Innards);

#[derive(Debug, Clone, Copy)]
enum Innards {
    InvalidConversion(TryFromIntError),
    NotInRange { value: u8, valid: &'static [u8] },
}

impl InvalidNumber {
    fn out_of_range(value: u8, valid: &'static [u8]) -> InvalidNumber {
        InvalidNumber(Innards::NotInRange { value, valid })
    }
}

impl fmt::Display for InvalidNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Innards::InvalidConversion(_) => write!(f, "invalid int conversion"),
            Innards::NotInRange { value, valid } => {
                write!(f, "value {} not in range {:?}", value, valid)
            }
        }
    }
}

#[cfg(feature = "std")]
impl Error for InvalidNumber {}

impl From<TryFromIntError> for InvalidNumber {
    fn from(err: TryFromIntError) -> Self {
        InvalidNumber(Innards::InvalidConversion(err))
    }
}

#[allow(clippy::needless_range_loop)]
#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::vec;
    use breadx::protocol::xproto::{ImageFormat, ImageOrder};
    use std::panic;

    use super::{Format, Image, XyFormatType};

    const TEST_WIDTH: usize = 35;
    const TEST_HEIGHT: usize = 35;

    fn test_permutation(format: Format, depth: u8, scanline_pad: u8, byte_order: ImageOrder) {
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
        let len = crate::storage_bytes(
            TEST_WIDTH as _,
            TEST_HEIGHT as _,
            depth,
            None,
            format.format(),
            scanline_pad,
        );
        let storage = vec![0u8; len];
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
                    "Pixel at ({}, {}) not equal for format {:?}, byte order {:?} and depth {}",
                    x,
                    y,
                    format,
                    byte_order,
                    depth
                );
            }
        }
    }

    #[test]
    fn test_get_and_set_pixel() {
        for format in &[
            ImageFormat::Z_PIXMAP,
            ImageFormat::XY_PIXMAP,
            ImageFormat::XY_BITMAP,
        ] {
            for depth in 1u8..=32 {
                // x-y bitmaps only ever have a depth of 1
                if *format == ImageFormat::XY_BITMAP && depth != 1 {
                    continue;
                }

                // this is broken for now, TODO
                if *format == ImageFormat::Z_PIXMAP && depth == 1 {
                    continue;
                }

                let scanline_pad = 32;

                for byte_order in &[ImageOrder::LSB_FIRST, ImageOrder::MSB_FIRST] {
                    for bit_order in &[ImageOrder::LSB_FIRST, ImageOrder::MSB_FIRST] {
                        // bit order doesn't matter for Z Pixmaps
                        if *format == ImageFormat::Z_PIXMAP && *bit_order != ImageOrder::LSB_FIRST {
                            continue;
                        }

                        // make the format struct
                        let format = match *format {
                            ImageFormat::XY_BITMAP => Format::Xy {
                                format: XyFormatType::Bitmap,
                                quantum: crate::Quantum::ThirtyTwo,
                                bit_order: *bit_order,
                                left_pad: 0,
                            },
                            ImageFormat::XY_PIXMAP => Format::Xy {
                                format: XyFormatType::Pixmap { depth },
                                quantum: crate::Quantum::ThirtyTwo,
                                bit_order: *bit_order,
                                left_pad: 0,
                            },
                            ImageFormat::Z_PIXMAP => Format::Z {
                                depth: depth as _,
                                bits_per_pixel: crate::BitsPerPixel::roundup_from_depth(depth as _),
                            },
                            _ => unreachable!(),
                        };

                        if let Err(e) = panic::catch_unwind(|| {
                            test_permutation(format, depth, scanline_pad, *byte_order);
                        }) {
                            std::eprintln!(
                                "panic occurred with format {:?} and byte order {:?}",
                                format,
                                byte_order
                            );
                            panic::resume_unwind(e);
                        }
                    }
                }
            }
        }
    }
}
