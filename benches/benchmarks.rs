// MIT/Apache2 License

//! Benchmark encoding/decoding an image.

use breadx::protocol::xproto::{ImageFormat, ImageOrder};
use breadx_image::{Image, Format, BitsPerPixel, XyFormatType};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

// create a rather large image
const TEST_WIDTH: usize = 1024;
const TEST_HEIGHT: usize = 512;
const TEST_DEPTH: usize = 32;

fn encode(c: &mut Criterion) {
    // generate a large image
    let mut storage = vec![0u8; TEST_WIDTH * TEST_HEIGHT * TEST_DEPTH];

    let mut group = c.benchmark_group("encode");
    for byte_order in [ImageOrder::LSB_FIRST, ImageOrder::MSB_FIRST] {
        for image_format in [ImageFormat::XY_PIXMAP, ImageFormat::Z_PIXMAP] {
            let format = match image_format {
                ImageFormat::Z_PIXMAP => Format::Z {
                    depth: TEST_DEPTH as _,
                    bits_per_pixel: BitsPerPixel::roundup_from_depth(TEST_DEPTH as _),
                },
                _ => Format::Xy {
                    format: XyFormatType::Pixmap { depth: TEST_DEPTH as _ },
                    quantum: breadx_image::Quantum::ThirtyTwo,
                    bit_order: byte_order,
                    left_pad: 0,
                }
            };

            let mut image = Image::new(
                &mut storage,
                TEST_WIDTH as _,
                TEST_HEIGHT as _,
                format,
                byte_order,
                32
            );

            let name = format!("{:?} {:?}", byte_order, image_format);
            group.bench_function(name, |b| {
                b.iter(|| {
                    for y in 0..TEST_HEIGHT {
                        for x in 0..TEST_WIDTH {
                            image.set_pixel(x, y, black_box((x * y) as u32));
                        }
                    }
                })
            });
        }
    }
}

fn decode(c: &mut Criterion) {
    // generate a large image
    let mut storage = vec![0u8; TEST_WIDTH * TEST_HEIGHT * TEST_DEPTH];

    let mut group = c.benchmark_group("decode");
    for byte_order in [ImageOrder::LSB_FIRST, ImageOrder::MSB_FIRST] {
        for image_format in [ImageFormat::XY_PIXMAP, ImageFormat::Z_PIXMAP] {
            let format = match image_format {
                ImageFormat::Z_PIXMAP => Format::Z {
                    depth: TEST_DEPTH as _,
                    bits_per_pixel: BitsPerPixel::roundup_from_depth(TEST_DEPTH as _),
                },
                _ => Format::Xy {
                    format: XyFormatType::Pixmap { depth: TEST_DEPTH as _ },
                    quantum: breadx_image::Quantum::ThirtyTwo,
                    bit_order: byte_order,
                    left_pad: 0,
                }
            };

            let mut image = Image::new(
                &mut storage,
                TEST_WIDTH as _,
                TEST_HEIGHT as _,
                format,
                byte_order,
                32,
            );

            // encode a series of numbers into it
            for y in 0..TEST_HEIGHT {
                for x in 0..TEST_WIDTH {
                    image.set_pixel(x, y, black_box((x * y) as u32));
                }
            }

            let name = format!("{:?} {:?}", byte_order, image_format);
            group.bench_function(name, |b| {
                b.iter(|| {
                    for y in 0..TEST_HEIGHT {
                        for x in 0..TEST_WIDTH {
                            black_box(image.pixel(x, y));
                        }
                    }
                })
            });
        }
    }
}

criterion_group!(benches, decode, encode);
criterion_main!(benches);
