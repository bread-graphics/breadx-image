// MIT/Apache2 License

//! Benchmark encoding/decoding an image.

use breadx::protocol::xproto::{ImageFormat, ImageOrder};
use breadx_image::Image;
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
            let mut image = Image::new(
                &mut storage,
                TEST_WIDTH as _,
                TEST_HEIGHT as _,
                image_format,
                TEST_DEPTH as _,
                byte_order,
                TEST_DEPTH as _,
                0,
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
            let mut image = Image::new(
                &mut storage,
                TEST_WIDTH as _,
                TEST_HEIGHT as _,
                image_format,
                TEST_DEPTH as _,
                byte_order,
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
