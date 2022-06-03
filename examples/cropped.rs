// MIT/Apache2 License

use breadx::{display::DisplayConnection, prelude::*, protocol::xproto};
use breadx_image::{prelude::*, Image};
use std::{boxed::Box, error::Error, io::Cursor};

const EISENHOWER: &[u8] = include_bytes!("../images/eisenhower.png");

#[cfg(feature = "std")]
fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    use breadx::protocol::Event;
    use breadx_image::DisplayExt;

    let mut conn = DisplayConnection::connect(None)?;

    // set up a window to be displayed and a gc for that window
    // see basic.rs in breadx for a more in depth explanation
    let events = xproto::EventMask::EXPOSURE;
    let background = conn.default_screen().white_pixel;
    let parent = conn.default_screen().root;

    // use the image-rs library to load eisenhower.webp
    let img = image::io::Reader::with_format(Cursor::new(EISENHOWER), image::ImageFormat::Png)
        .decode()?;
    let img = img.to_rgb8();
    let wid = conn.generate_xid()?;

    conn.create_window_checked(
        0, // depth for 24-bit color
        wid,
        parent,
        0,
        0,
        50,
        50,
        0,
        xproto::WindowClass::COPY_FROM_PARENT,
        0,
        xproto::CreateWindowAux::new()
            .event_mask(events)
            .background_pixel(background),
    )?;
    conn.map_window_checked(wid)?;
    let title = "Eisencropped";
    conn.change_property(
        xproto::PropMode::REPLACE,
        wid,
        xproto::AtomEnum::WM_NAME.into(),
        xproto::AtomEnum::STRING.into(),
        8,
        title.len() as u32,
        title,
    )?;

    let window_gc = conn.generate_xid()?;
    conn.create_gc(
        window_gc,
        wid,
        xproto::CreateGCAux::new().graphics_exposures(0),
    )?;

    // create a new image and copy the image data into it
    let depth = conn.get_geometry_immediate(wid)?.depth;
    let format = xproto::ImageFormat::Z_PIXMAP;
    let len =
        breadx_image::storage_bytes(img.width() as _, img.height() as _, depth, None, format, 32);
    let mut ximage = Image::with_display(
        vec![0; len],
        img.width() as _,
        img.height() as _,
        format,
        depth,
        conn.setup(),
    )?;

    for (x, y, pixel) in img.enumerate_pixels() {
        // copy the pixel into the image
        let [r, g, b] = pixel.0;
        let channel = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;

        ximage.set_pixel(x as _, y as _, channel);
    }

    // create a pixmap and copy the image into it
    let pixmap = conn.generate_xid()?;
    let pixmap_gc = conn.generate_xid()?;
    conn.create_pixmap_checked(depth, pixmap, wid, 50, 50)?;
    conn.create_gc_checked(
        pixmap_gc,
        pixmap,
        xproto::CreateGCAux::new()
            .foreground(conn.default_screen().black_pixel)
            .graphics_exposures(0),
    )?;
    conn.put_subimage_checked(
        &ximage,
        pixmap,
        pixmap_gc,
        50,
        50,
        50,
        50,
        0,
        0
    )?;

    // now, let's enter the main loop
    let wm_protocols = conn.intern_atom(false, "WM_PROTOCOLS")?;
    let wm_delete_window = conn.intern_atom(false, "WM_DELETE_WINDOW")?;
    let wm_protocols = conn.wait_for_reply(wm_protocols)?.atom;
    let wm_delete_window = conn.wait_for_reply(wm_delete_window)?.atom;

    conn.change_property(
        xproto::PropMode::REPLACE,
        wid,
        wm_protocols,
        xproto::AtomEnum::ATOM.into(),
        32,
        1,
        &wm_delete_window,
    )?;

    loop {
        let event = conn.wait_for_event()?;

        match event {
            Event::Expose(_) => {
                // repaint by copying the area from the pixmap
                // to the window
                conn.copy_area_checked(
                    pixmap,
                    wid,
                    window_gc,
                    0,
                    0,
                    0,
                    0,
                    50,
                    50,
                )?;
            }
            Event::ClientMessage(cme) => {
                if cme.data.as_data32()[0] == wm_delete_window {
                    break;
                }
            }
            _ => {}
        }
    }

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {
    println!("This example requires the `std` feature to be enabled");
}
