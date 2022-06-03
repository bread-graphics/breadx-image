// MIT/Apache2 License

use breadx::{prelude::*, display::DisplayConnection, protocol::xproto};
use breadx_image::{prelude::*, Image};

#[cfg(feature = "std")]
fn main() -> breadx::Result<()> {
    let mut conn = DisplayConnection::connect(None)?;

    // set up a window to be displayed
    // see basic.rs in breadx for a more in depth explanation
    let events = xproto::EventMask::EXPOSURE;
    let background = conn.default_screen().white_pixel;

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {
    println!("This example requires the `std` feature to be enabled");
}