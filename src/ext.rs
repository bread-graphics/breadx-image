// MIT/Apache2 License

use super::Image;
use breadx::{display::{Display, DisplayExt as _}, protocol::xproto::{Drawable, Gcontext}, display::Cookie};

pub trait DisplayExt: Display {
    /// Send an [`Image`] to the display.
    fn put_ximage(
        &mut self,
        image: &Image<impl AsRef<[u8]>>,
        drawable: impl Into<Drawable>,
        gc: impl Into<Gcontext>,
        dst_x: i16,
        dst_y: i16,
    ) -> breadx::Result<Cookie<()>> {
        let req = image.put_image_request(drawable, gc, dst_x, dst_y);
        self.send_void_request(req)
    }

    fn put_ximage_checked(
        &mut self,
        image: &Image<impl AsRef<[u8]>>,
        drawable: impl Into<Drawable>,
        gc: impl Into<Gcontext>,
        dst_x: i16,
        dst_y: i16,
    ) -> breadx::Result<()> {
        let cookie = self.put_ximage(drawable, gc, dst_x, dst_y, image)?;
        self.wait_for_reply(cookie)
    }

    /// Send a subimage of an [`Image`] to the display.
    fn put_subimage(
        &mut self,
        image: &Image<impl AsRef<[u8]>>,
        drawable: impl Into<Drawable>,
        gc: impl Into<Gcontext>,
        src_x: usize,
        src_y: usize,
        width: usize,
        height: usize,
        dst_x: i16,
        dst_y: i16,
    ) -> breadx::Result<Cookie<()>> {
        image.put_subimage_request(drawable, gc, src_x, src_y, width, height, dst_x, dst_y, |req| {
            self.send_request_raw(req).map(Cookie::from_sequence)
        })
    }

    fn put_subimage_checked(
        &mut self,
        image: &Image<impl AsRef<[u8]>>,
        drawable: impl Into<Drawable>,
        gc: impl Into<Gcontext>,
        src_x: usize,
        src_y: usize,
        width: usize,
        height: usize,
        dst_x: i16,
        dst_y: i16,
    ) -> breadx::Result<()> {
        let cookie = self.put_subimage(drawable, gc, src_x, src_y, width, height, dst_x, dst_y, image)?;
        self.wait_for_reply(cookie)
    }
}

impl<D: Display + ?Sized> DisplayExt for D {}