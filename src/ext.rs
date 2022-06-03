// MIT/Apache2 License

#![allow(clippy::too_many_arguments)]

use super::Image;
use alloc::vec::Vec;
use breadx::{
    display::Cookie,
    display::{Display, DisplayExt as _, DisplayFunctionsExt as _},
    protocol::xproto::{Drawable, Gcontext, ImageFormat},
};

#[cfg(feature = "async")]
use breadx::{
    display::{AsyncDisplay, AsyncDisplayExt as _},
    futures,
};

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
        let cookie = self.put_ximage(image, drawable, gc, dst_x, dst_y)?;
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
        image.put_subimage_request(
            drawable,
            gc,
            src_x,
            src_y,
            width,
            height,
            dst_x,
            dst_y,
            |req| self.send_request_raw(req).map(Cookie::from_sequence),
        )
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
        let cookie = self.put_subimage(
            image, drawable, gc, src_x, src_y, width, height, dst_x, dst_y,
        )?;
        self.wait_for_reply(cookie)
    }

    fn get_ximage(
        &mut self,
        drawable: impl Into<Drawable>,
        format: ImageFormat,
        x: i16,
        y: i16,
        width: u16,
        height: u16,
        plane_mask: u32,
    ) -> breadx::Result<Image<Vec<u8>>> {
        let repl =
            self.get_image_immediate(format, drawable.into(), x, y, width, height, plane_mask)?;
        Image::from_get_image_reply(repl, width, height, format, self.setup())
    }
}

impl<D: Display + ?Sized> DisplayExt for D {}

#[cfg(feature = "async")]
pub trait AsyncDisplayExt: AsyncDisplay {
    /// Send an [`Image`] to the display.
    fn put_ximage(
        &mut self,
        image: &Image<impl AsRef<[u8]>>,
        drawable: impl Into<Drawable>,
        gc: impl Into<Gcontext>,
        dst_x: i16,
        dst_y: i16,
    ) -> futures::SendRequest<'_, Self, ()> {
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
    ) -> futures::CheckedSendRequest<'_, Self, ()> {
        let cookie = self.put_ximage(image, drawable, gc, dst_x, dst_y);
        cookie.into()
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
    ) -> futures::SendRequestRaw<'_, Self> {
        image.put_subimage_request(
            drawable,
            gc,
            src_x,
            src_y,
            width,
            height,
            dst_x,
            dst_y,
            |req| self.send_request_raw(req),
        )
    }
}

#[cfg(feature = "async")]
impl<D: AsyncDisplay + ?Sized> AsyncDisplayExt for D {}
