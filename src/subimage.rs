//               Copyright John Nunley, 2022.
// Distributed under the Boost Software License, Version 1.0.
//       (See accompanying file LICENSE or copy at
//         https://www.boost.org/LICENSE_1_0.txt)

use super::Image;
use breadx::{
    display::{from_void_request, RawRequest},
    protocol::xproto::{Drawable, Gcontext},
};

impl<Storage: AsRef<[u8]> + ?Sized> Image<Storage> {
    /// Get a `PutImageRequest` that could be used to send a part of this
    /// image to the display.
    #[allow(clippy::too_many_arguments)]
    pub fn put_subimage_request<R>(
        &self,
        drawable: impl Into<Drawable>,
        gc: impl Into<Gcontext>,
        src_x: usize,
        src_y: usize,
        width: usize,
        height: usize,
        dst_x: i16,
        dst_y: i16,
        discard_reply: bool,
        raw_request_handler: impl FnOnce(RawRequest<'_, '_>) -> R,
    ) -> R {
        let drawable = drawable.into();
        let gc = gc.into();

        // simple case: src_x/y are 0, and width/height are equal to image's
        // then this degenerates into a put_image_request() call
        if (src_x, src_y, width, height) == (0, 0, self.width as usize, self.height as usize) {
            let pir = self.put_image_request(drawable, gc, dst_x, dst_y);
            return from_void_request(pir, discard_reply, raw_request_handler);
        }

        // optimized case: if we are a ZPixmap image, and the first pixel of each row of a theoretical
        // cropped image is on a byte boundary, then we can create a raw request uses I/O slices to
        // contain each row of the cropped image. this ensures no actual byte copying is done, which
        // in the best case can save a lot of time.

        // TODO

        // degenerate case: no optimization is possible, this simplifies down to
        // crop() then put_image_request()
        let cropped = self.crop(src_x, src_y, width, height);
        let pir = cropped.put_image_request(drawable, gc, dst_x, dst_y);
        from_void_request(pir, discard_reply, raw_request_handler)
    }
}
