// MIT/Apache2 License

use breadx::Display;

pub trait DisplayExt: Display {

}

impl<D: Display + ?Sized> DisplayExt for D {}