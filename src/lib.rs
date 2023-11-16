pub mod acf;
pub mod sim;
pub mod util;

pub mod estimate;

#[derive(Debug, Clone)]
pub struct ArimaError;

#[cfg(feature = "accelerate")]
extern crate accelerate_src as raw;

#[cfg(feature = "intel-mkl")]
extern crate intel_mkl_src as raw;

#[cfg(feature = "netlib")]
extern crate netlib_src as raw;

#[cfg(feature = "openblas")]
extern crate openblas_src as raw;

#[cfg(feature = "lapack")]
extern crate lapack;
#[cfg(feature = "lapack")]
extern crate lapack_sys;

extern crate num;
