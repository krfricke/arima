use num::Float;

use std::convert::From;
use std::cmp::min;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div};
use std::result::Result;

use crate::ArimaError;

/// Calculate residuals given a time series, an intercept, and ARMA parameters
/// phi and theta. Any differencing and centering should be done before.
/// Squaring and summing the residuals yields the conditional sum of squares (CSS),
/// which can be used as an objective function to estimate the AR and MA parameters.
/// The variance can be then estimated via `CSS/(x.len()-phi.len())`.
///
/// # Arguments
///
/// * `&x` - Vector of the timeseries.
/// * `intercept` - Intercept parameter.
/// * `&phi` - AR parameter vector.
/// * `&theta` - MA parameter vector.
///
/// # Returns
///
/// * Vector of residuals. The first `phi.len()` items are zeros.
///
/// # Example
///
/// ```
/// use arima::estimate;
/// let x = [1.0, 1.2, 1.4, 1.6];
/// let res = estimate::residuals(&x, 0.0, Some(&[0.6, 0.4]), Some(&[0.3])).unwrap();
/// assert!((res[0] - 0.00).abs() < 1.0e-7);
/// assert!((res[1] - 0.00).abs() < 1.0e-7);
/// assert!((res[2] - 0.27999999).abs() < 1.0e-7);
/// assert!((res[3] - 0.196).abs() < 1.0e-7);
/// ```
pub fn residuals<T: Float + From<u32> + From<f64> + Copy + Add + AddAssign + Div + Debug>(
    x: &[T],
    intercept: T,
    phi: Option<&[T]>,
    theta: Option<&[T]>
) -> Result<Vec<T>, ArimaError>{
    let phi = match phi {
        Some(phi) => phi,
        None => &[]
    };
    let theta = match theta {
        Some(theta) => theta,
        None => &[]
    };

    if x.len() < phi.len() || x.len() < theta.len() {
        return Err(ArimaError);
    }

    let zero: T = From::from(0.0);

    let mut residuals: Vec<T> = Vec::new();
    for _ in 0..phi.len() {
        residuals.push(zero);
    }
    for t in phi.len()..x.len() {
        let mut xt: T = intercept;
        for j in 0..phi.len() {
            xt += phi[j] * x[t-j-1];
        }
        for j in 0..min(theta.len(), t) {
            xt += theta[j] * residuals[t-j-1];
        }
        residuals.push(x[t]-xt);
    }

    Ok(residuals)
}