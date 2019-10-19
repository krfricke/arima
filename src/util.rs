extern crate num;

use core::ops::{Neg, Sub};
use num::{Num, Float};
use std::ops::{Add, AddAssign};

/// Returns a n-tau vector containing the time series lagged by tau.
///
/// # Arguments
///
/// * `&x` - Reference to input vector slice of length n.
/// * `tau` - Lag.
///
/// # Returns
///
/// * Output vector of length n-tau.
///
/// # Example
///
/// ```
/// use arima::util;
/// let x = [-4, -9, 20, 23, -18, 6];
/// let y = [20, 23, -18, 6];
/// assert_eq!(util::lag(&x, 2), y);
/// ```
pub fn lag<T: Num + Copy>(x: &[T], tau: u32) -> Vec<T> {
    let mut y: Vec<T> = Vec::new();
    assert!(tau < x.len() as u32);
    for i in tau as usize..x.len() {
        y.push(x[i]);
    }
    y
}

/// Returns a n-1 vector containing the pairwise difference x_t - x_t-1.
///
/// # Arguments
///
/// * `&x` - Reference to input vector slice of length n.
/// * `d` - Number of differences to be taken.
///
/// # Returns
///
/// * Output vector of length n-1.
///
/// # Example
///
/// ```
/// use arima::util;
/// let x = [1, 2, 3];
/// util::diff(&x, 1);
/// assert_eq!(x, [1, 1])
/// ```
pub fn diff<T: Num + Copy + Neg<Output=T> + Sub>(x: &[T], d: u32) -> Vec<T> {
    let d = d as usize;
    let mut y: Vec<T> = x.to_vec().clone();
    let mut z = y.clone();
    for s in 0..d {
        for i in d..x.len() {
            z[i] = y[i] - y[i - 1];
        }
        y = z.clone();
    }
    y[d..].to_vec()
}

/// Returns a n-1 vector containing the pairwise difference of log(x_t) - log(x_t-1).
///
/// # Arguments
///
/// * `&x` - Reference to input vector slice of length n.
///
/// # Returns
///
/// * Output vector of length n-1.
///
/// # Example
///
/// ```
/// use arima::util;
/// let x = [1, 4, 6];
/// util::diff_log(&x);
/// assert_eq!(x, [1.3862944, 0.4054651]);
/// ```
pub fn diff_log<T: Float>(x: &[T]) -> Vec<T> {
    let mut y: Vec<T> = Vec::new();
    for i in 1..x.len() {
        y.push(x[i].ln() - x[i-1].ln());
    }
    y
}

/// Calculate the cumulative sum of a vector.
///
/// # Arguments
///
/// * `&x` - Reference to input vector slice of length n.
///
/// # Returns
///
/// * Output vector of length n containing the cumulated values.
///
/// # Example
///
/// ```
/// use arima::util;
/// let x = [1, 2, 3];
/// util::cumsum(&x);
/// assert_eq!(x, [1, 3, 6]);
/// ```
pub fn cumsum<T: Num + Add + AddAssign + Copy + From<u8>>(x: &[T]) -> Vec<T> {
    let mut y: Vec<T> = Vec::new();
    if x.len() < 2 {
        y.push(From::from(0));
        return y;
    }
    y.push(x[0].clone());
    for i in 1..x.len() {
        y.push(y[i-1] + x[i]);
    }
    y
}

/// Calculate the inverse difference of a vector.
///
/// # Arguments
///
/// * `&x` - Reference to input vector slice of length n.
/// * `d` - How often the inverse differences should be applied.
///
/// # Returns
///
/// * Output vector of length n+d containing the inversed values. The first d values are zero.
///
/// # Example
///
/// ```
/// use arima::util;
/// let x = [1, 1, 1, 1, 1];
/// let y = util::diffinv(&x, 1);
/// assert_eq!(y, [1, 2, 3, 4, 5]);
///
/// let z = util::diff(&y, 1);
/// assert_eq!(x, y);
/// ```
pub fn diffinv<T: Num + Add + AddAssign + Copy + From<u8>>(x: &[T], d: u32) -> Vec<T> {
    let mut y: Vec<T> = Vec::new();
    let zero = From::from(0);

    // build cumulative sum n times where n is the order of differences
    let mut cum: Vec<T> = From::from(x);
    for _ in 0..d {
        y.push(zero);
        cum = cumsum(&cum);
    }

    // append the cumsum to the result vector
    for i in 0..cum.len() {
        y.push(cum[i]);
    }
    y
}