extern crate num;

use num::{Num, NumCast, Float};
use core::ops::{Neg, Sub};
use std::fmt::Debug;

pub fn lag<T: Num + Copy>(x: &Vec<T>, tau: u32) -> Vec<T> {
    let mut y: Vec<T> = Vec::new();
    assert!(tau < x.len() as u32);
    for i in tau as usize..x.len() {
        y.push(x[i]);
    }
    y
}

/// Returns a n-1 vector containing the pairwise difference x_t - x_t-1
///
/// # Arguments
///
/// * `&x` - Reference to input vector of length n
///
/// # Returns
///
/// * Output vector of length n-1
///
/// # Example
///
/// ```
/// use arima::util;
/// let x = vec![1, 2, 3];
/// util::diff(&x);
/// // returns equivalent to vec![1, 1];
/// ```
pub fn diff<T: Num + Copy + Neg<Output=T> + Sub>(x: &Vec<T>) -> Vec<T> {
    let mut y: Vec<T> = Vec::new();
    for i in 1..x.len() {
        y.push(x[i] - x[i-1]);
    }
    y
}

pub fn diff_log<T: Float>(x: &Vec<T>) -> Vec<T> {
    let mut y: Vec<T> = Vec::new();
    for i in 1..x.len() {
        y.push((x[i].ln() - x[i-1].ln()));
    }
    y
}

