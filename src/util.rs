extern crate num;

use core::ops::{Neg, Sub};
use num::{Num, Float};
use std::ops::{Add, AddAssign};


pub fn lag<T: Num + Copy>(x: &[T], tau: u32) -> Vec<T> {
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
pub fn diff<T: Num + Copy + Neg<Output=T> + Sub>(x: &[T]) -> Vec<T> {
    let mut y: Vec<T> = Vec::new();
    for i in 1..x.len() {
        y.push(x[i] - x[i-1]);
    }
    y
}

pub fn diff_log<T: Float>(x: &[T]) -> Vec<T> {
    let mut y: Vec<T> = Vec::new();
    for i in 1..x.len() {
        y.push(x[i].ln() - x[i-1].ln());
    }
    y
}

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

pub fn diffinv<T: Num + Add + AddAssign + Copy + From<u8>>(x: &[T], differences: u32) -> Vec<T> {
    let mut y: Vec<T> = Vec::new();
    let zero = From::from(0);

    // build cumulative sum n times where n is the order of differences
    let mut cum: Vec<T> = From::from(x);
    for i in 0..differences {
        y.push(zero);
        cum = cumsum(&cum);
    }

    // append the cumsum to the result vector
    for i in 0..cum.len() {
        y.push(cum[i]);
    }
    y
}