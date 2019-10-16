use crate::util;

use num::Float;
use std::cmp;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div};

pub fn acf_cov<T: Float + From<u32> + From<f64> + Copy + Sum + Add + AddAssign + Div>(x: &Vec<T>, max_lag: Option<u32>) -> Vec<T> {
    let max_lag = match max_lag {
        // max lag should be inclusive, so add 1
        Some(max_lag) => cmp::min(max_lag+1, x.len() as u32) as usize,
        None => x.len() as usize
    };

    let len_x_usize = x.len();
    let len_x: T = std::convert::From::from(len_x_usize as u32);
    let mut sum: T = std::convert::From::from(0.0);

    let sum_x: T = x.iter().fold(sum, |sum, &xi| sum + xi);
    let mean_x: T = sum_x / len_x;

    //let mut y: Vec<T> = Vec::with_capacity(max_lag);
    let mut y: Vec<T> = vec![std::convert::From::from(0.0); max_lag];

    for t in 0..max_lag {
        for i in 0..len_x_usize-t {
            let xi = x[i] - mean_x;
            let xi_t = x[i+t] - mean_x;
            y[t] = y[t] + (xi * xi_t) / len_x;
        }
    }
    y
}