use crate::ArimaError;

use crate::util;
use rand::Rng;

/// Simulate an ARIMA model time series
///
/// # Arguments
///
/// * `n` - Length of the time series
/// * `ar` - Model parameters for the AR part
/// * `ma` - Model parameters for the MA part
/// * `d` - Model parameter for the differences
/// * `noise_fn` - Function that takes a `Rng' as input and returns noise
/// * `rng` - Reference to a mutable `Rng`.
///
/// # Returns
///
/// * Output vector of length n containing the time series data.
///
/// # Example
///
/// ```
/// use rand::prelude::*;
/// use rand::distributions::{Normal, Distribution};
///
/// let normal = Normal::new(0.0, 2.0);
///
/// let x = arima::sim::arima_sim(
///     100,
///     Some(&[0.9, -0.3, 0.2]),
///     Some(&[0.4, 0.2]),
///     1,
///     &|mut rng| { normal.sample(&mut rng) },
///     &mut thread_rng()
/// ).unwrap();
/// ```
pub fn arima_sim<T: Rng>(
    n: usize,
    ar: Option<&[f64]>,
    ma: Option<&[f64]>,
    d: u32,
    noise_fn: &dyn Fn(&mut T) -> f64,
    rng: &mut T
) -> Result<Vec<f64>, ArimaError> {
    let mut x: Vec<f64> = Vec::new();

    // get orders
    let ar_order = match ar {
        Some(par) => par.len(),
        None => 0 as usize
    };
    let ma_order = match ma {
        Some(par) => par.len(),
        None => 0 as usize
    };

    // create some noise for the startup
    let burn_in = ar_order + ma_order + 10;
    for _ in 0..burn_in+n {
        let e = noise_fn(rng);
        x.push(e);
    }

    // create further noise and calculate MA part
    if ma_order > 0 {
        let ma = ma.unwrap();
        // x currently contains only noise
        // copy into noise vector for MA regression
        let noise = x.clone();
        // the first 0..ma_order elements are not regressed
        for i in ma_order..burn_in+n {
            for j in 0..ma_order {
                x[i] += ma[j] * noise[i - j - 1];
            }
        }

        // set the un-regressed first 0..ma_order elements to zero
        for i in 0..ma_order {
            x[i] = 0.0
        }
    }

    // calculate AR part
    if ar_order > 0 {
        let ar = ar.unwrap();

        // the first 0..ma_order+ar_order are not regressed
        for i in ma_order+ar_order..burn_in+n {
            for j in 0..ar_order {
                x[i] += ar[j] * x[i - j - 1];
            }
        }
    }

    // remove burn_in part from vector, calculate differences
    if d > 0 {
        // also remove last d elements as there will be d zeros at the start
        x = util::diffinv(&x[burn_in..x.len()-d as usize], d);
    } else {
        x = Vec::from(&x[burn_in..]);
    }

    Ok(x)
}