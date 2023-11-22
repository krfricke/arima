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
/// use rand_distr::{Distribution, Normal};
///
/// let normal = Normal::new(0.0, 2.0).unwrap();
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
    d: usize,
    noise_fn: &dyn Fn(&mut T) -> f64,
    rng: &mut T,
) -> Result<Vec<f64>, ArimaError> {
    let mut x: Vec<f64> = Vec::new();

    // get orders
    let ar_order = match ar {
        Some(par) => par.len(),
        None => 0_usize,
    };
    let ma_order = match ma {
        Some(par) => par.len(),
        None => 0_usize,
    };

    // create some noise for the startup
    let burn_in = ar_order + ma_order + 10;
    for _ in 0..burn_in + n {
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
        for i in ma_order..burn_in + n {
            for j in 0..ma_order {
                x[i] += ma[j] * noise[i - j - 1];
            }
        }

        // set the un-regressed first 0..ma_order elements to zero
        for a in x.iter_mut().take(ma_order) {
            *a = 0.0
        }
    }

    // calculate AR part
    if ar_order > 0 {
        let ar = ar.unwrap();

        // the first 0..ma_order+ar_order are not regressed
        for i in ma_order + ar_order..burn_in + n {
            for j in 0..ar_order {
                x[i] += ar[j] * x[i - j - 1];
            }
        }
    }

    // remove burn_in part from vector, calculate differences
    if d > 0 {
        // also remove last d elements as there will be d zeros at the start
        x = util::diffinv(&x[burn_in..x.len() - d], d);
    } else {
        x.drain(0..burn_in);
    }

    Ok(x)
}

/// Forecast an ARIMA model time series
///
/// # Arguments
///
/// * `ts` - Time series to forecast from
/// * `n` - Length to forecast
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
/// use rand_distr::{Distribution, Normal};
///
/// let normal = Normal::new(0.0, 2.0).unwrap();
///
/// let ts = [0.632, 0.594, -2.750, -5.389, -5.645, -7.672, -12.595, -18.260, -24.147, -31.427];
///
/// let x = arima::sim::arima_forecast(
///     &ts,
///     100,
///     Some(&[0.9, -0.3, 0.2]),
///     Some(&[0.4, 0.2]),
///     1,
///     &|i, mut rng| { normal.sample(&mut rng) },
///     &mut thread_rng()
/// ).unwrap();
/// ```
pub fn arima_forecast<F: Fn(usize, &mut T) -> f64, T: Rng>(
    ts: &[f64],
    n: usize,
    ar: Option<&[f64]>,
    ma: Option<&[f64]>,
    d: usize,
    noise_fn: &F,
    rng: &mut T,
) -> Result<Vec<f64>, ArimaError> {
    let n_past = ts.len();
    let mut x = ts.to_vec();

    // get orders
    let ar_order = match ar {
        Some(par) => par.len(),
        None => 0_usize,
    };
    let ma_order = match ma {
        Some(par) => par.len(),
        None => 0_usize,
    };

    // initialize forecast with noise
    for i in 0..n {
        let e = noise_fn(i, rng);
        x.push(e);
    }

    // create further noise and calculate MA part
    if ma_order > 0 {
        let ma = ma.unwrap();
        let x_ = x.clone();
        for i in n_past..n_past + n {
            for j in 0..ma_order {
                x[i] += ma[j] * x_[i - j - 1];
            }
        }
    }

    // calculate AR part
    if ar_order > 0 {
        let ar = ar.unwrap();
        for i in n_past..n_past + n {
            for j in 0..ar_order {
                x[i] += ar[j] * x[i - j - 1];
            }
        }
    }

    // remove burn_in part from vector, calculate differences
    if d > 0 {
        x = util::diffinv(&x[n_past..x.len()], d);
        // drop the d zeros at the start
        x.drain(0..d);
    } else {
        x.drain(0..n_past);
    }

    Ok(x)
}
