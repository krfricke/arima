use anyhow::Result;

use num::Float;

use std::cmp::min;
use std::convert::From;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div};

use finitediff::FiniteDiff;
use liblbfgs::lbfgs;

use crate::{acf, util};

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
    theta: Option<&[T]>,
) -> Result<Vec<T>> {
    let phi = phi.unwrap_or(&[]);
    let theta = theta.unwrap_or(&[]);

    if x.len() < phi.len() || x.len() < theta.len() {
        anyhow::bail!("Too many items in phi or theta");
    }

    let zero: T = From::from(0.0);

    let mut residuals: Vec<T> = Vec::new();
    for _ in 0..phi.len() {
        residuals.push(zero);
    }
    for t in phi.len()..x.len() {
        let mut xt: T = intercept;
        for j in 0..phi.len() {
            xt += phi[j] * x[t - j - 1];
        }
        for j in 0..min(theta.len(), t) {
            xt += theta[j] * residuals[t - j - 1];
        }
        residuals.push(x[t] - xt);
    }

    Ok(residuals)
}

/// Fit an ARIMA model. Returns the fitted coefficients.
/// This method uses the L-BFGS algorithm and the conditional sum of squares (CSS)
/// as the objective function.
///
/// # Arguments
///
/// * `&x` - Vector of the timeseries.
/// * `ar` - Order of the AR coefficients.
/// * `d` - Order of differencing.
/// * `ma` - Order of the MA coefficients.
///
/// # Returns
///
/// * ARIMA coefficients minimizing the conditional sum of squares (CSS).
///
/// # Example
///
/// ```
/// use arima::estimate;
/// let x = [1.0, 1.2, 1.4, 1.6, 1.4, 1.2, 1.0];
/// let coef = estimate::fit(&x, 0, 0, 1).unwrap();
/// assert!((coef[0] - 1.2051).abs() < 1.0e-3); // intercept
/// assert!((coef[1] - 0.5637).abs() < 1.0e-3); // phi_1
/// ```
pub fn fit<T: Float + From<u32> + From<f64> + Into<f64> + Copy + Add + AddAssign + Div + Debug>(
    x: &[T],
    ar: usize,
    d: usize,
    ma: usize,
) -> Result<Vec<f64>> {
    // Convert into f64 as the optimizer functions only support f64
    let mut x64: Vec<f64> = Vec::new();
    for a in x {
        x64.push((*a).into());
    }
    let mut x = x64;

    if d > 0 {
        x = util::diff(&x, d);
    }
    let x = x;

    let total_size = 1 + ar + ma;

    // The objective is to minimize the conditional sum of squares (CSS),
    // i.e. the sum of the squared residuals
    let f = |coef: &Vec<f64>| {
        assert_eq!(coef.len(), total_size);

        let intercept = coef[0];
        let phi = &coef[1..ar + 1];
        let theta = &coef[ar + 1..];

        let residuals = residuals(&x, intercept, Some(phi), Some(theta)).unwrap();

        let mut css: f64 = 0.0;
        for residual in &residuals {
            css += residual * residual;
        }
        css
    };
    let g = |coef: &Vec<f64>| coef.forward_diff(&f);

    // Initial coefficients
    // Todo: These initial guesses are rather arbitrary.
    let mut coef: Vec<f64> = Vec::new();

    // Initial guess for the intercept: First value of x
    coef.push(util::mean(&x));

    // Initial guess for the AR coefficients: Values of the PACF
    if ar > 0 {
        let pacf = acf::pacf(&x, Some(ar)).unwrap();
        for p in pacf {
            coef.push(p);
        }
    }

    // Initial guess for the MA coefficients: 1.0
    if ma > 0 {
        coef.resize(coef.len() + ma, 1.0);
    }

    let evaluate = |x: &[f64], gx: &mut [f64]| {
        let x_vec = x.to_vec();
        let fx = f(&x_vec);
        let gx_eval = g(&x_vec);
        // copy values from gx_eval into gx
        gx[..gx_eval.len()].copy_from_slice(&gx_eval[..]);
        Ok(fx)
    };

    let fmin = lbfgs().with_max_iterations(200);
    if let Err(e) = fmin.minimize(
        &mut coef, // input variables
        evaluate,  // define how to evaluate function
        |_prgr| {
            false // returning true will cancel optimization
        },
    ) {
        tracing::warn!("Got error during fit: {}", e);
    }

    Ok(coef)
}

/// TODO clean up
/// Auto-fit an ARIMA model, guessing AR and MA orders.
/// See `fit` for more details.
///
/// # Arguments
///
/// * `&x` - Vector of the timeseries.
/// * `d` - Order of differencing.
///
/// # Returns
///
/// * ARIMA coefficients minimizing the conditional sum of squares (CSS).
pub fn autofit<
    T: Float + From<u32> + From<f64> + Into<f64> + Copy + Add + AddAssign + Div + Debug,
>(
    x: &[T],
    d: usize,
) -> Result<Vec<f64>> {
    let x: Vec<f64> = x.iter().map(|v| (*v).into()).collect();
    let n = x.len() as f64;
    let n_lags = 12;

    // Hardcoding for now
    // let alpha = 0.05;
    // ppf = scipy.stats.norm.ppf(1 - alpha / 2.0)
    let ppf = 1.959963984540054;

    // Estimate MA order
    // <https://www.statsmodels.org/devel/_modules/statsmodels/tsa/stattools.html#acf>
    let _acf = acf::acf(&x, Some(n_lags), false).unwrap();
    let mult: Vec<f64> = _acf[1.._acf.len() - 1]
        .iter()
        .scan(0., |acc, v| {
            *acc += v.powf(2.);
            Some(1. + 2. * *acc)
        })
        .collect();
    let mut varacf = vec![0., 1. / n];
    let varacf_end: Vec<f64> = (0.._acf.len() - 2).map(|i| 1. / n * mult[i]).collect();
    varacf.extend(varacf_end);

    let interval: Vec<f64> = varacf.iter().map(|v| ppf * v.sqrt()).collect();
    let confint: Vec<(f64, f64)> = _acf
        .iter()
        .zip(&interval)
        .map(|(p, q)| (p - q, p + q))
        .collect();
    let bounds: Vec<(f64, f64)> = confint
        .iter()
        .zip(&_acf)
        .map(|((l, u), a)| (l - a, u - a))
        .collect();

    // Subtract one to compensate for the first value (lag=0)
    let ma_order = _acf
        .iter()
        .zip(bounds)
        .take_while(|(a, (l, u))| a < &l || a > &u)
        .count()
        - 1;

    // <https://www.statsmodels.org/devel/_modules/statsmodels/tsa/stattools.html#pacf>
    let _pacf = acf::pacf(&x, Some(n_lags)).unwrap();
    let pacf_varacf = 1.0 / n;
    let pacf_interval = ppf * pacf_varacf.sqrt();
    let pacf_confint: Vec<(f64, f64)> = _pacf
        .iter()
        .map(|p| (p - pacf_interval, p + pacf_interval))
        .collect();

    let pacf_bounds: Vec<(f64, f64)> = pacf_confint
        .iter()
        .zip(&_pacf)
        .map(|((l, u), a)| (l - a, u - a))
        .collect();

    // lag=0 isn't included so no need to subtract one
    let ar_order = _pacf
        .iter()
        .zip(pacf_bounds)
        .take_while(|(a, (l, u))| a < &l || a > &u)
        .count();

    fit(&x, ar_order, d, ma_order)
}
