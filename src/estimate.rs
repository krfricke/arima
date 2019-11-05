use num::Float;

use std::convert::From;
use std::cmp::min;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div};
use std::result::Result;

use lbfgs::*;
use optimization::{NumericalDifferentiation, Func, Function1};

use crate::{ArimaError, acf, util};

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
    ma: usize
) -> Result<Vec<f64>, ArimaError>{
    // Convert into f64 as the optimizer functions only support f64
    let mut x64: Vec<f64> = Vec::new();
    for i in 0..x.len() {
        x64.push(x[i].into());
    }
    let mut x = x64;

    if d > 0 {
        x = util::diff(&x, d);
    }
    let x = x;

    let total_size = 1 + ar + ma;

    // The objective is to minimize the conditional sum of squares (CSS),
    // i.e. the sum of the squared residuals
    let objective = NumericalDifferentiation::new(Func(
        |coef: &[f64]| {
            assert_eq!(coef.len(), total_size);

            let intercept = coef[0];
            let phi = &coef[1..ar+1];
            let theta = &coef[ar+1..];

            let residuals = residuals(&x, intercept, Some(&phi), Some(&theta)).unwrap();

            let mut css: f64 = 0.0;
            for i in 0..residuals.len() {
                css += residuals[i] * residuals[i];
            }
            css
        }
    ));

    // Initial coefficients
    // Todo: These initial guesses are rather arbitrary.
    let mut coef: Vec<f64> = Vec::new();

    // Initial guess for the intercept: First value of x
    coef.push(util::mean(&x));

    let rho = acf::acf(&x, None, false).unwrap();
    // Initial guess for the AR coefficients: Values of the PACF
    if ar > 0 {
        let pacf = acf::pacf_rho(&rho, Some(ar)).unwrap();
        for i in 0..ar {
            coef.push(pacf[i]);
        }
    }

    // Initial guess for the MA coefficients: 1.0
    if ma > 0 {
        for _ in 0..ma {
            coef.push(1.0);
        }
    }

    // Optimizer parameters
    // Todo: Are these heuristics sensible?
    let tolerance = 1e-14;
    let lbfgs_memory = 5;
    let max_iter: usize = 500;

    let mut lbfgs = Lbfgs::new(total_size, lbfgs_memory)
        .with_sy_epsilon(1e-8);

    // Initialize hessian at the origin.
    lbfgs.update_hessian(&objective.gradient(&vec![0.0; total_size]), &vec![0.0; total_size]);

    let mut gradient: Vec<f64>;
    for _ in 0..max_iter {
        // calculate gradient from coefficients
        gradient = objective.gradient(&coef);
        // update curvature information
        lbfgs.update_hessian(&gradient, &coef);
        // determine next direction
        lbfgs.apply_hessian(&mut gradient);

        let mut converged = true;
        for i in 0..gradient.len() {
            // Todo: We need to discover (and possibly mitigate?) divergence.
            if gradient[i].abs() > tolerance {
                converged = false;
            }
            coef[i] = coef[i] - gradient[i];
        }

        if converged {
            break;
        }
    }

    Ok(coef)
}