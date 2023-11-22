use num::Float;

use std::cmp;
use std::convert::From;
use std::ops::{Add, AddAssign, Div};
use std::result::Result;

use crate::ArimaError;

/// Calculate the auto-correlation function of a time series of length n.
///
/// # Arguments
///
/// * `&x` - Reference to input vector slice of length n.
/// * `max_lag` - Calculate ACF for this maximum lag. Defaults to n-1.
/// * `covariance` - If true, returns auto-covariances. If false, returns auto-correlations.
///
/// # Returns
///
/// * Output vector of length max_lag+1.
///
/// # Example
///
/// ```
/// use arima::acf;
/// let x = [1.0, 1.2, 1.4, 1.6];
/// let ac = acf::acf(&x, Some(2), false).unwrap();
/// assert!((ac[0] - 1.0).abs() < 1.0e-7);
/// assert!((ac[1] - 0.25).abs() < 1.0e-7);
/// assert!((ac[2] - (-0.3)).abs() < 1.0e-7);
/// ```
pub fn acf<T: Float + From<u32> + From<f64> + Copy + Add + AddAssign + Div>(
    x: &[T],
    max_lag: Option<usize>,
    covariance: bool,
) -> Result<Vec<T>, ArimaError> {
    let max_lag = match max_lag {
        // if upper bound for max_lag is n-1
        Some(max_lag) => cmp::min(max_lag, x.len() - 1),
        None => x.len() - 1,
    };
    let m = max_lag + 1;

    let len_x_usize = x.len();
    let len_x: T = From::from(len_x_usize as u32);
    let sum: T = From::from(0.0);

    let sum_x: T = x.iter().fold(sum, |sum, &xi| sum + xi);
    let mean_x: T = sum_x / len_x;

    //let mut y: Vec<T> = Vec::with_capacity(max_lag);
    let mut y: Vec<T> = vec![From::from(0.0); m];

    for t in 0..m {
        for i in 0..len_x_usize - t {
            let xi = x[i] - mean_x;
            let xi_t = x[i + t] - mean_x;
            y[t] += (xi * xi_t) / len_x;
        }
        // we need y[0] to calculate the correlations, so we set it to 1.0 at the end
        if !covariance && t > 0 {
            y[t] = y[t] / y[0];
        }
    }
    if !covariance {
        y[0] = From::from(1.0);
    }
    Ok(y)
}

/// Calculate the auto-regressive coefficients of a time series of length n.
/// If you already calculated the auto-correlation coefficients (ACF), consider
/// using `ar_rho` instead.
///
/// # Arguments
///
/// * `&x` - Reference to input vector slice of length n.
/// * `order` - Order of the AR model.
///
/// # Returns
///
/// * Output vector of length order containing the AR coefficients.
///
/// # Example
///
/// ```
/// use arima::acf;
/// let x = [1.0, 1.2, 1.4, 1.6];
/// let (ar, _var) = acf::ar(&x, Some(2)).unwrap();
/// assert!((ar[0] - 0.3466667).abs() < 1.0e-7);
/// assert!((ar[1] - -0.3866667).abs() < 1.0e-7);
/// ```
pub fn ar<T: Float + From<u32> + From<f64> + Into<f64> + Copy + AddAssign>(
    x: &[T],
    order: Option<usize>,
) -> Result<(Vec<T>, T), ArimaError> {
    let max_lag = order.map(|order| order + 1);
    let rho = acf(x, max_lag, false)?;
    let cov0 = acf(x, Some(0), true)?[0];
    ar_dl_rho_cov(&rho, cov0, order)
}

/// Calculate the auto-regressive coefficients of a time series of length n, given
/// the auto-correlation coefficients rho. Uses LAPACK's DPOSV function to solve the
/// linear system and requires BLAS (e.g. OpenBLAS). Only enabled with feature `lapack`.
///
/// # Arguments
///
/// * `&rho` - Reference to auto-correlation coefficients rho.
/// * `order` - Order of the AR model.
///
/// # Returns
///
/// * Output vector of length order containing the AR coefficients.
///
/// # Example
///
/// ```
/// use arima::acf;
/// let x = [1.0, 1.2, 1.4, 1.6];
/// let rho = acf::acf(&x, None, false).unwrap();
/// let ar = acf::ar_lapack_rho(&rho, Some(2)).unwrap();
/// assert!((ar[0] - 0.3466667).abs() < 1.0e-7);
/// assert!((ar[1] - -0.3866667).abs() < 1.0e-7);
/// ```
#[cfg(feature = "lapack")]
pub fn ar_lapack_rho<T: Float + From<f64> + Into<f64> + Copy>(
    rho: &[T],
    order: Option<usize>,
) -> Result<Vec<T>, ArimaError> {
    // phi_0 will be calculated separately
    let n = match order {
        Some(order) => cmp::min(order, rho.len() - 1),
        None => rho.len() - 1,
    };

    // we try to solve mr * x = r for x

    // build lower triangle matrix
    let mut mr: Vec<f64> = vec![1.0; n * n];

    for i in 0..n {
        for j in i + 1..n {
            mr[i * n + j] = std::convert::Into::into(rho[j - i]);
        }
    }

    // build right hand vector rho_1..rho_n
    let mut b: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        b[i] = std::convert::Into::into(rho[i + 1]);
    }

    // build arguments to pass
    let mut info: i32 = 0;
    let ni = n as i32;

    // run lapack routine to solve symmetric positive-definite matrix system
    unsafe {
        lapack::dposv(b'L', ni, 1, &mut mr, ni, &mut b, ni, &mut info);
    }

    if info != 0 {
        return Err(ArimaError);
    }

    // convert back to T
    let mut phi: Vec<T> = vec![From::from(0.0); n];
    for i in 0..n {
        phi[i] = std::convert::Into::into(b[i]);
    }
    Ok(phi)
}

/// Calculate the auto-regressive coefficients of a time series of length n, given
/// the auto-correlation coefficients rho and auto covariance at lag 0, cov0.
/// This method uses the Durbin-Levinson algorithm to iteratively estimate the coefficients,
/// and it also returns the standard error for the 1-step look-ahead prediction (i.e. the
/// estimated variance).
///
/// # Arguments
///
/// * `&rho` - Reference to auto-correlation coefficients rho.
/// * `cov0` - Autocovariance at lag 0.
/// * `order` - Order of the AR model.
///
/// # Returns
///
/// * Tuple of an output vector containing the AR coefficients and an estimate for the variance.
///
/// # Example
///
/// ```
/// use arima::acf;
/// let x = [1.0, 1.2, 1.4, 1.6];
/// let rho = acf::acf(&x, None, false).unwrap();
/// let cov0 = acf::acf(&x, Some(0), true).unwrap()[0];
/// let (ar, err) = acf::ar_dl_rho_cov(&rho, cov0, Some(2)).unwrap();
/// assert!((ar[0] - 0.3466667).abs() < 1.0e-7);
/// assert!((ar[1] - -0.3866667).abs() < 1.0e-7);
/// ```
pub fn ar_dl_rho_cov<T: Float + From<u32> + From<f64> + Copy + Add + AddAssign + Div>(
    rho: &[T],
    cov0: T,
    order: Option<usize>,
) -> Result<(Vec<T>, T), ArimaError> {
    let order = match order {
        Some(order) => cmp::min(order, rho.len() - 1),
        None => rho.len() - 1,
    };

    // we need zero values more than once, so we'll use this helper var
    let zero = From::from(0.0);
    let one = From::from(1.0);

    // these vectors will hold the parameter values
    let mut phi: Vec<Vec<T>> = vec![Vec::new(); order + 1];
    let mut var: Vec<T> = Vec::new();

    // initialize zero-order estimates
    phi[0].push(zero);
    var.push(cov0);

    for i in 1..order + 1 {
        // first allocate values for the phi vector so we can use phi[i][i-1]
        for _ in 0..i {
            phi[i].push(zero);
        }

        // estimate phi_ii, which is stored as phi[i][i-1]
        // phi_i,i = rho(i) - sum_{k=1}^{n-1}(phi_{n-1,k} * rho(n-k) /
        //  (1 - sum_{k=1}^{n-1}(phi_{n-1,k} * rho(k))

        let mut num_sum = zero; // numerator sum
        let mut den_sum = one; // denominator sum

        for k in 1..i {
            let p = phi[i - 1][k - 1];
            num_sum += p * rho[i - k];
            den_sum += -p * rho[k];
        }

        let phi_ii = (rho[i] - num_sum) / den_sum;
        phi[i][i - 1] = phi_ii;

        var.push(var[i - 1] * (one - phi_ii * phi_ii));

        for k in 1..i {
            phi[i][k - 1] = phi[i - 1][k - 1] - phi[i][i - 1] * phi[i - 1][i - k - 1];
        }
    }

    Ok((phi[order].clone(), var[order]))
}

/// Estimate the variance of a time series of length n via Durbin-Levinson.
/// If you already calculated the AR parameters, auto-correlation coefficients (ACF), and
/// the auto-covariance for lag zero, consider using `var_phi_rho_cov` instead. Please note that
/// this might yield a different result.
///
/// # Arguments
///
/// * `&x` - Reference to input vector slice of length n.
/// * `order` - Order of the AR model. Defaults to n-1.
///
/// # Returns
///
/// * Estimated variance.
///
/// # Example
///
/// ```
/// use arima::acf;
/// let x = [1.0, 1.2, 1.4, 1.6];
/// acf::var(&x, Some(2));
/// ```
pub fn var<T: Float + From<u32> + From<f64> + Into<f64> + Copy + Add + AddAssign + Div>(
    x: &[T],
    order: Option<usize>,
) -> Result<T, ArimaError> {
    let max_lag = order.map(|order| order + 1);
    let rho = acf(x, max_lag, false)?;
    let cov0 = acf(x, Some(0), true)?[0];
    let (_phi, var) = ar_dl_rho_cov(&rho, cov0, order).unwrap();

    Ok(var)
}

/// Estimate the variance of a time series of length n, given the AR parameters,
/// auto-correlation coefficients (ACF), and the auto-covariance for lag zero.
///
/// # Arguments
///
/// * `&x` - Reference to input vector slice of length n.
/// * `order` - Order of the AR model. Defaults to n-1.
///
/// # Returns
///
/// * Estimated variance.
///
/// # Example
///
/// ```
/// use arima::acf;
/// let x = [1.0, 1.2, 1.4, 1.6];
/// let rho = acf::acf(&x, Some(3), false).unwrap();
/// let cov0 = acf::acf(&x, Some(0), true).unwrap()[0].clone();
/// let (phi, _var) = acf::ar_dl_rho_cov(&rho, cov0, Some(2)).unwrap();
/// acf::var_phi_rho_cov(&phi, &rho, cov0);
/// ```
pub fn var_phi_rho_cov<T: Float + From<u32> + From<f64> + Copy + Add + AddAssign + Div>(
    phi: &[T],
    rho: &[T],
    cov0: T,
) -> Result<T, ArimaError> {
    assert!(rho.len() > phi.len());

    let mut sum: T = From::from(0.0);
    for i in 0..phi.len() {
        sum += phi[i] * rho[i + 1];
    }
    let one: T = From::from(1.0);
    Ok(cov0 * (one - sum))
}

/// Calculate the partial auto-correlation coefficients of a time series of length n.
/// If you already calculated the auto-correlation coefficients (ACF), consider
/// using `pacf_rho` instead.
///
/// # Arguments
///
/// * `&x` - Reference to input vector slice of length n.
/// * `max_lag` - Maximum lag to calculate the PACF for. Defaults to n.
///
/// # Returns
///
/// * Output vector of length `max_lag`.
///
/// # Example
///
/// ```
/// use arima::acf;
/// let x = [1.0, 1.2, 1.4, 1.6];
/// let pr = acf::pacf(&x, Some(2)).unwrap();
/// assert!((pr[0] - 0.25).abs() < 1.0e-7);
/// assert!((pr[1] - -0.3866667).abs() < 1.0e-7);
/// ```
pub fn pacf<T: Float + From<u32> + From<f64> + Into<f64> + Copy + AddAssign>(
    x: &[T],
    max_lag: Option<usize>,
) -> Result<Vec<T>, ArimaError> {
    // get autocorrelations
    let rho = acf(x, max_lag, false)?;
    let cov0 = acf(x, Some(0), true)?[0];
    pacf_rho_cov0(&rho, cov0, max_lag)
}

/// Calculate the partial auto-correlation coefficients of a time series of length n, given
/// the auto-correlation coefficients rho.
///
/// # Arguments
///
/// * `&rho` - Reference to auto-correlation coefficients rho.
/// * `max_lag` - Maximum lag to calculate the PACF for. Defaults to n.
///
/// # Returns
///
/// * Output vector of length `max_lag`.
///
/// # Example
///
/// ```
/// use arima::acf;
/// let x = [1.0, 1.2, 1.4, 1.6];
/// let rho = acf::acf(&x, None, false).unwrap();
/// let cov0 = acf::acf(&x, Some(0), true).unwrap()[0];
/// let pr = acf::pacf_rho_cov0(&rho, cov0, Some(2)).unwrap();
/// assert!((pr[0] - 0.25).abs() < 1.0e-7);
/// assert!((pr[1] - -0.3866667).abs() < 1.0e-7);
/// ```
pub fn pacf_rho_cov0<T: Float + From<u32> + From<f64> + Into<f64> + Copy + AddAssign>(
    rho: &[T],
    cov0: T,
    max_lag: Option<usize>,
) -> Result<Vec<T>, ArimaError> {
    let max_lag = match max_lag {
        // if upper bound for max_lag is n-1
        Some(max_lag) => cmp::min(max_lag, rho.len() - 1),
        None => rho.len() - 1,
    };
    let m = max_lag + 1;

    // build output vector
    let mut y: Vec<T> = Vec::new();

    // calculate AR coefficients for each solution of order 1..max_lag
    for i in 1..m {
        let (coef, _var) = ar_dl_rho_cov(rho, cov0, Some(i))?;
        // we now have a vector with i items, the last item is our partial correlation
        y.push(coef[i - 1]);
    }
    Ok(y)
}
