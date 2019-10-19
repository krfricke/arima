![https://crates.io/crates/arima](https://img.shields.io/crates/v/arima.svg)
![https://docs.rs/arima](https://docs.rs/arima/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/krfricke/arima/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/krfricke/arima.svg?branch=master)](https://travis-ci.org/krfricke/arima)
# ARIMA

Rust crate for ARIMA model coefficient estimation and simulation.

Please note that this crate relies on [LAPACK](https://crates.io/crates/lapack) which needs a working
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) implementation. Per default, this
crate looks for a system-installed OpenBLAS. On Debian and Ubuntu, you can install this via
`apt install libopenblas-base libopenblas-dev`.

## Example

```rust
extern crate rand;
use rand::prelude::*;
use rand::distributions::{Normal, Distribution};

use arima::{acf, sim};

fn main() {
    // initialize RNG with seed
    let mut rng: StdRng = SeedableRng::from_seed([100; 32]);

    // our noise should be normally distributed
    let normal = Normal::new(10.0, 2.0);

    // simulate time series
    let ts = sim::arima_sim(
        1000,                   // number of samples
        Some(&[0.7, 0.2]),      // AR parameters
        None,                   // MA parameters
        0,                      // difference parameter
        &|mut rng| { normal.sample(&mut rng) }, // noise fn
        &mut rng                // RNG
    ).unwrap();

    // estimate AR parameters
    let ar = acf::ar(&ts, Some(2)).unwrap();

    println!("Estimated parameters: {:?}", ar);
    // Estimated parameters: [0.7436892808499717, 0.14774749031248915]
}
```

## Features

- Auto-correlation/covariance calculation
- Partial auto-correlation calculation
- AR parameter estimation
- Variance estimation
- ARIMA time series simulation

## Roadmap

- Full ARIMA model parameter estimation
- Order estimation

# License

This crate is licensed under the [Apache-2.0](LICENSE) license.