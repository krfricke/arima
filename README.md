[![Crate](https://img.shields.io/crates/v/arima.svg)](https://crates.io/crates/arima)
[![Docs](https://docs.rs/arima/badge.svg)](https://docs.rs/arima)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/krfricke/arima/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/krfricke/arima.svg?branch=master)](https://travis-ci.org/krfricke/arima)
# ARIMA

Rust crate for ARIMA model coefficient estimation and simulation.

## Example

```rust
extern crate rand;
use rand::prelude::*;
use rand::distributions::{Normal, Distribution};

use arima::{estimate, sim};

fn main() {
    // initialize RNG with seed
    let mut rng: StdRng = SeedableRng::from_seed([100; 32]);

    // our noise should be normally distributed
    let normal = Normal::new(10.0, 2.0);

    // simulate time series
    let ts = sim::arima_sim(
        1000,                   // number of samples
        Some(&[0.7, 0.2]),      // AR parameters
        Some(&[0.4]),                   // MA parameters
        0,                      // difference parameter
        &|mut rng| { normal.sample(&mut rng) }, // noise fn
        &mut rng                // RNG
    ).unwrap();

    // estimate AR parameters
    let coef = estimate::fit(&ts, 2, 0, 1).unwrap();

    println!("Estimated parameters: {:?}", coef);
    // Estimated parameters: [14.904840907703845, 0.7524268545022731, 0.14075584488434256, 0.35966423499627603]
}
```

## Features

- Full ARIMA model parameter estimation
- Auto-correlation/covariance calculation
- Partial auto-correlation calculation
- AR parameter estimation
- Variance estimation
- ARIMA time series simulation

## Roadmap

- Order estimation

# License

This crate is licensed under the [Apache-2.0](LICENSE) license.