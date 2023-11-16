extern crate rand;
use rand::distributions::{Distribution, Normal};
use rand::prelude::*;

use arima::{acf, sim};

fn main() {
    // initialize RNG with seed
    let mut rng: StdRng = SeedableRng::from_seed([100; 32]);

    // our noise should be normally distributed
    let normal = Normal::new(10.0, 2.0);

    // simulate time series
    let ts = sim::arima_sim(
        1000,                               // number of samples
        Some(&[0.7, 0.2]),                  // AR parameters
        None,                               // MA parameters
        0,                                  // difference parameter
        &|mut rng| normal.sample(&mut rng), // noise fn
        &mut rng,                           // RNG
    )
    .unwrap();

    // estimate AR parameters
    let ar = acf::ar(&ts, Some(2)).unwrap();

    println!("Estimated parameters: {:?}", ar);
    // Estimated parameters: [0.7436892808499717, 0.14774749031248915]
}
