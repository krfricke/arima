#[cfg(test)]
mod test_sim {
    extern crate rand;
    use rand::distributions::{Distribution, Normal};
    use rand::prelude::*;

    #[test]
    fn sim_ma() {
        let mut rng: StdRng = SeedableRng::from_seed([100; 32]);
        let normal = Normal::new(0.0, 2.0);

        let x = arima::sim::arima_sim(
            100,
            None,
            Some(&[0.4, 0.2]),
            0,
            &|mut rng| normal.sample(&mut rng),
            &mut rng,
        )
        .unwrap();

        // currently we only check if a timeseries was created correctly.
        // we do not check if it follows the given parameters.
        assert_eq!(x.len(), 100);
    }

    #[test]
    fn sim_ar() {
        let mut rng: StdRng = SeedableRng::from_seed([100; 32]);
        let normal = Normal::new(0.0, 2.0);

        let x = arima::sim::arima_sim(
            100,
            Some(&[0.9]),
            None,
            0,
            &|mut rng| normal.sample(&mut rng),
            &mut rng,
        )
        .unwrap();

        // It's hard to reliably test a simulated model, so we use a seed and calculate
        // the lag-1 PACF of an AR(1) model an expect it to match the coefficient
        let pacf = arima::acf::pacf(&x, Some(1)).unwrap()[0];

        assert!(pacf - 0.9 < 0.05);
    }
}
