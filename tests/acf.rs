#[cfg(test)]
mod test_acf {
    use more_asserts::assert_lt;

    #[test]
    fn acf_cov_f64() {
        // AR(2) model with phi=[0.5, 0.2], mean=13.0, sd=2.0
        let x = vec![22.71659, 23.24932, 24.86742, 25.19197, 22.92390, 24.80207, 25.71119, 25.90546, 21.85956, 24.35609, 30.51819, 25.80506];
        // auto covariance
        let acf_real = vec![4.58489144, 0.38749482, -1.91179140, 0.28256939, 1.35258379, -0.06345611, -1.22621493, 0.21676391, 0.63269957];
        let acf_calc = arima::acf::acf_cov(&x, Some(8));

        println!("Res: {:?}", acf_calc);

        assert_eq!(acf_real.len(), acf_calc.len());

        for i in 0..acf_real.len() {
            println!("Diff: {:?}", (acf_real[i] - acf_calc[i] as f64).abs());
            assert_lt!((acf_real[i] - acf_calc[i] as f64).abs(), 1.0e-7);
        }
    }
}