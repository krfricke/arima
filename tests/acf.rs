#[cfg(test)]
mod test_acf {
    use more_asserts::assert_lt;

    #[test]
    fn acf_cov_full_f64() {
        // AR(2) model with phi=[0.8, -0.42], mean=20.0, sd=4.0
        let x = vec![99.97373, 104.86273, 103.04437, 105.23321, 101.36082, 99.22805, 92.31765, 94.61127, 94.22430, 99.60984, 96.72139, 91.98367];
        // auto covariance
        let acf_real = vec![19.6438949, 10.8881122, 5.7696993, 0.3208599, -1.6602776, -3.5061394, -4.6417722, -4.9257720, -4.3255709, -3.3143572, -3.6682543];
        let acf_calc = arima::acf::acf(&x, None, true).unwrap();

        assert_eq!(acf_real.len(), acf_calc.len());

        for i in 0..acf_real.len() {
            assert_lt!((acf_real[i] - acf_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn acf_cov_maxlag_f64() {
        // AR(2) model with phi=[0.5, 0.2], mean=13.0, sd=2.0
        let x = vec![22.71659, 23.24932, 24.86742, 25.19197, 22.92390, 24.80207, 25.71119, 25.90546, 21.85956, 24.35609, 30.51819, 25.80506];
        // auto covariance
        let acf_real = vec![4.58489144, 0.38749482, -1.91179140, 0.28256939, 1.35258379];
        let acf_calc = arima::acf::acf(&x, Some(4), true).unwrap();

        assert_eq!(acf_real.len(), acf_calc.len());

        for i in 0..acf_real.len() {
            assert_lt!((acf_real[i] - acf_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn acf_cor_full_f64() {
        // AR(2) model with phi=[0.8, -0.42], mean=20.0, sd=4.0
        let x = vec![99.97373, 104.86273, 103.04437, 105.23321, 101.36082, 99.22805, 92.31765, 94.61127, 94.22430, 99.60984, 96.72139, 91.98367];
        // auto covariance
        let acf_real = vec![1.0, 0.55427461, 0.29371463, 0.01633382, -0.08451876, -0.17848494, -0.23629592, -0.25075333, -0.22019925, -0.16872200, -0.18673763];
        let acf_calc = arima::acf::acf(&x, None, false).unwrap();

        assert_eq!(acf_real.len(), acf_calc.len());

        for i in 0..acf_real.len() {
            assert_lt!((acf_real[i] - acf_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn acf_cor_maxlag_f64() {
        // AR(2) model with phi=[0.5, 0.2], mean=13.0, sd=2.0
        let x = vec![22.71659, 23.24932, 24.86742, 25.19197, 22.92390, 24.80207, 25.71119, 25.90546, 21.85956, 24.35609, 30.51819, 25.80506];
        // auto covariance
        let acf_real = vec![1.0, 0.08451559, -0.41697637, 0.06163055, 0.29500890];
        let acf_calc = arima::acf::acf(&x, Some(4), false).unwrap();

        assert_eq!(acf_real.len(), acf_calc.len());

        for i in 0..acf_real.len() {
            assert_lt!((acf_real[i] - acf_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn ar_coef_full_f64() {
        // AR(2) model with phi=[0.8, -0.42], mean=20.0, sd=4.0
        let x = vec![99.97373, 104.86273, 103.04437, 105.23321, 101.36082, 99.22805, 92.31765, 94.61127, 94.22430, 99.60984, 96.72139, 91.98367];
        // auto covariance
        let acf_calc = arima::acf::acf(&x, None, false).unwrap();

        let ar_real = vec![0.519461308, 0.067499687, -0.213575357, 0.021804873, -0.040571104, -0.087928887, -0.072462752, -0.014305985, 0.031606043, -0.159437957];
        let ar_calc = arima::acf::ar_coef_rho(&acf_calc, None).unwrap();

        assert_eq!(ar_real.len(), ar_calc.len());

        for i in 0..ar_real.len() {
            assert_lt!((ar_real[i] - ar_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn ar_coef_order_f64() {
        // AR(2) model with phi=[0.8, -0.42], mean=20.0, sd=4.0
        let x = vec![99.97373, 104.86273, 103.04437, 105.23321, 101.36082, 99.22805, 92.31765, 94.61127, 94.22430, 99.60984, 96.72139, 91.98367];
        // auto covariance
        let acf_calc = arima::acf::acf(&x, None, false).unwrap();

        let ar_real = vec![0.558121536, 0.095217133, -0.19193744, -0.015215631];
        let ar_calc = arima::acf::ar_coef_rho(&acf_calc, Some(4)).unwrap();

        assert_eq!(ar_real.len(), ar_calc.len());

        for i in 0..ar_real.len() {
            assert_lt!((ar_real[i] - ar_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn pacf_full_f64() {
        // AR(2) model with phi=[0.8, -0.42], mean=20.0, sd=4.0
        let x = vec![99.97373, 104.86273, 103.04437, 105.23321, 101.36082, 99.22805, 92.31765, 94.61127, 94.22430, 99.60984, 96.72139, 91.98367];
        // auto covariance
        let acf_calc = arima::acf::acf(&x, None, false).unwrap();

        let pacf_real = vec![0.554274612, -0.01949497, -0.200476024, -0.015215631, -0.102418265, -0.126449729, -0.071949601, -0.053608797, -0.052551695, -0.159437957];
        let pacf_calc = arima::acf::pacf_rho(&acf_calc, None).unwrap();

        assert_eq!(pacf_real.len(), pacf_calc.len());

        for i in 0..pacf_real.len() {
            assert_lt!((pacf_real[i] - pacf_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn pacf_maxlag_f64() {
        // AR(2) model with phi=[0.8, -0.42], mean=20.0, sd=4.0
        let x = vec![99.97373, 104.86273, 103.04437, 105.23321, 101.36082, 99.22805, 92.31765, 94.61127, 94.22430, 99.60984, 96.72139, 91.98367];
        // auto covariance
        let acf_calc = arima::acf::acf(&x, None, false).unwrap();

        let pacf_real = vec![0.554274612, -0.01949497, -0.200476024, -0.015215631];
        let pacf_calc = arima::acf::pacf_rho(&acf_calc, Some(4)).unwrap();

        assert_eq!(pacf_real.len(), pacf_calc.len());

        for i in 0..pacf_real.len() {
            assert_lt!((pacf_real[i] - pacf_calc[i] as f64).abs(), 1.0e-7);
        }
    }
}