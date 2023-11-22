#[cfg(test)]
mod test_acf {
    use more_asserts::assert_lt;

    /// AR(3) model with phi=[0.8, -0.5, 0.3], mean=20.0, sd=40.0
    /// Solutions created with R script:
    /// ```R
    /// set.seed(1000)
    /// x <- round(as.vector(arima.sim(model=list(ar=c(0.8, -0.5, 0.3)), 20, mean=20.0, sd=40.0)), 10)
    /// ```
    const AR3: [f64; 20] = [
        149.8228533548,
        86.8388399871,
        42.3116899484,
        76.6796578536,
        60.3665347774,
        66.7733563129,
        -5.1144504108,
        14.0294086329,
        76.2517878809,
        121.2898170491,
        74.65663878,
        69.9331198692,
        46.7476543397,
        26.2225173663,
        -32.0638217183,
        2.8335240789,
        31.5182582874,
        76.4827451823,
        36.6122657518,
        -33.430444607,
    ];
    const AR3_COR: [f64; 20] = [
        1.0,
        0.3592447603,
        -0.0829015382,
        -0.1133314351,
        0.0044486631,
        -0.0038713998,
        -0.1672560659,
        0.0014017379,
        0.2630181693,
        0.1733800793,
        -0.1281310184,
        -0.1007529581,
        0.0503572835,
        -0.0227804107,
        -0.2575441171,
        -0.1645560738,
        -0.0715102441,
        0.0684177063,
        -0.1062164941,
        -0.2014166444,
    ];
    const AR3_COV: [f64; 20] = [
        2065.0573327104,
        741.8610264385,
        -171.196429437,
        -234.0359111105,
        9.1867444412,
        -7.9946626208,
        -345.3933653193,
        2.8946692215,
        543.1475990511,
        358.0398041259,
        -264.5978990137,
        -208.0606349333,
        103.9906776017,
        -47.042854152,
        -531.8433675447,
        -339.8177268526,
        -147.672753869,
        141.2864860104,
        -219.343149901,
        -415.9369184917,
    ];
    const AR3_PACF: [f64; 19] = [
        0.3592447603,
        -0.2433664279,
        0.0135795645,
        0.0364847159,
        -0.0567770304,
        -0.1726207939,
        0.1776766479,
        0.1898676229,
        -0.043490948,
        -0.1502576512,
        0.1089261154,
        0.0038887419,
        -0.1299288957,
        -0.1737979563,
        0.0670438445,
        -0.2479599317,
        0.1109474805,
        -0.16347413,
        -0.1005977032,
    ];

    #[test]
    fn acf_cov_full_f64() {
        let x = AR3;
        let acf_real = AR3_COV;
        let acf_calc = arima::acf::acf(&x, None, true).unwrap();

        assert_eq!(acf_real.len(), acf_calc.len());

        for i in 0..acf_real.len() {
            assert_lt!((acf_real[i] - acf_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn acf_cov_maxlag_f64() {
        const LAG: usize = 4;
        let x = AR3;
        let acf_real = &AR3_COV[0..LAG + 1];
        let acf_calc = arima::acf::acf(&x, Some(LAG), true).unwrap();

        assert_eq!(acf_real.len(), acf_calc.len());

        for i in 0..acf_real.len() {
            assert_lt!((acf_real[i] - acf_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn acf_cor_full_f64() {
        let x = AR3;
        let acf_real = AR3_COR;
        let acf_calc = arima::acf::acf(&x, None, false).unwrap();

        assert_eq!(acf_real.len(), acf_calc.len());

        for i in 0..acf_real.len() {
            assert_lt!((acf_real[i] - acf_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn acf_cor_maxlag_f64() {
        const LAG: usize = 4;
        let x = AR3;
        let acf_real = &AR3_COR[0..LAG + 1];

        let acf_calc = arima::acf::acf(&x, Some(LAG), false).unwrap();

        assert_eq!(acf_real.len(), acf_calc.len());

        for i in 0..acf_real.len() {
            assert_lt!((acf_real[i] - acf_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn ar_coef_full_f64() {
        let x = AR3;

        let ar_real = [
            0.491757153,
            -0.2238924687,
            -0.0503643329,
            -0.0536357344,
            0.1276281502,
            -0.2391384254,
            0.1094285637,
            0.1615799989,
            0.0456564625,
            -0.2268276724,
            0.1671995584,
            -0.015627203,
            -0.0510423127,
            -0.2160332708,
            0.1816141455,
            -0.332740582,
            0.1666038874,
            -0.1123501484,
            -0.1005977032,
        ];
        let (ar_calc, _var) = arima::acf::ar(&x, None).unwrap();

        assert_eq!(ar_real.len(), ar_calc.len());

        for i in 0..ar_real.len() {
            assert_lt!((ar_real[i] - ar_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn ar_coef_order_f64() {
        const ORDER: usize = 3;
        let x = AR3;

        let ar_real = [0.4499776844, -0.249432051, 0.0135795645];
        let (ar_calc, _var) = arima::acf::ar(&x, Some(ORDER)).unwrap();

        assert_eq!(ar_real.len(), ar_calc.len());

        for i in 0..ar_real.len() {
            assert_lt!((ar_real[i] - ar_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn ar_dl_coef_full_f64() {
        let x = AR3;

        let acf_calc = arima::acf::acf(&x, None, false).unwrap();
        let acv_calc = arima::acf::acf(&x, Some(0), true).unwrap();
        let cov0 = acv_calc[0];

        let ar_real = [
            0.491757153,
            -0.2238924687,
            -0.0503643329,
            -0.0536357344,
            0.1276281502,
            -0.2391384254,
            0.1094285637,
            0.1615799989,
            0.0456564625,
            -0.2268276724,
            0.1671995584,
            -0.015627203,
            -0.0510423127,
            -0.2160332708,
            0.1816141455,
            -0.332740582,
            0.1666038874,
            -0.1123501484,
            -0.1005977032,
        ];
        let (ar_calc, var) = arima::acf::ar_dl_rho_cov(&acf_calc, cov0, None).unwrap();

        assert_eq!(ar_real.len(), ar_calc.len());

        assert_lt!(var - 1691.7126551, 1.0e-7);

        for i in 0..ar_real.len() {
            assert_lt!((ar_real[i] - ar_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn ar_dl_coef_order_f64() {
        const ORDER: usize = 3;
        let x = AR3;

        let acf_calc = arima::acf::acf(&x, None, false).unwrap();
        let acv_calc = arima::acf::acf(&x, Some(0), true).unwrap();
        let cov0 = acv_calc[0];

        let ar_real = [0.4499776844, -0.249432051, 0.0135795645];
        let (ar_calc, var) = arima::acf::ar_dl_rho_cov(&acf_calc, cov0, Some(ORDER)).unwrap();

        assert_eq!(ar_real.len(), ar_calc.len());

        assert_lt!(var - 1691.7126551, 1.0e-7);

        for i in 0..ar_real.len() {
            assert_lt!((ar_real[i] - ar_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn ar_var_order_f64() {
        const ORDER: usize = 3;
        let x = AR3;

        let var_real = 1691.7126551005;
        let var_calc = arima::acf::var(&x, Some(ORDER)).unwrap();

        assert_lt!((var_real - var_calc).abs(), 1.0e-7);
    }

    #[test]
    fn pacf_full_f64() {
        let x = AR3;

        let acf_calc = arima::acf::acf(&x, None, false).unwrap();
        let cov0 = arima::acf::acf(&x, Some(0), true).unwrap()[0];

        let pacf_real = AR3_PACF;
        let pacf_calc = arima::acf::pacf_rho_cov0(&acf_calc, cov0, None).unwrap();

        assert_eq!(pacf_real.len(), pacf_calc.len());

        for i in 0..pacf_real.len() {
            assert_lt!((pacf_real[i] - pacf_calc[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn pacf_maxlag_f64() {
        const LAG: usize = 4;

        let x = AR3;

        let acf_calc = arima::acf::acf(&x, None, false).unwrap();
        let cov0 = arima::acf::acf(&x, Some(0), true).unwrap()[0];

        let pacf_real = &AR3_PACF[0..LAG];
        let pacf_calc = arima::acf::pacf_rho_cov0(&acf_calc, cov0, Some(LAG)).unwrap();

        assert_eq!(pacf_real.len(), pacf_calc.len());

        for i in 0..pacf_real.len() {
            assert_lt!((pacf_real[i] - pacf_calc[i] as f64).abs(), 1.0e-7);
        }
    }
}
