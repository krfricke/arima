#[cfg(test)]
mod test_estimate {
    use more_asserts::assert_lt;

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

    const AR3_RES: [f64; 20] = [
        0.0,
        0.0,
        0.0,
        46.2603808,
        -7.7972931,
        28.510325,
        -57.7569706,
        14.2417414,
        31.2183008,
        48.5090956,
        -2.716499,
        38.8984537,
        -5.402662,
        -8.4669355,
        -62.7063041,
        4.5063279,
        -14.4924325,
        31.271378,
        -29.2554603,
        -54.8047308,
    ];

    #[test]
    fn residuals_ar3_f64() {
        let x = AR3;
        let (y, _mean) = arima::util::center(&x);
        let intercept = -5.954353;
        let phi = [0.67715294, -0.44171525, 0.08249936];

        let residuals_real = AR3_RES;
        let residuals = arima::estimate::residuals(&y, intercept, Some(&phi), None).unwrap();

        assert_eq!(residuals.len(), residuals_real.len());

        for i in 0..residuals_real.len() {
            // the residuals were collected from R's arima() routine. allow for some variance.
            assert_lt!((residuals_real[i] - residuals[i] as f64).abs(), 1.0e-3);
        }
    }

    #[test]
    fn residuals_arima_102_f64() {
        let x = AR3;
        let (y, _mean) = arima::util::center(&x);

        let intercept = -23.64706;
        let phi = [0.48359302];
        let theta = [1.05643909, 1.51029256];

        let residuals_real = [
            0.0,
            12.5024401,
            -14.7741471,
            51.0605505,
            -10.2274033,
            -30.6143332,
            8.4998564,
            51.8766267,
            -0.0576161,
            4.2438554,
            9.7222585,
            15.2661396,
            -19.7658293,
            -0.442378,
            -16.3084314,
            34.3532355,
            16.6032739,
            -10.0661619,
            -16.6988839,
            -20.1747913,
        ];
        let residuals =
            arima::estimate::residuals(&y, intercept, Some(&phi), Some(&theta)).unwrap();

        assert_eq!(residuals.len(), residuals_real.len());

        for i in 0..residuals_real.len() {
            // the residuals were collected from R's arima() routine. allow for some variance.
            assert_lt!((residuals_real[i] - residuals[i] as f64).abs(), 1.0e-3);
        }
    }

    #[test]
    fn fit_arima_2002_f64() {
        let x = AR3;

        let coef = arima::estimate::fit(&x, 2, 0, 0).unwrap();

        // Results obtained from R with
        // `cf <- arima(x, order=c(2, 0, 0), method="CSS", optim.method="L-BFGS-B")$coef`
        // R's intercept must be transformed with `cf["intercept"] * (1 - sum(cf[1:2]))`
        assert_lt!((coef[0] - 29.3546).abs(), 1.0e-4); // Intercept
        assert_lt!((coef[1] - 0.6465575).abs(), 1.0e-4); // AR 1
        assert_lt!((coef[2] - -0.3452993).abs(), 1.0e-4); // AR 2
    }

    #[test]
    fn fit_arima_101_f64() {
        let x = AR3;

        let coef = arima::estimate::fit(&x, 1, 0, 1).unwrap();

        // Results obtained from R with
        // `cf <- arima(x, order=c(1, 0, 1), method="CSS", optim.method="L-BFGS-B")$coef`
        // R's intercept must be transformed with `cf["intercept"] * (1 - sum(cf[1]))`
        assert_lt!((coef[0] - 24.18111).abs(), 1.0e-3); // Intercept
        assert_lt!((coef[1] - 0.3596548).abs(), 1.0e-4); // AR 1
        assert_lt!((coef[2] - 0.2880067).abs(), 1.0e-4); // MA 1
    }

    #[test]
    fn fit_arima_102_f64() {
        let x = AR3;

        let coef = arima::estimate::fit(&x, 1, 0, 2).unwrap();
        println!("{:?}", coef);

        // Results obtained from R with
        // `cf <- arima(x, order=c(1, 0, 2), method="CSS", optim.method="L-BFGS-B")$coef`
        // R's intercept must be transformed with `cf["intercept"] * (1 - sum(cf[1]))`

        // With MA > 1, we often find multiple similarly good models that mostly differ in
        // the intercept. Thus, we ignore it in this test.
        //assert_lt!((coef[0] - 1.884872).abs(), 1.0e-7);  // Intercept

        assert_lt!((coef[1] - 0.4835826).abs(), 1.0e-2); // AR 1
        assert_lt!((coef[2] - 1.0564438).abs(), 1.0e-2); // MA 1
        assert_lt!((coef[3] - 1.5102864).abs(), 1.0e-2); // MA 2
    }
}
