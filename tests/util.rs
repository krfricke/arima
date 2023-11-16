#[cfg(test)]
mod test_util {
    use more_asserts::assert_lt;

    #[test]
    fn lag_i32() {
        let x = [-4, -9, 20, 23, -18, 6];
        let y = [20, 23, -18, 6];
        assert_eq!(arima::util::lag(&x, 2), y);
    }

    #[test]
    fn diff_1_i32() {
        let x = [-4, -9, 20, 23, -18, 6];
        let y = [-5, 29, 3, -41, 24];
        assert_eq!(arima::util::diff(&x, 1), y);
    }

    #[test]
    fn diff_2_i32() {
        let x = [-4, -9, 20, 23, -18, 6];
        let y = [34, -26, -44, 65];
        assert_eq!(arima::util::diff(&x, 2), y);
    }

    #[test]
    fn diff_1_f64() {
        let x = [
            4.1341055, 4.5212322, -9.1234667, -1.3249472, -8.9102578, -7.5955399, -1.8054393,
            8.6400979, 0.7207072, 6.6751565,
        ];
        let y = [
            0.3871267,
            -13.6446989,
            7.7985195,
            -7.5853106,
            1.3147179,
            5.7901006,
            10.4455372,
            -7.9193907,
            5.9544493,
        ];
        let x_diff = arima::util::diff(&x, 1);

        assert_eq!(x_diff.len(), y.len());

        for i in 0..y.len() {
            assert_lt!((x_diff[i] - y[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn diff_2_f64() {
        let x = [
            4.1341055, 4.5212322, -9.1234667, -1.3249472, -8.9102578, -7.5955399, -1.8054393,
            8.6400979, 0.7207072, 6.6751565,
        ];
        let y = [
            -14.0318256,
            21.4432184,
            -15.3838301,
            8.9000285,
            4.4753827,
            4.6554366,
            -18.3649279,
            13.87384,
        ];
        let x_diff = arima::util::diff(&x, 2);

        assert_eq!(x_diff.len(), y.len());

        for i in 0..y.len() {
            assert_lt!((x_diff[i] - y[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn diff_log_f64() {
        let x = [
            9.9902684, 4.3772393, 1.8550282, 9.7252195, 2.8445105, 0.2348111, 7.6587723, 8.9285881,
            7.6012410, 3.6073980,
        ];
        let y = [
            -0.8251932, -0.8585183, 1.6568225, -1.2293315, -2.4943650, 3.4848257, 0.1534066,
            -0.1609467, -0.7453248,
        ];
        let x_diff = arima::util::diff_log(&x);

        assert_eq!(x_diff.len(), y.len());

        for i in 0..y.len() {
            assert_lt!((x_diff[i] - y[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn cumsum_i32() {
        let x = [-4, -9, 20, 23, -18, 6];
        let y = [-4, -13, 7, 30, 12, 18];

        let x_cumsum = arima::util::cumsum(&x);

        assert_eq!(x_cumsum, y);
    }

    #[test]
    fn cumsum_f64() {
        let x = [
            4.1341055, 4.5212322, -9.1234667, -1.3249472, -8.9102578, -7.5955399, -1.8054393,
            8.6400979, 0.7207072, 6.6751565,
        ];
        let y = [
            4.1341055,
            8.6553377,
            -0.468128999999999,
            -1.7930762,
            -10.703334,
            -18.2988739,
            -20.1043132,
            -11.4642153,
            -10.7435081,
            -4.0683516,
        ];
        let x_cumsum = arima::util::cumsum(&x);

        assert_eq!(x_cumsum.len(), y.len());

        for i in 0..y.len() {
            assert_lt!((x_cumsum[i] - y[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn diffinv_1_i32() {
        let x = [-5, 29, 3, -41, 24];
        let y = [0, -5, 24, 27, -14, 10];

        let x_diffinv = arima::util::diffinv(&x, 1);

        assert_eq!(x_diffinv, y);

        // check backwards
        let z = arima::util::diff(&x_diffinv, 1);

        assert_eq!(z, x);
    }

    #[test]
    fn diffinv_2_i32() {
        let x = [-5, 29, 3, -41, 24];
        let y = [0, 0, -5, 19, 46, 32, 42];

        let x_diffinv = arima::util::diffinv(&x, 2);

        assert_eq!(x_diffinv, y);
    }

    #[test]
    fn diffinv_1_f64() {
        let x = [
            4.1341055, 4.5212322, -9.1234667, -1.3249472, -8.9102578, -7.5955399, -1.8054393,
            8.6400979, 0.7207072, 6.6751565,
        ];
        let y = [
            0.0,
            4.1341055,
            8.6553377,
            -0.468128999999999,
            -1.7930762,
            -10.703334,
            -18.2988739,
            -20.1043132,
            -11.4642153,
            -10.7435081,
            -4.0683516,
        ];
        let x_diffinv = arima::util::diffinv(&x, 1);

        assert_eq!(x_diffinv.len(), y.len());

        for i in 0..y.len() {
            assert_lt!((x_diffinv[i] - y[i] as f64).abs(), 1.0e-7);
        }

        // check backwards
        let z = arima::util::diff(&x_diffinv, 1);

        for i in 0..z.len() {
            assert_lt!((z[i] - x[i] as f64).abs(), 1.0e-7);
        }
    }

    #[test]
    fn diffinv_2_f64() {
        let x = [
            4.1341055, 4.5212322, -9.1234667, -1.3249472, -8.9102578, -7.5955399, -1.8054393,
            8.6400979, 0.7207072, 6.6751565,
        ];
        let y = [
            0.0,
            0.0,
            4.1341055,
            12.7894432,
            12.3213142,
            10.528238,
            -0.175095999,
            -18.4739699,
            -38.5782831,
            -50.0424984,
            -60.7860065,
            -64.8543581,
        ];
        let x_diffinv = arima::util::diffinv(&x, 2);

        assert_eq!(x_diffinv.len(), y.len());

        for i in 0..y.len() {
            assert_lt!((x_diffinv[i] - y[i] as f64).abs(), 1.0e-7);
        }
    }
}
