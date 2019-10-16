#[cfg(test)]
mod test_util {
    use more_asserts::assert_lt;

    #[test]
    fn lag_i32() {
        let x = vec![-4, -9, 20, 23, -18, 6];
        let y = vec![20, 23, -18, 6];
        assert_eq!(arima::util::lag(&x, 2), y);
    }

    #[test]
    fn diff_i32() {
        let x = vec![-4, -9, 20, 23, -18, 6];
        let y = vec![-5, 29, 3, -41, 24];
        assert_eq!(arima::util::diff(&x), y);
    }

    #[test]
    fn diff_f64() {
        let x = vec![4.1341055, 4.5212322, -9.1234667, -1.3249472, -8.9102578, -7.5955399, -1.8054393,  8.6400979,  0.7207072,  6.6751565];
        let y = vec![0.3871267, -13.6446989, 7.7985195, -7.5853106, 1.3147179,  5.7901006, 10.4455372, -7.9193907,  5.9544493];
        let x_diff = arima::util::diff(&x);

        for i in 0..y.len() {
            assert_lt!((x_diff[i] - y[i] as f64).abs(), 1.0e7);
        }
    }

    #[test]
    fn diff_log_f64() {
        let x = vec![9.9902684, 4.3772393, 1.8550282, 9.7252195, 2.8445105, 0.2348111, 7.6587723, 8.9285881, 7.6012410, 3.6073980];
        let y = vec![-0.8251932, -0.8585183, 1.6568225, -1.2293315, -2.4943650, 3.4848257, 0.1534066, -0.1609467, -0.7453248];
        let x_diff = arima::util::diff_log(&x);

        for i in 0..y.len() {
            assert_lt!((x_diff[i] - y[i] as f64).abs(), 1.0e7);
        }
    }
}