// tests/integration_tests.rs
// Written by Gemini. Needs some human love.

use booted::bootstrap::Bootstrap;
use booted::samplers::SamplingStrategy;
use booted::summary::{BootstrapSummary, BootstrapSummaryVec, Summarizable};
use rand_distr::{Distribution, Normal}; // Requires rand_distr in [dev-dependencies]

/// Helper to generate noisy data from a Normal distribution
fn generate_data(n: usize, mean: f64, std_dev: f64) -> Vec<f64> {
    let normal = Normal::new(mean, std_dev).unwrap();
    let mut rng = rand::rng();
    (0..n).map(|_| normal.sample(&mut rng)).collect()
}

#[test]
fn test_scalar_bootstrap_mean() {
    // 1. Setup Data using the helper
    // Generate 1000 samples from a Normal distribution (Mean = 10.0, SD = 2.0)
    let true_mean = 10.0;
    let true_std_dev = 2.0;
    let n_samples = 1000;

    let data = generate_data(n_samples, true_mean, true_std_dev);
    dbg!(&data);

    // 2. Configure Bootstrap
    // We expect the Standard Error of the mean to be approx sigma / sqrt(n)
    // 2.0 / sqrt(1000) ~= 0.063
    let bootstrap = Bootstrap::builder()
        .n_boot(2000)
        .data_len(n_samples)
        .sampler(SamplingStrategy::Simple)
        .estimator(move |indices| {
            if indices.is_empty() {
                return None;
            }
            let sum: f64 = indices.iter().map(|&i| data[i]).sum();
            Some(sum / indices.len() as f64)
        })
        .build();

    // 3. Run and Summarize
    let result = bootstrap.run();
    let summary: BootstrapSummary = result.summarize();

    println!("Scalar Summary: {:?}", summary);

    // 4. Assertions
    assert_eq!(summary.n_boot, 2000);
    assert_eq!(summary.failed_samples, 0);

    // Check 1: The Bootstrap Mean should be very close to the True Mean (within 0.2)
    assert!(
        (summary.statistics.mean - true_mean).abs() < 0.2,
        "Mean deviated too far"
    );

    // Check 2: The Standard Error (stddev) should be close to theoretical (2.0 / sqrt(1000) ≈ 0.063)
    // We allow a little wiggle room due to randomness.
    assert!(
        summary.statistics.stddev > 0.05 && summary.statistics.stddev < 0.08,
        "Standard Error out of expected range"
    );

    // Check 3: The 95% Confidence Interval should contain the True Mean (10.0)
    assert!(
        summary.statistics.ci_95.low < true_mean,
        "Lower CI bound is too high"
    );
    assert!(
        summary.statistics.ci_95.high > true_mean,
        "Upper CI bound is too low"
    );
}

#[test]
fn test_vector_bootstrap_multivariate() {
    // 1. Setup Data: Two correlated variables
    let col0 = vec![4.0, 5.0, 6.0, 5.0, 5.0];
    let col1 = vec![18.0, 20.0, 22.0, 20.0, 20.0];
    let n = col0.len();

    // 2. Configure Bootstrap
    let bootstrap = Bootstrap::builder()
        .n_boot(500)
        .data_len(n)
        .estimator(move |indices| {
            if indices.is_empty() {
                return None;
            }

            let sum0: f64 = indices.iter().map(|&i| col0[i]).sum();
            let mean0 = sum0 / indices.len() as f64;

            let sum1: f64 = indices.iter().map(|&i| col1[i]).sum();
            let mean1 = sum1 / indices.len() as f64;

            Some(vec![mean0, mean1])
        })
        .build();

    // 3. Run and Summarize
    let result = bootstrap.run();

    // Explicitly ask for Vector Summary
    let summary: BootstrapSummaryVec = result.summarize();

    println!("Vector Summary: {:?}", summary);

    assert_eq!(summary.n_boot, 500);
    assert_eq!(summary.statistics.len(), 2);

    // Check Column 0 (Mean ~ 5.0)
    assert!((summary.statistics[0].mean - 5.0).abs() < 0.5);

    // Check Column 1 (Mean ~ 20.0)
    assert!((summary.statistics[1].mean - 20.0).abs() < 1.0);
}

#[test]
fn test_bias_corrected_bootstrap() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // Outlier heavily skews mean
    let n = data.len();

    let bootstrap = Bootstrap::builder()
        .n_boot(200)
        .data_len(n)
        .bias_correct_nboot(100) // Enable bias correction
        .estimator(move |indices| {
            let sum: f64 = indices.iter().map(|&i| data[i]).sum();
            Some(sum / indices.len() as f64)
        })
        .build();

    let result = bootstrap.run();
    let summary: BootstrapSummary = result.summarize();

    assert_eq!(summary.n_boot, 200);
    assert!(summary.failed_samples == 0);
    assert!(summary.statistics.stddev > 0.0);
}

#[test]
fn test_handling_failures() {
    let n = 10;

    // Estimator that fails 50% of the time based on random indices
    let bootstrap = Bootstrap::builder()
        .n_boot(100)
        .data_len(n)
        .estimator(move |indices| {
            // Fail if the first index is even (arbitrary failure condition)
            if indices[0] % 2 == 0 { None } else { Some(1.0) }
        })
        .build();

    let result = bootstrap.run();
    let summary: BootstrapSummary = result.summarize();

    assert!(summary.failed_samples > 0);
    assert!(summary.failed_samples < 100);
    assert_eq!(summary.statistics.mean, 1.0);
}
