// tests/integration_tests.rs

use booted::bootstrap::Bootstrap;
use booted::bootstrap::Estimator;
use booted::samplers::SamplingStrategy;
use booted::summary::{BootstrapSummary, Summarizable};
use rand_distr::{Distribution, Normal};

/// Helper to generate noisy data from a Normal distribution
fn generate_data(n: usize, mean: f64, std_dev: f64) -> Vec<f64> {
    let normal = Normal::new(mean, std_dev).unwrap();
    let mut rng = rand::rng();
    (0..n).map(|_| normal.sample(&mut rng)).collect()
}

#[test]
fn test_scalar_bootstrap_mean() {
    // 1. Setup Data
    let true_mean = 10.0;
    let true_std_dev = 2.0;
    let n_samples = 1000;
    let data = generate_data(n_samples, true_mean, true_std_dev);

    // 2. Configure Estimator
    let estimator = Estimator::new()
        .indices((0..n_samples).collect())
        .from(move |indices: &[usize]| {
            let sum: f64 = indices.iter().map(|&i| data[i]).sum();
            Some(sum / indices.len() as f64)
        })
        .build();
    // 3. Configure Bootstrap
    let bootstrap = Bootstrap::new()
        .estimator(estimator)
        .n_boot(2000)
        .sampler(SamplingStrategy::Simple)
        .build();

    // 4. Run and Summarize
    let result = bootstrap.run();
    // Explicitly typing the summary for clarity, though inference works too
    let summary: BootstrapSummary<f64> = result.summarize();

    println!("Scalar Summary: {:?}", summary);

    assert_eq!(summary.n_boot, 2000);
    assert_eq!(summary.failed_samples, 0);
    let statistics = summary.statistics.unwrap();
    assert!(
        (statistics.mean - true_mean).abs() < 0.2,
        "Mean deviated too far"
    );
    assert!(
        statistics.stddev > 0.05 && statistics.stddev < 0.08,
        "Standard Error out of expected range"
    );
    assert!(statistics.ci_95.low < true_mean);
    assert!(statistics.ci_95.high > true_mean);
}

#[test]
fn test_vector_bootstrap_multivariate() {
    // 1. Setup Data
    let col0 = vec![4.0, 5.0, 6.0, 5.0, 5.0];
    let col1 = vec![18.0, 20.0, 22.0, 20.0, 20.0];
    let n = col0.len();

    // 2. Configure Estimator
    let estimator = Estimator::new()
        .indices((0..n).collect())
        .from(move |indices: &[usize]| {
            if indices.is_empty() {
                return None;
            }
            let sum0: f64 = indices.iter().map(|&i| col0[i]).sum();
            let sum1: f64 = indices.iter().map(|&i| col1[i]).sum();
            Some(vec![
                sum0 / indices.len() as f64,
                sum1 / indices.len() as f64,
            ])
        })
        .build();

    // 3. Configure Bootstrap
    let bootstrap = Bootstrap::new().estimator(estimator).n_boot(500).build();

    // 4. Run and Summarize
    let result = bootstrap.run();
    // Using generic Summary with Vec<f64>
    let summary: BootstrapSummary<Vec<f64>> = result.summarize();

    println!("Vector Summary: {:?}", summary);
    assert_eq!(summary.n_boot, 500);
    let statistics = summary.statistics.unwrap();
    assert_eq!(statistics.len(), 2);
    assert!((statistics[0].mean - 5.0).abs() < 0.5);
    assert!((statistics[1].mean - 20.0).abs() < 1.0);
}

#[test]
fn test_bias_corrected_bootstrap() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // Outlier heavily skews mean
    let n = data.len();

    // Create an estimator and apply bias correction
    let estimator = Estimator::new()
        .indices((0..n).collect())
        .from(move |indices: &[usize]| {
            let sum: f64 = indices.iter().map(|&i| data[i]).sum();
            Some(sum / indices.len() as f64)
        })
        .build()
        .bias_correct(100);

    let bootstrap = Bootstrap::new().estimator(estimator).n_boot(200).build();

    let result = bootstrap.run();
    let summary: BootstrapSummary<f64> = result.summarize();

    assert_eq!(summary.n_boot, 200);
    assert!(summary.failed_samples == 0);
    let statistics = summary.statistics.unwrap();
    assert!(statistics.stddev > 0.0);
}

#[test]
fn test_handling_failures() {
    let n = 10;

    let estimator = Estimator::new()
        .indices((0..n).collect())
        .from(move |indices: &[usize]| {
            // Fail if the first index is even
            if indices[0] % 2 == 0 { None } else { Some(1.0) }
        })
        .build();

    let bootstrap = Bootstrap::new().estimator(estimator).n_boot(100).build();

    let result = bootstrap.run();
    let summary: BootstrapSummary<f64> = result.summarize();

    assert!(summary.failed_samples > 0);
    assert!(summary.failed_samples < 100);
    let statistics = summary.statistics.unwrap();
    assert_eq!(statistics.mean, 1.0);
}

#[test]
fn test_double_bootstrap() {
    // 1. Setup Data
    let true_mean = 10.0;
    let true_std_dev = 2.0;
    let n_samples = 2000;
    let n_boot = 100;
    let data = generate_data(n_samples, true_mean, true_std_dev);

    // 2. Configure Estimator
    let outer_estimator = Estimator::new()
        .indices((0..n_samples).collect())
        .from(move |indices: &[usize]| {
            let data = data.clone();
            // Inner estimator logic
            let inner_estimator = Estimator::new()
                .indices(indices.to_owned())
                .from(move |indices: &[usize]| {
                    let sum: f64 = indices.iter().map(|&i| data[i]).sum();
                    Some(sum / indices.len() as f64)
                })
                .build();

            // Run inner bootstrap to get stddev
            Some(
                Bootstrap::new()
                    .estimator(inner_estimator)
                    .n_boot(n_boot)
                    .sampler(SamplingStrategy::Simple)
                    .build()
                    .run()
                    .summarize() // Infers BootstrapSummary<f64>
                    .statistics
                    .unwrap()
                    .stddev,
            )
        })
        .build();

    // 3. Configure Bootstrap
    let bootstrap = Bootstrap::new()
        .estimator(outer_estimator)
        .n_boot(n_boot)
        .sampler(SamplingStrategy::Simple)
        .build();

    // 4. Run and Summarize
    let result = bootstrap.run();
    let summary: BootstrapSummary<f64> = result.summarize();

    println!("Scalar Summary: {:?}", summary);
}
