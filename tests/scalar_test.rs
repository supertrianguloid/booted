use booted::{
    Bootstrap, BootstrapSummary, Estimator, EstimatorError, SamplingStrategy, Summarisable,
};
use rand_distr::{Distribution, Normal};
use serde_json::Value;

/// Downstream tooling (analysis pipelines, reports) reads `central_val` as
/// a bare scalar and `failed_samples` as a count. Assert that both keys
/// still exist in the emitted JSON, even after the 0.6 refactor that
/// changed the underlying in-memory representation to `Result` /
/// `Vec<EstimatorError>`.
#[test]
fn legacy_json_shape_preserved() {
    let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let est = Estimator::new((0..data.len()).collect(), move |ind| {
        Ok(ind.iter().map(|&i| data[i]).sum::<f64>() / ind.len() as f64)
    });
    let summary: BootstrapSummary<f64> = Bootstrap::new(est)
        .n_boot(50)
        .seed(1)
        .run()
        .unwrap()
        .summarise();
    let v: Value = serde_json::from_str(&serde_json::to_string(&summary).unwrap()).unwrap();
    assert!(v.get("central_val").unwrap().is_number());
    assert!(v.get("failed_samples").unwrap().is_number());
    assert!(v.get("replicas").unwrap().is_array());
    assert!(v.get("statistics").unwrap().is_object());
    assert!(v.get("n_boot").unwrap().is_number());
    assert!(v.get("sampler").is_some());
    // New diagnostic fields are also present
    assert!(v.get("failure_reasons").unwrap().is_array());
    assert!(v.get("truncated").is_some());
}

fn generate_data(n: usize, mean: f64, std_dev: f64) -> Vec<f64> {
    let normal = Normal::new(mean, std_dev).unwrap();
    let mut rng = rand::rng();
    (0..n).map(|_| normal.sample(&mut rng)).collect()
}

#[test]
fn scalar_bootstrap_mean() {
    let true_mean = 10.0;
    let true_std_dev = 2.0;
    let n_samples = 1000;
    let data = generate_data(n_samples, true_mean, true_std_dev);

    let estimator = Estimator::new((0..n_samples).collect(), move |indices: &[usize]| {
        let sum: f64 = indices.iter().map(|&i| data[i]).sum();
        Ok(sum / indices.len() as f64)
    });

    let summary: BootstrapSummary<f64> = Bootstrap::new(estimator)
        .n_boot(2000)
        .sampler(SamplingStrategy::Iid)
        .run()
        .unwrap()
        .summarise();

    assert_eq!(summary.n_boot, 2000);
    assert_eq!(summary.failures.len(), 0);
    let statistics = summary.statistics.unwrap();
    assert!((statistics.mean - true_mean).abs() < 0.2);
    assert!(statistics.stddev > 0.05 && statistics.stddev < 0.08);
    assert!(statistics.ci_95.low < true_mean);
    assert!(statistics.ci_95.high > true_mean);
}

#[test]
fn vector_bootstrap_multivariate() {
    let col0 = vec![4.0, 5.0, 6.0, 5.0, 5.0];
    let col1 = vec![18.0, 20.0, 22.0, 20.0, 20.0];
    let n = col0.len();

    let estimator = Estimator::new((0..n).collect(), move |indices: &[usize]| {
        if indices.is_empty() {
            return Err(EstimatorError::new("empty"));
        }
        let sum0: f64 = indices.iter().map(|&i| col0[i]).sum();
        let sum1: f64 = indices.iter().map(|&i| col1[i]).sum();
        Ok(vec![sum0 / indices.len() as f64, sum1 / indices.len() as f64])
    });

    let summary: BootstrapSummary<Vec<f64>> = Bootstrap::new(estimator)
        .n_boot(500)
        .run()
        .unwrap()
        .summarise();

    assert_eq!(summary.n_boot, 500);
    let statistics = summary.statistics.unwrap();
    assert_eq!(statistics.len(), 2);
    assert!((statistics[0].mean - 5.0).abs() < 0.5);
    assert!((statistics[1].mean - 20.0).abs() < 1.0);
}

#[test]
fn bias_corrected_bootstrap() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    let n = data.len();

    let estimator = Estimator::new((0..n).collect(), move |indices: &[usize]| {
        let sum: f64 = indices.iter().map(|&i| data[i]).sum();
        Ok(sum / indices.len() as f64)
    })
    .bias_correct(100, SamplingStrategy::Iid, Some(1));

    let summary: BootstrapSummary<f64> = Bootstrap::new(estimator)
        .n_boot(200)
        .run()
        .unwrap()
        .summarise();

    assert_eq!(summary.n_boot, 200);
    assert_eq!(summary.failures.len(), 0);
    assert!(summary.statistics.unwrap().stddev > 0.0);
}

#[test]
fn handling_failures() {
    let n = 10;

    let estimator = Estimator::new((0..n).collect(), move |indices: &[usize]| {
        if indices[0] % 2 == 0 {
            Err(EstimatorError::new("first index even"))
        } else {
            Ok(1.0)
        }
    });

    let summary: BootstrapSummary<f64> = Bootstrap::new(estimator)
        .n_boot(100)
        .run()
        .unwrap()
        .summarise();

    assert!(!summary.failures.is_empty());
    assert!(summary.failures.len() < 100);
    let statistics = summary.statistics.unwrap();
    assert_eq!(statistics.mean, 1.0);
}

#[test]
fn double_bootstrap() {
    let true_mean = 10.0;
    let true_std_dev = 2.0;
    let n_samples = 2000;
    let n_boot = 100;
    let data = generate_data(n_samples, true_mean, true_std_dev);

    // The outer estimator's closure builds and runs an inner bootstrap.
    // Because `Estimator<T>` is now a nameable, `Clone`able type, this
    // pattern no longer requires `impl Fn` in an unnameable position.
    let outer = Estimator::new((0..n_samples).collect(), move |indices: &[usize]| {
        let data = data.clone();
        let inner = Estimator::new(indices.to_owned(), move |idx: &[usize]| {
            Ok(idx.iter().map(|&i| data[i]).sum::<f64>() / idx.len() as f64)
        });
        let inner_summary: BootstrapSummary<f64> = Bootstrap::new(inner)
            .n_boot(n_boot)
            .sampler(SamplingStrategy::Iid)
            .run()
            .map_err(|e| EstimatorError::new(e.to_string()))?
            .summarise();
        Ok(inner_summary
            .statistics
            .ok_or_else(|| EstimatorError::new("no stats"))?
            .stddev)
    });

    let summary: BootstrapSummary<f64> = Bootstrap::new(outer)
        .n_boot(n_boot)
        .sampler(SamplingStrategy::Iid)
        .run()
        .unwrap()
        .summarise();

    assert_eq!(summary.n_boot, n_boot);
}
