use crate::bootstrap::{BootstrapResult, EstimatorError, EstimatorResult};
use crate::samplers::SamplingStrategy;
use serde::Serialize;
use std::fmt::Debug;

const ONE_SIGMA: f64 = 0.682_689_492_137_086;
const TWO_SIGMA: f64 = 0.954_499_736_103_642;
const THREE_SIGMA: f64 = 0.997_300_203_936_740;

#[derive(Debug, Clone, Copy, Serialize)]
#[non_exhaustive]
pub struct ConfidenceInterval {
    pub low: f64,
    pub high: f64,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct Statistics {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    pub stddev: f64,
    pub iqr: f64,
    pub max: f64,
    pub min: f64,
    pub ci_68: ConfidenceInterval,
    pub ci_95: ConfidenceInterval,
    pub ci_99: ConfidenceInterval,
}

/// Compute summary stats on a slice of samples. Uses `f64::total_cmp` for
/// sorting so NaN inputs land in a well-defined place rather than silently
/// corrupting quantiles.
pub fn calculate_stats(data: &mut [f64]) -> Option<Statistics> {
    if data.is_empty() {
        return None;
    }

    data.sort_unstable_by(f64::total_cmp);

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let stddev = variance.sqrt();
    let mid = data.len() / 2;
    let median = if data.len().is_multiple_of(2) {
        (data[mid - 1] + data[mid]) / 2.0
    } else {
        data[mid]
    };

    let quantile = |q: f64| -> f64 {
        let idx = (q * (data.len() - 1) as f64).round() as usize;
        data[idx]
    };

    Some(Statistics {
        n: data.len(),
        mean,
        median,
        stddev,
        min: *data.first().unwrap(),
        max: *data.last().unwrap(),
        iqr: quantile(0.75) - quantile(0.25),
        ci_68: ConfidenceInterval {
            low: quantile((1.0 - ONE_SIGMA) / 2.0),
            high: quantile((1.0 + ONE_SIGMA) / 2.0),
        },
        ci_95: ConfidenceInterval {
            low: quantile((1.0 - TWO_SIGMA) / 2.0),
            high: quantile((1.0 + TWO_SIGMA) / 2.0),
        },
        ci_99: ConfidenceInterval {
            low: quantile((1.0 - THREE_SIGMA) / 2.0),
            high: quantile((1.0 + THREE_SIGMA) / 2.0),
        },
    })
}

/// Types that can be summarised by aggregating replicas. Deliberately does
/// **not** require arithmetic ops on `Self` — bias correction is the only
/// operation that needs those and lives on `Estimator`.
pub trait SummaryStatistic: Sized + Clone + Send + Sync + Serialize + Debug + 'static {
    /// Per-component stats. `f64` -> `Statistics`; `Vec<f64>` -> `Vec<Statistics>`.
    type Stats: Serialize + Debug + Clone + Send + Sync;

    /// Reduce replicas to summary stats.
    fn compute_stats(samples: &[Self]) -> Option<Self::Stats>;

    /// Standard-error projection back into `Self` (used for double-bootstrap
    /// composition: `Bootstrap<Bootstrap<T>>::standard_error → T`).
    fn standard_error(stats: &Self::Stats) -> Self;
}

impl SummaryStatistic for f64 {
    type Stats = Statistics;

    fn compute_stats(samples: &[Self]) -> Option<Self::Stats> {
        let mut data = samples.to_vec();
        calculate_stats(&mut data)
    }

    fn standard_error(stats: &Self::Stats) -> Self {
        stats.stddev
    }
}

impl SummaryStatistic for Vec<f64> {
    type Stats = Vec<Statistics>;

    fn compute_stats(samples: &[Self]) -> Option<Self::Stats> {
        if samples.is_empty() {
            return None;
        }
        let vec_len = samples[0].len();
        let n_samples = samples.len();
        let mut transposed: Vec<Vec<f64>> = (0..vec_len)
            .map(|_| Vec::with_capacity(n_samples))
            .collect();
        for sample in samples {
            for (i, val) in sample.iter().enumerate() {
                transposed[i].push(*val);
            }
        }
        let mut statistics_vec = Vec::with_capacity(vec_len);
        for mut col_data in transposed.into_iter() {
            statistics_vec.push(calculate_stats(&mut col_data)?);
        }
        Some(statistics_vec)
    }

    fn standard_error(stats: &Self::Stats) -> Self {
        stats.iter().map(|s| s.stddev).collect()
    }
}

pub trait Summarisable<S> {
    fn summarise(self) -> S;
}

#[derive(Debug, Serialize)]
#[non_exhaustive]
pub struct BootstrapSummary<T: SummaryStatistic> {
    pub n_boot: usize,
    pub sampler: SamplingStrategy,
    pub seed: Option<u64>,
    pub truncated: usize,
    /// Central estimator result. If the central sample failed, the error is
    /// preserved verbatim rather than being replaced by a zero value.
    pub central: EstimatorResult<T>,
    pub replicas: Vec<T>,
    pub failures: Vec<EstimatorError>,
    pub statistics: Option<T::Stats>,
}

impl<T: SummaryStatistic> Summarisable<BootstrapSummary<T>> for BootstrapResult<T> {
    fn summarise(self) -> BootstrapSummary<T> {
        let statistics = T::compute_stats(&self.samples);
        BootstrapSummary {
            n_boot: self.n_boot,
            sampler: self.sampler,
            seed: self.seed,
            truncated: self.truncated,
            central: self.central,
            replicas: self.samples,
            failures: self.failures,
            statistics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_on_integers() {
        let mut data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let s = calculate_stats(&mut data).unwrap();
        assert!((s.mean - 50.5).abs() < 1e-9);
        assert_eq!(s.n, 100);
    }

    #[test]
    fn stats_handle_nan_deterministically() {
        // With total_cmp, NaN sorts at the end, so quantiles on finite
        // regions are still well-defined and the run does not silently
        // scramble ordering.
        let mut data = vec![1.0, 2.0, f64::NAN, 3.0, 4.0, 5.0];
        let s = calculate_stats(&mut data).unwrap();
        // min is the smallest finite; max ends up NaN under total_cmp.
        assert_eq!(s.min, 1.0);
        assert!(s.max.is_nan());
    }
}
