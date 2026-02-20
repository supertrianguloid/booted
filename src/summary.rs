use crate::bootstrap::{BootstrapResult, BootstrapStatistic};
use serde::Serialize;
use std::fmt::Debug;

const ONE_SIGMA: f64 = 0.682689492137086;
const TWO_SIGMA: f64 = 0.954499736103642;
const THREE_SIGMA: f64 = 0.997300203936740;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ConfidenceInterval {
    pub low: f64,
    pub high: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Statistics {
    pub mean: f64,
    pub median: f64,
    pub stddev: f64,
    pub ci_68: ConfidenceInterval,
    pub ci_95: ConfidenceInterval,
    pub ci_99: ConfidenceInterval,
}

fn calculate_stats(data: &mut [f64]) -> Option<Statistics> {
    if data.is_empty() {
        return None;
    }

    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
        mean,
        median,
        stddev,
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

pub trait SummaryStatistic: BootstrapStatistic + Debug {
    /// The type of the statistics object (e.g. `Statistics` or `Vec<Statistics>`)
    type Stats: Serialize + Debug + Clone + Send + Sync;

    /// Logic to reduce a list of replicas into the Stats type.
    fn compute_stats(samples: &[Self]) -> Self::Stats;

    /// Extract the standard error (stddev) from the stats back into the type T.
    fn standard_error(stats: &Self::Stats) -> Self;
}

impl SummaryStatistic for f64 {
    type Stats = Statistics;

    fn compute_stats(samples: &[Self]) -> Self::Stats {
        let mut data = samples.to_vec();
        calculate_stats(&mut data).expect("No samples to calculate stats")
    }

    fn standard_error(stats: &Self::Stats) -> Self {
        stats.stddev
    }
}

impl SummaryStatistic for Vec<f64> {
    type Stats = Vec<Statistics>;

    fn compute_stats(samples: &[Self]) -> Self::Stats {
        if samples.is_empty() {
            panic!("No valid bootstrap samples generated.");
        }
        // ... (transpose logic remains the same) ...
        let vec_len = samples[0].len();
        let n_samples = samples.len();
        let mut transposed = vec![Vec::with_capacity(n_samples); vec_len];
        for sample in samples {
            for (i, val) in sample.iter().enumerate() {
                transposed[i].push(*val);
            }
        }
        let mut statistics_vec = Vec::with_capacity(vec_len);
        for mut col_data in transposed.into_iter() {
            let statistics = calculate_stats(&mut col_data).unwrap();
            statistics_vec.push(statistics);
        }
        statistics_vec
    }

    fn standard_error(stats: &Self::Stats) -> Self {
        stats.iter().map(|s| s.stddev).collect()
    }
}

pub trait Summarizable<SummaryType> {
    fn summarize(self) -> SummaryType;
}

#[derive(Debug, Serialize)]
pub struct BootstrapSummary<T: SummaryStatistic> {
    pub n_boot: usize,
    pub replicas: Vec<T>,
    pub central_val: T,
    pub failed_samples: usize,
    pub statistics: T::Stats,
}

impl<T: SummaryStatistic> Summarizable<BootstrapSummary<T>> for BootstrapResult<T> {
    fn summarize(self) -> BootstrapSummary<T> {
        let statistics = T::compute_stats(&self.samples);

        // Determine central value, default to Zero if missing (and assume dimension from samples)
        let central_val = self.central_val.unwrap_or_else(|| {
            let len = self.samples.first().map(|s| s.len()).unwrap_or(1);
            T::zero(len)
        });

        BootstrapSummary {
            n_boot: self.n_boot,
            replicas: self.samples,
            central_val,
            failed_samples: self.failed_samples,
            statistics,
        }
    }
}
