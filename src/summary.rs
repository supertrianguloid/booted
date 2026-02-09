use crate::bootstrap::BootstrapResult;
use serde::Serialize;

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
    let median = if data.len() % 2 == 0 {
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

pub trait Summarizable<SummaryType> {
    fn summarize(self) -> SummaryType;
}
#[derive(Debug, Serialize)]
pub struct BootstrapSummary {
    pub n_boot: usize,
    pub replicas: Vec<f64>,
    pub central_val: f64,
    pub failed_samples: usize,
    pub statistics: Statistics,
}

impl Summarizable<BootstrapSummary> for BootstrapResult<f64> {
    fn summarize(self) -> BootstrapSummary {
        let mut replicas = self.samples;
        let statistics = calculate_stats(&mut replicas).unwrap();

        let central_val = self.central_val.unwrap_or(0.0);

        BootstrapSummary {
            n_boot: self.n_boot,
            central_val,
            failed_samples: self.failed_samples,
            replicas,
            statistics,
        }
    }
}
#[derive(Debug, Serialize)]
pub struct BootstrapSummaryVec {
    pub n_boot: usize,
    pub replicas: Vec<Vec<f64>>,
    pub central_val: Vec<f64>,
    pub failed_samples: usize,
    pub statistics: Vec<Statistics>,
}

impl Summarizable<BootstrapSummaryVec> for BootstrapResult<Vec<f64>> {
    fn summarize(self) -> BootstrapSummaryVec {
        if self.samples.is_empty() {
            panic!("No valid bootstrap samples generated.");
        }

        let vec_len = self.samples[0].len();
        let n_samples = self.samples.len();

        // Transpose data: [sample][stat] -> [stat][sample]
        let mut transposed = vec![Vec::with_capacity(n_samples); vec_len];
        for sample in &self.samples {
            for (i, val) in sample.iter().enumerate() {
                transposed[i].push(*val);
            }
        }

        let central_vals = self.central_val.unwrap_or_else(|| vec![0.0; vec_len]);

        let mut statistics_vec = Vec::with_capacity(vec_len);

        for mut col_data in transposed.into_iter() {
            let statistics = calculate_stats(&mut col_data).unwrap();
            statistics_vec.push(statistics);
        }

        BootstrapSummaryVec {
            n_boot: self.n_boot,
            replicas: self.samples,
            central_val: central_vals,
            failed_samples: self.failed_samples,
            statistics: statistics_vec,
        }
    }
}
