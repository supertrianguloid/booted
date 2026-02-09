use crate::samplers::{Sampler, SamplingStrategy};
use bon::Builder;
use rayon::prelude::*;
use serde::Serialize;

pub trait BootstrapStatistic: Sized + Clone + Send + Sync + Serialize + 'static {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn scale(&self, factor: f64) -> Self;
    fn zero(len: usize) -> Self;
    fn len(&self) -> usize;
}

impl BootstrapStatistic for f64 {
    fn add(&self, other: &Self) -> Self {
        *self + *other
    }
    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }
    fn scale(&self, factor: f64) -> Self {
        *self * factor
    }
    fn zero(_len: usize) -> Self {
        0.0
    }
    fn len(&self) -> usize {
        1
    }
}

impl BootstrapStatistic for Vec<f64> {
    fn add(&self, other: &Self) -> Self {
        self.iter().zip(other).map(|(a, b)| a + b).collect()
    }
    fn sub(&self, other: &Self) -> Self {
        self.iter().zip(other).map(|(a, b)| a - b).collect()
    }
    fn scale(&self, factor: f64) -> Self {
        self.iter().map(|a| a * factor).collect()
    }
    fn zero(len: usize) -> Self {
        vec![0.0; len]
    }
    fn len(&self) -> usize {
        self.len()
    }
}

#[derive(Builder)]
pub struct Bootstrap<F, T>
where
    F: Fn(&[usize]) -> Option<T> + Send + Sync,
    T: BootstrapStatistic,
{
    estimator: F,
    #[builder(default = 1000)]
    n_boot: usize,
    #[builder(default = SamplingStrategy::Simple)]
    sampler: SamplingStrategy,
    data_len: usize,
    #[builder(default = 0)]
    bias_correct_nboot: usize,
}

#[derive(Debug, Serialize)]
pub struct BootstrapResult<T> {
    pub n_boot: usize,
    pub failed_samples: usize,
    pub samples: Vec<T>,
    pub central_val: Option<T>,
    pub sampler: SamplingStrategy,
}

impl<F, T> Bootstrap<F, T>
where
    F: Fn(&[usize]) -> Option<T> + Send + Sync + Clone + 'static,
    T: BootstrapStatistic,
{
    pub fn run(self) -> BootstrapResult<T> {
        let central_val = (self.estimator)(&(0..self.data_len).collect::<Vec<usize>>());

        let stat_fn: Box<dyn Fn(&[usize]) -> Option<T> + Send + Sync> =
            if self.bias_correct_nboot > 0 {
                let est = self.estimator.clone();
                let bc_n = self.bias_correct_nboot;

                Box::new(move |data: &[usize]| bootstrap_bias_correct(est.clone(), bc_n, data))
            } else {
                Box::new(self.estimator)
            };

        let samples: Vec<Option<T>> = (0..self.n_boot)
            .into_par_iter()
            .map(|_| {
                // Generate indices using the chosen strategy
                let resampled_indices = self.sampler.sample(self.data_len);
                stat_fn(&resampled_indices)
            })
            .collect();

        let (passed, failed): (Vec<_>, Vec<_>) = samples.into_iter().partition(Option::is_some);
        let valid_samples: Vec<T> = passed.into_iter().map(Option::unwrap).collect();

        BootstrapResult {
            n_boot: self.n_boot,
            failed_samples: failed.len(),
            samples: valid_samples,
            central_val,
            sampler: self.sampler,
        }
    }
}

fn bootstrap_bias_correct<F, T>(stat: F, n_boot: usize, data: &[usize]) -> Option<T>
where
    F: Fn(&[usize]) -> Option<T> + Send + Sync + Clone,
    T: BootstrapStatistic,
{
    let n = data.len();
    let theta_hat = stat(data)?;

    let mut boot_sum = T::zero(theta_hat.len());
    let mut valid_count = 0;

    for _ in 0..n_boot {
        let raw_indices = SamplingStrategy::Simple.sample(n);

        let resampled_data: Vec<usize> = raw_indices.iter().map(|&i| data[i]).collect();

        if let Some(val) = stat(&resampled_data) {
            boot_sum = boot_sum.add(&val);
            valid_count += 1;
        }
    }

    // TODO: Come up with something better for this
    if valid_count < n_boot / 2 {
        return None;
    }

    let mean_boot = boot_sum.scale(1.0 / valid_count as f64);

    Some(theta_hat.scale(2.0).sub(&mean_boot))
}
