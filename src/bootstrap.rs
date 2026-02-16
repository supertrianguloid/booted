use crate::samplers::{Sampler, SamplingStrategy};
use bon::Builder;
use rand::seq::IndexedRandom;
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

/// An Estimator contains the function to calculate the statistic (which contains the data), and the length of the data.
///
/// It is immutable and safe to share across threads.
#[derive(Builder)]
#[builder(start_fn = new)]
pub struct Estimator<F> {
    #[builder(name = from)]
    func: F, // The function which eats indices (a subset of the population indices) and produces the statistic
    indices: Vec<usize>, // The indices for the entire population
}

impl<F> Estimator<F> {
    /// Applies the estimator function to a set of indices.
    pub fn apply<T>(&self, indices: &[usize]) -> Option<T>
    where
        F: Fn(&[usize]) -> Option<T> + Sync,
    {
        (self.func)(indices)
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Consumes the current Estimator and returns a new one that applies bias correction.
    ///
    /// This works by wrapping the original estimator function in a new closure that performs
    /// an inner bootstrap loop.
    pub fn bias_correct<T>(
        self,
        n_boot: usize,
    ) -> Estimator<impl Fn(&[usize]) -> Option<T> + Send + Sync + Clone>
    where
        F: Fn(&[usize]) -> Option<T> + Send + Sync + Clone + 'static,
        T: BootstrapStatistic,
    {
        /// Helper function to perform the bias correction logic.
        fn bootstrap_bias_correct<F, T>(stat: &F, n_boot: usize, data: &[usize]) -> Option<T>
        where
            F: Fn(&[usize]) -> Option<T> + Send + Sync,
            T: BootstrapStatistic,
        {
            let n = data.len();
            let theta_hat = stat(data)?;

            let mut boot_sum = T::zero(theta_hat.len());
            let mut valid_count = 0;
            for _ in 0..n_boot {
                let resampled_data: Vec<usize> = (0..n)
                    .map(|_| *data.choose(&mut rand::rng()).unwrap())
                    .collect();

                if let Some(val) = stat(&resampled_data) {
                    boot_sum = boot_sum.add(&val);
                    valid_count += 1;
                }
            }

            if valid_count < n_boot / 2 {
                return None;
            }

            let mean_boot = boot_sum.scale(1.0 / valid_count as f64);
            Some(theta_hat.scale(2.0).sub(&mean_boot))
        }
        let func = self.func;
        let indices = self.indices;

        let new_func = move |indices: &[usize]| bootstrap_bias_correct(&func, n_boot, indices);

        Estimator {
            func: new_func,
            indices,
        }
    }
}

#[derive(Builder)]
pub struct Bootstrap<F> {
    estimator: Estimator<F>,
    #[builder(default = 1000)]
    n_boot: usize,
    #[builder(default = SamplingStrategy::Simple)]
    sampler: SamplingStrategy,
}

#[derive(Debug, Serialize)]
pub struct BootstrapResult<T> {
    pub n_boot: usize,
    pub failed_samples: usize,
    pub samples: Vec<T>,
    pub central_val: Option<T>,
    pub sampler: SamplingStrategy,
}

impl<F> Bootstrap<F> {
    pub fn run<T>(self) -> BootstrapResult<T>
    where
        F: Fn(&[usize]) -> Option<T> + Send + Sync,
        T: BootstrapStatistic,
    {
        let indices = self.estimator.indices();
        let central_val = self.estimator.apply(indices);

        // We access the function directly. Since `Bootstrap` owns the `Estimator`,
        // and we are inside `run(self)`, we own the function.
        // We pass a reference to the function to `map`, requiring F to be Sync.
        let func = &self.estimator.func;

        let samples: Vec<Option<T>> = (0..self.n_boot)
            .into_par_iter()
            .map(|_| {
                let resampled_indices = self.sampler.sample(indices);
                func(&resampled_indices)
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
