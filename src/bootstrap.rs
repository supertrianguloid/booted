use crate::samplers::{Sampler, SamplerError, SamplingStrategy};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::Serialize;
use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

// -----------------------------------------------------------------------
// Errors
// -----------------------------------------------------------------------

/// Reason for a single estimator invocation failing on a bootstrap replica
/// (or on the central sample). Kept lightweight so it can be tallied by
/// reason without heap-allocation churn.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct EstimatorError {
    pub reason: Cow<'static, str>,
}

impl EstimatorError {
    pub fn new(reason: impl Into<Cow<'static, str>>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for EstimatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for EstimatorError {}

pub type EstimatorResult<T> = Result<T, EstimatorError>;

/// Errors returned by `Bootstrap::run` itself (as opposed to individual
/// replicas). A sampler that cannot produce any valid draw at all is a
/// configuration error, not a per-replica failure.
#[derive(Debug, Clone)]
pub enum BootstrapError {
    Sampler(SamplerError),
    EmptyIndices,
}

impl fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BootstrapError::Sampler(e) => write!(f, "sampler configuration error: {e}"),
            BootstrapError::EmptyIndices => f.write_str("estimator has no indices to resample"),
        }
    }
}

impl std::error::Error for BootstrapError {}

// -----------------------------------------------------------------------
// Arithmetic (needed only by bias correction, aggregated tallies)
// -----------------------------------------------------------------------

/// Arithmetic on statistics required by bias correction. Purposely separate
/// from `SummaryStatistic` so simple summary use does not need to implement
/// scaling / addition on the payload type.
pub trait Arithmetic: Sized + Clone + Send + Sync + 'static {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn scale(&self, factor: f64) -> Self;
    fn zero(len: usize) -> Self;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn add_assign(&mut self, other: &Self);
}

impl Arithmetic for f64 {
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
    fn add_assign(&mut self, other: &Self) {
        *self += *other;
    }
}

impl Arithmetic for Vec<f64> {
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
        Vec::len(self)
    }
    fn add_assign(&mut self, other: &Self) {
        for (a, b) in self.iter_mut().zip(other) {
            *a += b;
        }
    }
}

// -----------------------------------------------------------------------
// Estimator
// -----------------------------------------------------------------------

type EstimatorFn<T> = dyn Fn(&[usize]) -> EstimatorResult<T> + Send + Sync;

/// A function `f(indices) -> Result<T>` together with the "population"
/// indices to be resampled. `Estimator<T>` is a nameable, `Clone`able type
/// (the underlying closure is shared behind an `Arc`) — callers can store
/// it in fields and pass it through generic functions without dealing with
/// unnameable `impl Fn` types.
pub struct Estimator<T> {
    func: Arc<EstimatorFn<T>>,
    indices: Vec<usize>,
}

impl<T> Clone for Estimator<T> {
    fn clone(&self) -> Self {
        Self {
            func: Arc::clone(&self.func),
            indices: self.indices.clone(),
        }
    }
}

impl<T: 'static> Estimator<T> {
    pub fn new<F>(indices: Vec<usize>, func: F) -> Self
    where
        F: Fn(&[usize]) -> EstimatorResult<T> + Send + Sync + 'static,
    {
        Self {
            func: Arc::new(func),
            indices,
        }
    }

    pub fn apply(&self, indices: &[usize]) -> EstimatorResult<T> {
        (self.func)(indices)
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn with_indices(mut self, indices: Vec<usize>) -> Self {
        self.indices = indices;
        self
    }
}

impl<T: Arithmetic> Estimator<T> {
    /// Wrap this estimator so each invocation runs a small inner bootstrap
    /// under the supplied sampler and returns the bias-corrected statistic
    /// `2·θ̂ − mean(θ̂ⁱ)`.
    ///
    /// Unlike the previous API, the *inner* resampling uses `sampler`
    /// rather than forcing plain iid. This matters when the outer bootstrap
    /// uses `Block` or `MovingBlock` for autocorrelated data: bias
    /// correction must resample the same way, or the correction is biased
    /// against the very structure it is meant to preserve.
    pub fn bias_correct(
        self,
        n_inner: usize,
        sampler: SamplingStrategy,
        seed: Option<u64>,
    ) -> Estimator<T> {
        let func = self.func;
        let indices = self.indices;

        let new_func = move |sample: &[usize]| -> EstimatorResult<T> {
            if sample.is_empty() {
                return Err(EstimatorError::new("empty inner sample"));
            }
            let theta_hat = (func)(sample)?;
            let mut sum = T::zero(theta_hat.len());
            let mut valid: usize = 0;
            let mut buf = Vec::with_capacity(sample.len());
            let mut rng = match seed {
                Some(s) => SmallRng::seed_from_u64(mix_seed(s, sample.len() as u64)),
                None => SmallRng::from_rng(&mut rand::rng()),
            };
            for _ in 0..n_inner {
                if sampler
                    .sample_into_buffer(sample, &mut buf, &mut rng)
                    .is_err()
                {
                    continue;
                }
                if let Ok(v) = (func)(&buf) {
                    sum.add_assign(&v);
                    valid += 1;
                }
            }
            if valid == 0 || valid * 2 < n_inner {
                return Err(EstimatorError::new("bias correction: too few valid draws"));
            }
            let mean_boot = sum.scale(1.0 / valid as f64);
            Ok(theta_hat.scale(2.0).sub(&mean_boot))
        };

        Estimator {
            func: Arc::new(new_func),
            indices,
        }
    }
}

// -----------------------------------------------------------------------
// Progress
// -----------------------------------------------------------------------

/// Progress hook. All methods default to no-ops so implementations only
/// need to override what they care about. The bootstrap runner calls
/// `on_start` before the parallel section, `on_step` once per completed
/// replica, and `on_finish` after collection.
pub trait Progress: Send + Sync {
    fn on_start(&self, _n: usize) {}
    fn on_step(&self) {}
    fn on_finish(&self) {}
}

impl Progress for () {}

#[cfg(feature = "indicatif")]
pub use indicatif_progress::IndicatifProgress;

#[cfg(feature = "indicatif")]
mod indicatif_progress {
    use super::Progress;
    use indicatif::{ProgressBar, ProgressStyle};

    /// `indicatif`-backed progress bar. Enable the `indicatif` feature to use.
    pub struct IndicatifProgress {
        bar: ProgressBar,
    }

    impl Default for IndicatifProgress {
        fn default() -> Self {
            Self::new()
        }
    }

    impl IndicatifProgress {
        pub fn new() -> Self {
            let bar = ProgressBar::hidden();
            bar.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{eta_precise}] [{wide_bar:.cyan/blue}] [{pos}/{len}]",
                )
                .unwrap(),
            );
            Self { bar }
        }
    }

    impl Progress for IndicatifProgress {
        fn on_start(&self, n: usize) {
            self.bar.set_length(n as u64);
            self.bar.set_draw_target(indicatif::ProgressDrawTarget::stderr());
        }
        fn on_step(&self) {
            self.bar.inc(1);
        }
        fn on_finish(&self) {
            self.bar.finish();
        }
    }
}

// -----------------------------------------------------------------------
// Bootstrap
// -----------------------------------------------------------------------

/// Builder + runner for a bootstrap. Construct with `Bootstrap::new(est)`;
/// override defaults with the chainable setters; call `.run()`.
pub struct Bootstrap<T> {
    estimator: Estimator<T>,
    n_boot: usize,
    sampler: SamplingStrategy,
    seed: Option<u64>,
    progress: Option<Arc<dyn Progress>>,
}

impl<T: 'static> Bootstrap<T> {
    pub fn new(estimator: Estimator<T>) -> Self {
        Self {
            estimator,
            n_boot: 1000,
            sampler: SamplingStrategy::Iid,
            seed: None,
            progress: None,
        }
    }

    pub fn n_boot(mut self, n: usize) -> Self {
        self.n_boot = n;
        self
    }
    pub fn sampler(mut self, s: SamplingStrategy) -> Self {
        self.sampler = s;
        self
    }
    /// Seed the run. When set, the same seed produces the same replicas
    /// regardless of rayon thread count, machine, or OS.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    pub fn progress(mut self, p: Arc<dyn Progress>) -> Self {
        self.progress = Some(p);
        self
    }
}

/// Outcome of a bootstrap. Preserves the reason for failed replicas and,
/// unlike the previous API, does **not** silently fill in a zero when the
/// central estimator fails.
#[derive(Debug, Serialize)]
#[non_exhaustive]
pub struct BootstrapResult<T> {
    pub n_boot: usize,
    pub sampler: SamplingStrategy,
    pub seed: Option<u64>,
    pub truncated: usize,
    pub central: EstimatorResult<T>,
    pub samples: Vec<T>,
    pub failures: Vec<EstimatorError>,
}

impl<T> BootstrapResult<T> {
    /// Failed-replica count (i.e. `failures.len()`).
    pub fn failed(&self) -> usize {
        self.failures.len()
    }

    /// Apply a transformation to the central value and every replica.
    pub fn map<U, F>(&self, mut f: F) -> BootstrapResult<U>
    where
        F: FnMut(T) -> U,
        T: Clone,
    {
        let central = match self.central.clone() {
            Ok(v) => Ok(f(v)),
            Err(e) => Err(e),
        };
        let samples = self.samples.clone().into_iter().map(f).collect();
        BootstrapResult {
            n_boot: self.n_boot,
            sampler: self.sampler,
            seed: self.seed,
            truncated: self.truncated,
            central,
            samples,
            failures: self.failures.clone(),
        }
    }
}

// SplitMix64-like mixer for deriving per-replica seeds.
#[inline]
fn mix_seed(seed: u64, i: u64) -> u64 {
    let mut z = seed
        .wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl<T> Bootstrap<T>
where
    T: Clone + Send + Sync + 'static,
{
    pub fn run(self) -> Result<BootstrapResult<T>, BootstrapError> {
        let Bootstrap {
            estimator,
            n_boot,
            sampler,
            seed,
            progress,
        } = self;

        let indices = estimator.indices.clone();
        if indices.is_empty() {
            return Err(BootstrapError::EmptyIndices);
        }
        let truncated = sampler.truncation_for(indices.len());

        // Do the central-value application first. Its failure is *not* fatal
        // to the run — we still produce replicas — but it is preserved
        // verbatim in the result.
        let central = estimator.apply(&indices);

        if let Some(p) = progress.as_ref() {
            p.on_start(n_boot);
        }

        let func = Arc::clone(&estimator.func);
        let capacity = indices.len();

        let replicas: Vec<EstimatorResult<T>> = (0..n_boot)
            .into_par_iter()
            .map_init(
                || {
                    let rng = match seed {
                        Some(_) => None,
                        None => Some(SmallRng::from_rng(&mut rand::rng())),
                    };
                    (Vec::with_capacity(capacity), rng)
                },
                |(buf, thread_rng), i| {
                    let result = match seed {
                        Some(s) => {
                            let mut r = SmallRng::seed_from_u64(mix_seed(s, i as u64));
                            match sampler.sample_into_buffer(&indices, buf, &mut r) {
                                Ok(()) => (func)(buf),
                                Err(e) => Err(EstimatorError::new(e.to_string())),
                            }
                        }
                        None => {
                            let r = thread_rng.as_mut().unwrap();
                            match sampler.sample_into_buffer(&indices, buf, r) {
                                Ok(()) => (func)(buf),
                                Err(e) => Err(EstimatorError::new(e.to_string())),
                            }
                        }
                    };
                    if let Some(p) = progress.as_ref() {
                        p.on_step();
                    }
                    result
                },
            )
            .collect();

        if let Some(p) = progress.as_ref() {
            p.on_finish();
        }

        let mut samples = Vec::with_capacity(replicas.len());
        let mut failures = Vec::new();
        for r in replicas {
            match r {
                Ok(v) => samples.push(v),
                Err(e) => failures.push(e),
            }
        }

        Ok(BootstrapResult {
            n_boot,
            sampler,
            seed,
            truncated,
            central,
            samples,
            failures,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::SamplingStrategy;

    #[test]
    fn mean_estimator_runs() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let est = Estimator::new((0..data.len()).collect(), move |ind| {
            let s: f64 = ind.iter().map(|&i| data[i]).sum();
            Ok(s / ind.len() as f64)
        });
        let out = Bootstrap::new(est)
            .n_boot(500)
            .sampler(SamplingStrategy::Iid)
            .seed(1)
            .run()
            .unwrap();
        assert_eq!(out.samples.len(), 500);
        assert!(out.central.is_ok());
        assert_eq!(out.failures.len(), 0);
        // mean of 1..=100 is 50.5; central value should equal that exactly
        assert!((out.central.unwrap() - 50.5).abs() < 1e-9);
    }

    #[test]
    fn seed_makes_run_reproducible() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let make_est = || {
            let d = data.clone();
            Estimator::new((0..d.len()).collect(), move |ind| {
                Ok(ind.iter().map(|&i| d[i]).sum::<f64>() / ind.len() as f64)
            })
        };
        let a = Bootstrap::new(make_est())
            .seed(1234)
            .n_boot(200)
            .run()
            .unwrap();
        let b = Bootstrap::new(make_est())
            .seed(1234)
            .n_boot(200)
            .run()
            .unwrap();
        assert_eq!(a.samples, b.samples);
    }

    #[test]
    fn failures_are_preserved_and_do_not_zero_central() {
        let est: Estimator<f64> =
            Estimator::new((0..10).collect(), |_| Err(EstimatorError::new("always fails")));
        let out = Bootstrap::new(est).n_boot(20).run().unwrap();
        assert!(out.central.is_err());
        assert_eq!(out.samples.len(), 0);
        assert_eq!(out.failures.len(), 20);
    }

    #[test]
    fn empty_indices_is_error() {
        let est: Estimator<f64> = Estimator::new(vec![], |_| Ok(1.0));
        let err = Bootstrap::new(est).run().unwrap_err();
        assert!(matches!(err, BootstrapError::EmptyIndices));
    }

    #[test]
    fn bias_correction_uses_configured_sampler() {
        // Not a numerical accuracy test — just verifies the wrapped
        // estimator runs and produces the right number of replicas.
        let data: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let est = Estimator::new((0..data.len()).collect(), move |ind| {
            Ok(ind.iter().map(|&i| data[i]).sum::<f64>() / ind.len() as f64)
        });
        let corrected = est.bias_correct(50, SamplingStrategy::Block { block_size: 4 }, Some(7));
        let out = Bootstrap::new(corrected)
            .sampler(SamplingStrategy::Block { block_size: 4 })
            .n_boot(50)
            .seed(7)
            .run()
            .unwrap();
        assert_eq!(out.samples.len() + out.failures.len(), 50);
    }

    #[test]
    fn truncation_reported() {
        let est: Estimator<f64> =
            Estimator::new((0..10).collect(), |ind| Ok(ind.len() as f64));
        let out = Bootstrap::new(est)
            .sampler(SamplingStrategy::Block { block_size: 3 })
            .seed(1)
            .run()
            .unwrap();
        assert_eq!(out.truncated, 1);
    }
}
