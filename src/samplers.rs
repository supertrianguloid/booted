use rand::Rng;
use rand::distr::{Distribution, Uniform};
use serde::Serialize;
use std::fmt;

/// Errors returned when a sampling strategy cannot draw a resample from the
/// given index set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplerError {
    /// Population is empty and the strategy requires at least one item.
    Empty,
    /// The population is smaller than the requested block size.
    BlockTooLarge { block_size: usize, n: usize },
    /// The population is not a multiple of the block size and truncation is
    /// forbidden by the caller.
    Truncation {
        block_size: usize,
        n: usize,
        dropped: usize,
    },
    /// The thinning factor is zero, or `n / factor == 0`.
    BadThinning { factor: usize, n: usize },
    /// A `Subsample { m }` was requested with `m == 0`.
    ZeroSample,
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplerError::Empty => write!(f, "sampler received an empty index set"),
            SamplerError::BlockTooLarge { block_size, n } => write!(
                f,
                "block size {block_size} exceeds population size {n}"
            ),
            SamplerError::Truncation {
                block_size,
                n,
                dropped,
            } => write!(
                f,
                "population size {n} is not a multiple of block size {block_size}; would drop {dropped} items"
            ),
            SamplerError::BadThinning { factor, n } => write!(
                f,
                "thinning factor {factor} is invalid for population size {n}"
            ),
            SamplerError::ZeroSample => write!(f, "requested sample size 0"),
        }
    }
}

impl std::error::Error for SamplerError {}

/// Ways to draw a resample from a population of configuration indices.
///
/// The variants split cleanly into *iid* schemes (`Iid`, `Subsample`,
/// `Thinning`) and *block* schemes (`Block`, `MovingBlock`). Block schemes
/// preserve local autocorrelation; iid schemes do not.
#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SamplingStrategy {
    /// Ordinary bootstrap: draw `n` items with replacement from a population
    /// of size `n`.
    Iid,
    /// m-out-of-n subsampling: draw `m` items with replacement.
    Subsample { m: usize },
    /// Thinning: keep an iid subsample of size `n / factor`. Equivalent to
    /// `Subsample { m: n / factor }` but resolved at draw time (does not need
    /// to know `n` at construction time).
    Thinning { factor: usize },
    /// Non-overlapping block bootstrap: partition the sequence into blocks of
    /// `block_size` and draw complete blocks with replacement.
    Block { block_size: usize },
    /// Moving (overlapping) block bootstrap of Künsch (1989).
    MovingBlock { block_size: usize },
}

pub trait Sampler {
    /// Draw a resample into `buffer`. `buffer` is cleared first.
    fn sample_into_buffer<R: Rng + ?Sized>(
        &self,
        indices: &[usize],
        buffer: &mut Vec<usize>,
        rng: &mut R,
    ) -> Result<(), SamplerError>;

    /// Convenience wrapper allocating a fresh `Vec<usize>`.
    fn sample<R: Rng + ?Sized>(
        &self,
        indices: &[usize],
        rng: &mut R,
    ) -> Result<Vec<usize>, SamplerError> {
        let mut buffer = Vec::with_capacity(indices.len());
        self.sample_into_buffer(indices, &mut buffer, rng)?;
        Ok(buffer)
    }
}

fn iid_draw<R: Rng + ?Sized>(indices: &[usize], m: usize, buffer: &mut Vec<usize>, rng: &mut R) {
    if m == 0 || indices.is_empty() {
        return;
    }
    buffer.reserve(m);
    let dist = Uniform::try_from(0..indices.len()).unwrap();
    buffer.extend(dist.sample_iter(rng).take(m).map(|i| indices[i]));
}

fn block_draw<R: Rng + ?Sized>(
    indices: &[usize],
    block_size: usize,
    buffer: &mut Vec<usize>,
    rng: &mut R,
) -> Result<(), SamplerError> {
    let n = indices.len();
    if n < block_size {
        return Err(SamplerError::BlockTooLarge { block_size, n });
    }
    let n_blocks = n / block_size;
    let effective_len = n_blocks * block_size;
    let offset = n - effective_len;
    buffer.reserve(effective_len);
    for _ in 0..n_blocks {
        let block = rng.random_range(0..n_blocks);
        let start = offset + block * block_size;
        buffer.extend_from_slice(&indices[start..start + block_size]);
    }
    Ok(())
}

fn moving_block_draw<R: Rng + ?Sized>(
    indices: &[usize],
    block_size: usize,
    buffer: &mut Vec<usize>,
    rng: &mut R,
) -> Result<(), SamplerError> {
    let n = indices.len();
    if n < block_size {
        return Err(SamplerError::BlockTooLarge { block_size, n });
    }
    let n_blocks = n / block_size;
    let total_len = n_blocks * block_size;
    let n_starts = n - block_size + 1;
    buffer.reserve(total_len);
    for _ in 0..n_blocks {
        let start = rng.random_range(0..n_starts);
        buffer.extend_from_slice(&indices[start..start + block_size]);
    }
    Ok(())
}

impl Sampler for SamplingStrategy {
    fn sample_into_buffer<R: Rng + ?Sized>(
        &self,
        indices: &[usize],
        buffer: &mut Vec<usize>,
        rng: &mut R,
    ) -> Result<(), SamplerError> {
        buffer.clear();
        if indices.is_empty() {
            return Err(SamplerError::Empty);
        }
        match self {
            SamplingStrategy::Iid => {
                iid_draw(indices, indices.len(), buffer, rng);
                Ok(())
            }
            SamplingStrategy::Subsample { m } => {
                if *m == 0 {
                    return Err(SamplerError::ZeroSample);
                }
                iid_draw(indices, *m, buffer, rng);
                Ok(())
            }
            SamplingStrategy::Thinning { factor } => {
                if *factor == 0 {
                    return Err(SamplerError::BadThinning {
                        factor: *factor,
                        n: indices.len(),
                    });
                }
                let m = indices.len() / factor;
                if m == 0 {
                    return Err(SamplerError::BadThinning {
                        factor: *factor,
                        n: indices.len(),
                    });
                }
                iid_draw(indices, m, buffer, rng);
                Ok(())
            }
            SamplingStrategy::Block { block_size } => {
                if *block_size == 0 {
                    return Err(SamplerError::ZeroSample);
                }
                block_draw(indices, *block_size, buffer, rng)
            }
            SamplingStrategy::MovingBlock { block_size } => {
                if *block_size == 0 {
                    return Err(SamplerError::ZeroSample);
                }
                moving_block_draw(indices, *block_size, buffer, rng)
            }
        }
    }
}

impl SamplingStrategy {
    /// If this strategy will truncate the population (block schemes on data
    /// whose size is not a multiple of `block_size`), return how many items
    /// are dropped. Returns 0 otherwise.
    pub fn truncation_for(&self, n: usize) -> usize {
        match self {
            SamplingStrategy::Block { block_size }
            | SamplingStrategy::MovingBlock { block_size } => {
                if *block_size == 0 || n < *block_size {
                    0
                } else {
                    n % block_size
                }
            }
            _ => 0,
        }
    }
}

/// Deterministic block-jackknife index sets: block index `k` is left out,
/// all others are kept, in order.
pub fn generate_block_jackknife_indices(blocksize: usize, data_length: usize) -> Vec<Vec<usize>> {
    assert!(blocksize > 0);
    let remainder = data_length % blocksize;
    let sample_start_index = remainder;
    let num_blocks = (data_length - remainder) / blocksize;

    (0..num_blocks)
        .map(|k| {
            let start_block = sample_start_index + k * blocksize;
            let end_block = start_block + blocksize;
            (sample_start_index..start_block)
                .chain(end_block..data_length)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    #[test]
    fn iid_full_returns_n() {
        let ind: Vec<usize> = (0..10).collect();
        let s = SamplingStrategy::Iid.sample(&ind, &mut rng()).unwrap();
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn subsample_m() {
        let ind: Vec<usize> = (0..10).collect();
        let s = SamplingStrategy::Subsample { m: 3 }.sample(&ind, &mut rng()).unwrap();
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn thinning() {
        let ind: Vec<usize> = (0..10).collect();
        let s = SamplingStrategy::Thinning { factor: 2 }
            .sample(&ind, &mut rng())
            .unwrap();
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn thinning_zero_is_error() {
        let ind: Vec<usize> = (0..10).collect();
        let err = SamplingStrategy::Thinning { factor: 0 }
            .sample(&ind, &mut rng())
            .unwrap_err();
        assert!(matches!(err, SamplerError::BadThinning { .. }));
    }

    #[test]
    fn block_returns_multiple_of_block_size() {
        let ind: Vec<usize> = (0..10).collect();
        let s = SamplingStrategy::Block { block_size: 3 }
            .sample(&ind, &mut rng())
            .unwrap();
        // 10/3 = 3 blocks, so 9 items
        assert_eq!(s.len(), 9);
    }

    #[test]
    fn block_too_large_is_error() {
        let ind: Vec<usize> = (0..3).collect();
        let err = SamplingStrategy::Block { block_size: 4 }
            .sample(&ind, &mut rng())
            .unwrap_err();
        assert!(matches!(err, SamplerError::BlockTooLarge { .. }));
    }

    #[test]
    fn moving_block_uses_overlapping_windows() {
        let ind: Vec<usize> = (0..10).collect();
        let s = SamplingStrategy::MovingBlock { block_size: 3 }
            .sample(&ind, &mut rng())
            .unwrap();
        assert_eq!(s.len(), 9);
        for chunk in s.chunks(3) {
            let start = chunk[0];
            assert!(start + 3 <= ind.len());
            assert_eq!(chunk, &ind[start..start + 3]);
        }
    }

    #[test]
    fn empty_input_is_error() {
        let ind: Vec<usize> = vec![];
        let err = SamplingStrategy::Iid.sample(&ind, &mut rng()).unwrap_err();
        assert_eq!(err, SamplerError::Empty);
    }

    #[test]
    fn truncation_reporting() {
        assert_eq!(
            SamplingStrategy::Block { block_size: 3 }.truncation_for(10),
            1
        );
        assert_eq!(SamplingStrategy::Iid.truncation_for(10), 0);
    }

    #[test]
    fn seeded_runs_are_reproducible() {
        let ind: Vec<usize> = (0..100).collect();
        let a = SamplingStrategy::Iid
            .sample(&ind, &mut SmallRng::seed_from_u64(7))
            .unwrap();
        let b = SamplingStrategy::Iid
            .sample(&ind, &mut SmallRng::seed_from_u64(7))
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn block_jackknife_shape() {
        let sets = generate_block_jackknife_indices(4, 10);
        // 10 = 2*4 + 2, so 2 blocks, 2 left-out sets, each length 4
        assert_eq!(sets.len(), 2);
        for s in &sets {
            assert_eq!(s.len(), 4);
        }
    }
}
