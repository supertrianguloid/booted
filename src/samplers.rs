use rand::Rng;
use rand::distr::{Distribution, Uniform};
use serde::Serialize;

pub trait Sampler {
    fn sample(&self, indices: &[usize]) -> Vec<usize>;
}
#[derive(Debug, Serialize, Clone)]
pub enum SamplingStrategy {
    Simple,
    MOutOfN { m: usize },
    Block { block_size: usize },
    Thinned { block_size: usize },
}

impl Sampler for SamplingStrategy {
    fn sample(&self, indices: &[usize]) -> Vec<usize> {
        // #[inline(always)]
        fn m_of_n_indices(indices: &[usize], m: usize) -> Vec<usize> {
            if m == 0 || indices.is_empty() {
                return Vec::new();
            }

            let mut rng = rand::rng();
            Uniform::try_from(0..indices.len())
                .unwrap()
                .sample_iter(&mut rng)
                .take(m)
                .map(|i| indices[i])
                .collect()
        }
        pub fn block_indices(indices: &[usize], block_size: usize) -> Vec<usize> {
            assert!(block_size > 0);
            let data_len = indices.len();

            let effective_len = data_len - data_len % block_size;
            if effective_len == 0 {
                return Vec::new();
            }

            let offset = data_len - effective_len;
            let n_blocks = effective_len / block_size;

            let mut rng = rand::rng();
            let mut indices_new = Vec::with_capacity(effective_len);

            for _ in 0..n_blocks {
                let block = rng.random_range(0..n_blocks);
                let start = offset + block * block_size;
                for j in 0..block_size {
                    indices_new.push(indices[start + j]);
                }
            }
            indices_new
        }

        match self {
            SamplingStrategy::Simple => m_of_n_indices(indices, indices.len()),
            SamplingStrategy::MOutOfN { m } => m_of_n_indices(indices, *m),
            SamplingStrategy::Block { block_size } => block_indices(indices, *block_size),
            SamplingStrategy::Thinned { block_size } => {
                let m = indices.len() / block_size;
                SamplingStrategy::MOutOfN { m }.sample(indices)
            }
        }
    }
}

pub fn generate_block_jackknife_indices(blocksize: usize, data_length: usize) -> Vec<Vec<usize>> {
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

// #[inline(always)]
// pub fn subsample_indices(sample: &[usize]) -> Vec<usize> {
//     let mut result = Vec::with_capacity(sample.len());
//     let mut rng = rand::rng();
//     for _ in 0..sample.len() {
//         let index = rng.random_range(..sample.len());
//         result.push(sample[index]);
//     }
//     result
// }
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_of_n_sample_test() {
        let sample = (SamplingStrategy::MOutOfN { m: 2 }).sample(&(0..10).collect::<Vec<usize>>());
        assert_eq!(sample.len(), 2);
        let sample = (SamplingStrategy::MOutOfN { m: 3 }).sample(&(0..10).collect::<Vec<usize>>());
        assert_eq!(sample.len(), 3);
    }
    #[test]
    fn block_jackknife_test() {
        dbg!(generate_block_jackknife_indices(4, 10));
    }
    #[test]
    fn block_indices_test() {
        let indices = (0..10).collect::<Vec<usize>>();
        let indices2 = vec![1, 1, 1, 2, 2, 2, 2];
        assert_eq!(
            &SamplingStrategy::Block { block_size: 10 }.sample(&indices),
            &indices
        );
        dbg!(SamplingStrategy::Block { block_size: 9 }.sample(&indices));
        dbg!(SamplingStrategy::Block { block_size: 2 }.sample(&indices2));
    }
    #[test]
    fn thinned_sample_test() {
        let indices: Vec<usize> = (0..10).collect();
        let sample_thinned_2 = SamplingStrategy::Thinned { block_size: 2 }.sample(&indices);
        assert_eq!(sample_thinned_2.len(), 5);
        dbg!(sample_thinned_2);
        let sample_thinned_3 = SamplingStrategy::Thinned { block_size: 3 }.sample(&indices);
        assert_eq!(sample_thinned_3.len(), 3);
        dbg!(sample_thinned_3);
    }
}
