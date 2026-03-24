use rand::Rng;
use rand::distr::{Distribution, Uniform};
use serde::Serialize;

pub trait Sampler {
    fn sample_into_buffer(&self, indices: &[usize], buffer: &mut Vec<usize>);
    fn sample(&self, indices: &[usize]) -> Vec<usize> {
        // Pre-allocate based on input length as a good default guess
        let mut buffer = Vec::with_capacity(indices.len());
        self.sample_into_buffer(indices, &mut buffer);
        buffer
    }
}
#[derive(Debug, Serialize, Clone)]
pub enum SamplingStrategy {
    Simple,
    MOutOfN { m: usize },
    Block { block_size: usize },
}

impl Sampler for SamplingStrategy {
    fn sample_into_buffer(&self, indices: &[usize], buffer: &mut Vec<usize>) {
        buffer.clear();

        fn m_of_n_into(indices: &[usize], m: usize, buffer: &mut Vec<usize>) {
            if m == 0 || indices.is_empty() {
                return;
            }

            buffer.reserve(m);
            let mut rng = rand::rng();
            buffer.extend(
                Uniform::try_from(0..indices.len())
                    .unwrap()
                    .sample_iter(&mut rng)
                    .take(m)
                    .map(|i| indices[i]),
            );
        }

        fn block_into(indices: &[usize], block_size: usize, buffer: &mut Vec<usize>) {
            assert!(block_size > 0);
            let data_len = indices.len();
            let effective_len = data_len - data_len % block_size;

            if effective_len == 0 {
                return;
            }

            buffer.reserve(effective_len);
            let offset = data_len - effective_len;
            let n_blocks = effective_len / block_size;
            let mut rng = rand::rng();

            for _ in 0..n_blocks {
                let block = rng.random_range(0..n_blocks);
                let start = offset + block * block_size;
                buffer.extend_from_slice(&indices[start..start + block_size]);
            }
        }

        match self {
            SamplingStrategy::Simple => m_of_n_into(indices, indices.len(), buffer),
            SamplingStrategy::MOutOfN { m } => m_of_n_into(indices, *m, buffer),
            SamplingStrategy::Block { block_size } => block_into(indices, *block_size, buffer),
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
}
