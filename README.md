# Booted

**Booted** is a fast, flexible Rust crate for bootstrapping estimators on arbitrary data structures. 

It makes no assumptions about the shape or type of your data. Instead, it operates on data *indices*, allowing you to bootstrap scalars, vectors, or complex nested structures seamlessly. Powered by `rayon`, Booted automatically parallelizes the resampling and estimation process for high performance.

## Features

- **Agnostic to data shape**: Bootstrap scalars, arrays, or custom structs.
- **Multiple Sampling Strategies**: Supports standard Simple (n-out-of-n), m-out-of-n, and Block bootstrapping.
- **Bias Correction**: Easily wrap estimators to perform double-bootstrap bias correction.
- **Parallel by Default**: Heavily leverages `rayon` to compute samples across all available CPU cores.
- **Ergonomic API**: Built using the `bon` builder pattern for clean, readable configuration.

## Quick Start

Here is a complete example of calculating the bootstrap mean, standard error, and 95% confidence interval for a simple 1D dataset:

```rust
use booted::bootstrap::{Bootstrap, Estimator};
use booted::summary::{BootstrapSummary, Summarizable};

fn main() {
    // 1. Your arbitrary data
    let data = vec![1.2, 2.3, 1.9, 2.5, 2.1, 3.0, 2.2, 1.8];
    let n = data.len();

    // 2. Configure the Estimator
    // The estimator closure receives a subset of indices and returns your calculated statistic.
    let estimator = Estimator::new()
        .indices((0..n).collect())
        .from(move |indices: &[usize]| {
            let sum: f64 = indices.iter().map(|&i| data[i]).sum();
            Some(sum / indices.len() as f64)
        })
        .build();

    // 3. Configure and run the Bootstrap
    let result = Bootstrap::builder()
        .estimator(estimator)
        .n_boot(5000) // Generate 5000 resamples
        .build()
        .run();

    // 4. Summarize the results
    let summary: BootstrapSummary<f64> = result.summarize();

    println!("Bootstrap Mean: {:.4}", summary.statistics.mean);
    println!("Standard Error (StdDev): {:.4}", summary.statistics.stddev);
    println!(
        "95% Confidence Interval: [{:.4}, {:.4}]",
        summary.statistics.ci_95.low, summary.statistics.ci_95.high
    );
}
````
