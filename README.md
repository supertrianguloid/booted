# Introduction

Booted is a Rust crate for bootstrapping estimators of arbitrary data structures. It makes no assumptions about what shape your data may be.

# Quick Start
```rust
let estimator = Estimator::new()
   .data_len(n)
   .from(move |indices: &[usize]| {
        let sum: f64 = indices.iter().map(|&i| data[i]).sum();
        Some(sum / indices.len() as f64)
    })
    .build();

let bootstrap = Bootstrap::builder()
    .estimator(estimator)
    .n_boot(1000)
    .build();

let result = bootstrap.run();
let summary: BootstrapSummary = result.summarize();
```


# Algorithms

Currently supported are simple n-out-of-n bootstrap, m-out-of-n bootstrap and block bootstrap.

# Roadmap

- Optional higher level API
- BCa, Bayesian, ABC
