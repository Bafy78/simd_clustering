# 🚧 Limitations and future work

## 🔭 Scope and non-goals

This project is a controlled benchmark and validation codebase for SIMD-oriented C++ clustering kernels. It is not intended to be a production clustering library, a drop-in replacement for scikit-learn, or a complete implementation of every clustering API exposed by scikit-learn or hdbscan-contrib.

The current scope is deliberately narrow:

* compare selected C++ SIMD implementations against Python/scikit-learn or hdbscan-contrib references;
* use shared input artifacts and, where applicable, shared initialization artifacts;
* measure the configured benchmark phases and HDBSCAN stages under the repository's benchmark methodology;
* validate parity for the phases/stages that emit comparable metrics;
* report results for dense finite numeric inputs under a fixed, documented timing regime.

The main in-scope algorithmic work is:

| Area              | Current scope                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| K-Means++         | Greedy K-Means++ initialization benchmark, with current stochastic measured                       |
| Lloyd / K-Means   | Lloyd-style K-Means with shared initial centers, Euclidean squared-distance assignment, and controlled iteration settings.                                                                                          |
| GMM EM            | Gaussian mixture EM for `spherical`, `diag`, and `full` covariance types.                         |
| HDBSCAN           | Dense Euclidean HDBSCAN stage benchmarking for distance, MST, linkage, selection, and full pipeline stages.                                                                                    |
| Layout conversion | AoS-to-SoA conversion costs used to interpret static-D and dynamic-D C++ layouts.                 |
| References        | scikit-learn references for K-Means/K-Means++/GMM/HDBSCAN-brute, plus hdbscan-contrib for HDBSCAN.|
| Inputs            | Dense, 2-D, numeric, finite arrays materialized into binary artifacts.                            |
| Datasets          | Synthetic `make_blobs` datasets. Also a few real datasets comparisons but secondary.              |
| Portability       | We do compare a few compilers and architecture, although we didn't investigate in details.        |
| Profiling         | Cachegrind and the Spill detector are here to help interepret results, in addition to their intended use during development                                                                                         |

The following are explicitly out of scope for the current paper/results:

| Area                         | Non-goal / out-of-scope item                                                           |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| Inputs                       | Sparse input matrices.                                                                 |
| Inputs                       | Missing values, `NaN`, and infinity handling beyond rejecting non-finite dense inputs during dataset materialization.                                                                                         |
| Inputs                       | Categorical, mixed-type, ragged, object-dtype, text, image, graph, or sequence inputs. |
| Inputs                       | General preprocessing pipelines such as imputation, normalization, standardization, feature extraction, or dimensionality reduction.                                                                        |
| K-Means                      | Non-Euclidean K-Means metrics.                                                         |
| K-Means                      | Elkan K-Means, MiniBatch K-Means, Bisecting K-Means, constrained K-Means, weighted K-Means, and other K-Means variants.                                                                                    |
| K-Means                      | Arbitrary initialization policies beyond the controlled shared initialization used for Lloyd/GMM and the current K-Means++ benchmark path.                                                                     |
| K-Means++                    | Output parity for K-Means++ center selection; the current K-Means++ phase is primarily a timing benchmark.                                                                                                       |
| GMM                          | The scikit-learn `tied` covariance model. Only `spherical`, `diag`, and `full` are currently implemented.                                                                                                  |
| GMM                          | Full `GaussianMixture` API coverage, including sampling, scoring APIs beyond benchmark metrics, warm starts as a user-facing feature, and general estimator lifecycle behavior.                                |
| GMM                          | Arbitrary GMM initialization strategies beyond the controlled shared weights/means/precision initialization artifacts.                                                                                     |
| HDBSCAN                      | Arbitrary HDBSCAN metrics. The current staged comparison is dense Euclidean.           |
| HDBSCAN                      | Sparse, approximate-nearest-neighbor, tree-based, graph-sparse, or accelerated HDBSCAN variants.                                                                                                               |
| HDBSCAN                      | Full scikit-learn or hdbscan-contrib HDBSCAN API coverage.                             |
| HDBSCAN                      | Prediction or approximate prediction for new samples.                                  |
| Other clustering algorithms  | Affinity Propagation, Mean Shift, Spectral Clustering, Agglomerative/Ward clustering, DBSCAN, OPTICS, BIRCH, Density Peaks, and other algorithms surveyed in the exploratory notes but not implemented in the benchmark pipeline.                                                                                                     |
| Sample weights               | Sample-weighted clustering.                                                            |
| Streaming and online learning | Streaming, online, incremental, mini-batch, or out-of-core clustering.                |
| Runtime model                | GPU acceleration.                                                                      |
| Runtime model                | Distributed-memory or multi-node execution.                                            |
| Runtime model                | Multi-threaded C++ scaling                                                             |
| API design                   | Production API design, packaging, ABI stability, serialization, model persistence, sklearn estimator protocol compliance, and user-facing error-message completeness.                                      |
| API design                   | Drop-in scikit-learn compatibility. scikit-learn is a reference for selected benchmark operations, not a complete behavioral contract.                                                                         |
| New-sample prediction        | Prediction, transformation, scoring, or probability estimation on new samples outside the measured fit/stage outputs.                                                                                         |
| Validation meaning           | Proof that an implementation found the mathematically best clustering. Parity only checks that selected outputs are sufficiently close to the configured reference under the current thresholds.           |
| Validation coverage          | Validation against every scikit-learn version, hdbscan-contrib version, compiler, platform, or CPU. Some reference behavior is version-sensitive and should be treated as part of the benchmark contract. |
| Bitwise equivalence          | Bit-for-bit equivalence with reference implementations. The project uses explicit numerical thresholds and, where necessary, documents known numerical differences.                                       |

These non-goals are part of the interpretation contract for the results. If future work adds any of these features, it should be introduced as a new benchmark regime or a separately documented extension, not silently folded into the current comparisons.


## Things we could have easily measured and compared

* A bigger amount of real datasets
* Decoupled `min_samples` and `min_cluster_size` in HDBSCAN