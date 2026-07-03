#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>

#include "../io/json.hpp"
#include "../simd.hpp"

// Forward declarations for the dynamic-D inertia overload. The concrete
// definitions are provided by k_means/dynamic_d/layout.hpp before use.
template<std::size_t D>
struct samples_soa_view;

template<std::size_t D>
struct centroids_storage;

// Scalar distance is kept for detailed metrics output. Benchmark timing uses the
// SIMD total-inertia helpers below, so that C++ pays the same final inertia pass
// that sklearn.cluster.KMeans.fit pays. Use double accumulation for parity with
// scikit-learn-facing metrics.
template <eve::product_type SampleT>
double compute_scalar_dist_sq(const SampleT& sample, const SampleT& centroid) {
    double dist_sq = 0.0;

    kumi::for_each(
        [&dist_sq](auto p, auto c) {
            const double diff = static_cast<double>(p) - static_cast<double>(c);
            dist_sq += diff * diff;
        },
        sample,
        centroid
    );

    return dist_sq;
}

template <class Assignments>
inline void check_lloyd_inertia_inputs(
    std::size_t sample_count,
    std::size_t centroid_count,
    const Assignments& assignments
) {
    if (centroid_count == 0) {
        throw std::runtime_error("Invalid number of clusters");
    }

    if (assignments.size() != sample_count) {
        throw std::runtime_error("Assignment count does not match sample count");
    }
}

inline void check_lloyd_assignment_label(int label, std::size_t centroid_count) {
    if (label < 0 || static_cast<std::size_t>(label) >= centroid_count) {
        throw std::runtime_error("Invalid cluster assignment");
    }
}

template <eve::product_type SampleT, class Assignments>
double compute_static_lloyd_total_inertia_simd(
    const eve::algo::soa_vector<SampleT>& samples,
    const std::vector<SampleT>& centroids,
    const Assignments& assignments
) {
    constexpr std::size_t card = static_cast<std::size_t>(wide_f::size());

    check_lloyd_inertia_inputs(samples.size(), centroids.size(), assignments);

    alignas(64) std::array<int, card> label_lanes{};
    alignas(64) std::array<float, card> centroid_lanes{};
    alignas(64) std::array<float, card> dist_lanes{};

    double total_inertia = 0.0;

    auto aligned_assignments = eve::as_aligned(assignments.data(), cardinal{});
    auto assignments_range = eve::algo::as_range(
        aligned_assignments,
        assignments.data() + assignments.size()
    );
    auto zipped = eve::views::zip(samples, assignments_range);

    eve::algo::for_each[eve::algo::force_cardinal<card>](
        zipped,
        [&](eve::algo::iterator auto it,
            eve::relative_conditional_expr auto ignore) {
            auto [sample_it, assignment_it] = it;

            const auto sample = eve::load[ignore](sample_it);

            wide_i labels = eve::load[ignore](assignment_it);
            labels = eve::if_else(
                ignore.mask(eve::as<wide_i>{}),
                labels,
                wide_zero_i
            );
            eve::store(labels, eve::as_aligned(label_lanes.data()));

            wide_f dist = wide_zero_f;

            kumi::for_each_index(
                [&](auto index, auto sample_dimension) {
                    constexpr std::size_t d = decltype(index)::value;

                    for (std::size_t lane = 0; lane < card; ++lane) {
                        const auto k = static_cast<std::size_t>(label_lanes[lane]);
                        centroid_lanes[lane] = static_cast<float>(
                            kumi::get<d>(centroids[k])
                        );
                    }

                    const wide_f centroid_dimension = eve::load(
                        eve::as_aligned(centroid_lanes.data())
                    );

                    const wide_f diff = sample_dimension - centroid_dimension;
                    dist = eve::fma[ignore](diff, diff, dist);
                },
                sample
            );

            const wide_f valid_dist = eve::if_else(
                ignore.mask(eve::as<wide_f>{}),
                dist,
                wide_zero_f
            );
            eve::store(valid_dist, eve::as_aligned(dist_lanes.data()));

            for (std::size_t lane = 0; lane < card; ++lane) {
                total_inertia += static_cast<double>(dist_lanes[lane]);
            }
        }
    );

    return total_inertia;
}

template <std::size_t D, class Assignments>
double compute_dynamic_lloyd_total_inertia_simd(
    samples_soa_view<D> samples,
    const centroids_storage<D>& centroids,
    const Assignments& assignments
) {
    constexpr std::size_t card = static_cast<std::size_t>(wide_f::size());

    check_lloyd_inertia_inputs(samples.N, centroids.K, assignments);

    alignas(64) std::array<int, card> label_lanes{};
    alignas(64) std::array<float, card> centroid_lanes{};
    alignas(64) std::array<float, card> dist_lanes{};

    double total_inertia = 0.0;

    auto process_block = [&](std::size_t n, auto ignore) {
        wide_i labels = eve::load[ignore](
            assignments.data() + n,
            eve::as<wide_i>{}
        );
        labels = eve::if_else(
            ignore.mask(eve::as<wide_i>{}),
            labels,
            wide_zero_i
        );
        eve::store(labels, eve::as_aligned(label_lanes.data()));

        wide_f dist = wide_zero_f;

        for (std::size_t d = 0; d < D; ++d) {
            for (std::size_t lane = 0; lane < card; ++lane) {
                const auto k = static_cast<std::size_t>(label_lanes[lane]);
                centroid_lanes[lane] = centroids.row(k, d);
            }

            const wide_f sample_dimension = eve::load[ignore](
                eve::as_aligned(samples.dimension(d) + n)
            );
            const wide_f centroid_dimension = eve::load(
                eve::as_aligned(centroid_lanes.data())
            );

            const wide_f diff = sample_dimension - centroid_dimension;
            dist = eve::fma[ignore](diff, diff, dist);
        }

        const wide_f valid_dist = eve::if_else(
            ignore.mask(eve::as<wide_f>{}),
            dist,
            wide_zero_f
        );
        eve::store(valid_dist, eve::as_aligned(dist_lanes.data()));

        for (std::size_t lane = 0; lane < card; ++lane) {
            total_inertia += static_cast<double>(dist_lanes[lane]);
        }
    };

    std::size_t n = 0;

    // Fast unmasked main loop.
    for (; n + card <= samples.N; n += card) {
        process_block(n, eve::ignore_none);
    }

    // Masked tail only.
    if (n < samples.N) {
        const std::size_t valid = samples.N - n;
        const std::size_t ignored_lanes = card - valid;

        process_block(n, eve::ignore_last(ignored_lanes));
    }

    return total_inertia;
}

template <eve::product_type SampleT, class Assignments>
void write_lloyd_metrics(
    const std::string& filename,
    const eve::algo::soa_vector<SampleT>& samples,
    const std::vector<SampleT>& centroids,
    const Assignments& assignments,
    int K,
    int algorithm_iterations
) {
    if (K <= 0) {
        throw std::runtime_error("Invalid number of clusters");
    }

    if (static_cast<std::size_t>(K) != centroids.size()) {
        throw std::runtime_error("Centroid count does not match K");
    }

    if (assignments.size() != samples.size()) {
        throw std::runtime_error("Assignment count does not match sample count");
    }

    std::vector<std::size_t> cluster_counts(K, 0);
    std::vector<double> cluster_inertia(K, 0.0);

    double total_inertia = 0.0;

    for (std::size_t n = 0; n < samples.size(); ++n) {
        const std::size_t k = static_cast<std::size_t>(assignments[n]);

        if (k >= static_cast<std::size_t>(K)) {
            throw std::runtime_error("Invalid cluster assignment");
        }

        const double dist_sq = compute_scalar_dist_sq(
            samples.get(n),
            centroids[k]
        );

        cluster_counts[k] += 1;
        cluster_inertia[k] += dist_sq;
        total_inertia += dist_sq;
    }

    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error("Could not open Lloyd metrics output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);

    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"phase\": \"lloyd\",\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"algorithm_iterations\": " << algorithm_iterations << ",\n";
    out << "  \"inertia\": " << total_inertia << ",\n";

    out << "  \"cluster_counts\": [";
    for (int k = 0; k < K; ++k) {
        if (k != 0) {
            out << ", ";
        }

        out << cluster_counts[static_cast<std::size_t>(k)];
    }
    out << "],\n";

    out << "  \"cluster_inertia\": [";
    for (int k = 0; k < K; ++k) {
        if (k != 0) {
            out << ", ";
        }

        out << cluster_inertia[static_cast<std::size_t>(k)];
    }
    out << "],\n";

    out << "  \"centroids\": [\n";
    for (std::size_t k = 0; k < centroids.size(); ++k) {
        out << "    ";
        write_sample_json(out, centroids[k]);

        if (k + 1 != centroids.size()) {
            out << ",";
        }

        out << "\n";
    }
    out << "  ]\n";

    out << "}\n";
}

