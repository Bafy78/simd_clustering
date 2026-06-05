#pragma once

#include <vector>
#include <span>
#include <type_traits>
#include <array>

#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "../core.hpp"
#include "../../layout/static_soa.hpp"

using wide_f = kmeans::wide_f;
using wide_i = kmeans::wide_i;

template <typename SimdSample, typename SampleType>
auto compute_simd_assignment_score(
    const SimdSample& sample,
    const SampleType& centroid,
    float centroid_norm_sq
) {
    auto dot = wide_zero_f;

    kumi::for_each([&](auto p, auto c) {
        dot = eve::fma(p, wide_f(c), dot);
    }, sample, centroid);

    return eve::fma(dot, wide_f(-2.0f), wide_f(centroid_norm_sq));
}

template <typename SimdSample, typename SampleType>
wide_i compute_closest_centroid_labels(
    const SimdSample& sample,
    const std::vector<SampleType>& centroids,
    std::span<const float> centroid_norms
) {
    auto min_distances = eve::valmax(eve::as<wide_f>());
    auto closest_centroid_labels = wide_zero_i;

    for (std::size_t k = 0; k < centroids.size(); ++k) {
        auto assignment_score = compute_simd_assignment_score(
            sample,
            centroids[k],
            centroid_norms[k]
        );

        auto is_closer = assignment_score < min_distances;

        min_distances = eve::min(min_distances, assignment_score);

        closest_centroid_labels = eve::if_else(
            is_closer,
            wide_i(static_cast<int>(k)),
            closest_centroid_labels
        );
    }

    return closest_centroid_labels;
}

template <bool TrackChanges = false, eve::algo::relaxed_range R, typename SampleType>
bool assign_samples_to_centroids(
    R&& zipped_data,
    const std::vector<SampleType>& centroids,
    std::span<const float> centroid_norms
) {
    bool any_changed = false;

    eve::algo::for_each(
        zipped_data,
        [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {

        auto [sample_it, assign_it] = it;
        auto sample = eve::load[ignore](sample_it);

        auto closest_centroid_labels = compute_closest_centroid_labels(sample, centroids, centroid_norms);

        if constexpr (TrackChanges) {
            if (!any_changed) {
                const wide_i previous_centroid_labels = eve::load[ignore](assign_it);

                any_changed = eve::any[ignore](
                    closest_centroid_labels != previous_centroid_labels
                    );
            }
        }

        eve::store[ignore](closest_centroid_labels, assign_it);
    }
    );

    return any_changed;
}


template <eve::product_type SampleType>
void add_sample_to_sum(
    const eve::algo::soa_vector<SampleType>& samples,
    std::vector<SampleType>& sum,
    std::size_t k,
    std::size_t n
) {
    const auto sample = samples.get(n);
    kumi::for_each([](auto& s, auto x) { s += x; }, sum[k], sample);
}

template <eve::product_type SampleType>
void subtract_sample_from_sum(
    const eve::algo::soa_vector<SampleType>& samples,
    std::vector<SampleType>& sum,
    std::size_t k,
    std::size_t n
) {
    const auto sample = samples.get(n);
    kumi::for_each([](auto& s, auto x) { s -= x; }, sum[k], sample);
}

template <eve::product_type SampleType>
float write_centroid_from_sum(
    std::vector<SampleType>& centroids,
    const std::vector<SampleType>& sum,
    std::size_t k,
    int count
) {
    const float inv_count = 1.0f / static_cast<float>(count);
    const SampleType new_centroid = kumi::map(
        [inv_count](auto s) { return s * inv_count; },
        sum[k]
    );

    const float shift_sq = kumi::inner_product(
        centroids[k],
        new_centroid,
        0.0f,
        [](auto acc, auto x) { return acc + x; },
        [](auto old_c, auto new_c) {
            const auto diff = new_c - old_c;
            return diff * diff;
        }
    );

    centroids[k] = new_centroid;

    return shift_sq;
}

template <eve::product_type SampleType>
bool resolve_dead_centroids(
    const eve::algo::soa_vector<SampleType>& samples,
    std::span<const int> assignments,
    const std::vector<SampleType>& old_centroids,
    std::vector<SampleType>& sum,
    std::span<int> counts
) {
    struct ops_t {
        const eve::algo::soa_vector<SampleType>& samples;
        const std::vector<SampleType>& old_centroids;
        std::vector<SampleType>& sum;

        std::size_t N() const { return samples.size(); }

        float distance_to_old_centroid(std::size_t n, std::size_t old_k) const {
            return kumi::inner_product(
                samples.get(n),
                old_centroids[old_k],
                0.0f,
                [](auto acc, auto x) { return acc + x; },
                [](auto p, auto c) {
                    auto diff = p - c;
                    return diff * diff;
                }
            );
        }

        void relocate_empty_cluster(
            std::size_t old_k,
            std::size_t new_k,
            std::size_t n
        ) {
            subtract_sample_from_sum(samples, sum, old_k, n);
            sum[new_k] = samples.get(n);
        }
    };

    ops_t ops{ samples, old_centroids, sum };

    return kmeans::resolve_dead_centroids_common(
        ops,
        assignments,
        counts
    );
}

template <eve::product_type SampleType>
kmeans::centroid_update_result update_centroids(
    const eve::algo::soa_vector<SampleType>& samples,
    std::span<const int> assignments,
    std::vector<SampleType>& centroids,
    std::vector<SampleType>& sum,
    std::vector<int>& counts
) {
    struct ops_t {
        const eve::algo::soa_vector<SampleType>& samples;
        std::vector<SampleType>& centroids;
        std::vector<SampleType>& sum;

        std::size_t N() const { return samples.size(); }
        std::size_t K() const { return centroids.size(); }

        void reset_sums() {
            for (auto& row : sum) row = SampleType{};
        }

        void add_sample_to_sum(std::size_t k, std::size_t n) {
            ::add_sample_to_sum(samples, sum, k, n);
        }

        bool resolve_dead_centroids(std::span<const int> assignments, std::span<int> counts) {
            return ::resolve_dead_centroids(samples, assignments, centroids, sum, counts);
        }

        float write_centroid_from_sum(std::size_t k, int count) {
            return ::write_centroid_from_sum(centroids, sum, k, count);
        }
    };

    ops_t ops{ samples, centroids, sum };

    return kmeans::update_centroids_common(
        ops,
        assignments,
        std::span<int>(counts.data(), counts.size())
    );
}

template <eve::product_type SampleType>
struct kumi_kmeans_backend {
    using assignment_vector = std::vector<int, eve::aligned_allocator<int>>;
    using counts_vector = std::vector<int>;

    const eve::algo::soa_vector<SampleType>& original_samples;

    eve::algo::soa_vector<SampleType> samples;
    std::vector<SampleType>& centroids;

    std::vector<SampleType> sum;
    std::vector<float> centroid_norms;
    kmeans::incremental_centroid_update_state<assignment_vector> incremental_update;
    SampleType dimension_mean{};

    kumi_kmeans_backend(
        const eve::algo::soa_vector<SampleType>& samples_,
        std::vector<SampleType>& centroids_
    )
        : original_samples(samples_),
        samples(),
        centroids(centroids_),
        sum(centroids_.size()),
        centroid_norms(centroids_.size()),
        incremental_update(centroids_.size()) {}

    void recompute_centroid_norm(std::size_t k) {
        centroid_norms[k] = kumi::inner_product(centroids[k], centroids[k], 0.0f);
    }

    void recompute_centroid_norms() {
        centroid_norms.resize(centroids.size());

        for (std::size_t k = 0; k < centroids.size(); ++k) {
            recompute_centroid_norm(k);
        }
    }

    void recompute_dirty_centroid_norms() {
        for (int k : incremental_update.dirty_clusters_span()) {
            recompute_centroid_norm(static_cast<std::size_t>(k));
        }
    }

    void compute_dimension_mean_from_original() {
        if (original_samples.size() == 0) {
            dimension_mean = SampleType{};
            return;
        }

        SampleType dimension_sum{};

        for (std::size_t n = 0; n < original_samples.size(); ++n) {
            const auto sample = original_samples.get(n);
            kumi::for_each([](auto& s, auto x) { s += x; }, dimension_sum, sample);
        }

        const float inv_N = 1.0f / static_cast<float>(original_samples.size());

        dimension_mean = kumi::map([inv_N](auto s) { return s * inv_N; }, dimension_sum);
    }

    float copy_centered_samples_from_original_and_compute_scaled_tolerance(float tol) {
        if (original_samples.size() == 0) {
            samples.clear();
            return 0.0f;
        }

        samples = eve::algo::soa_vector<SampleType>(
            eve::algo::no_init,
            original_samples.size(),
            samples.get_allocator()
        );

        wide_f total_squared_centered_norm_w = wide_zero_f;

        eve::algo::for_each(
            eve::views::zip(original_samples, samples),
            [&](eve::algo::iterator auto it,
                eve::relative_conditional_expr auto ignore) {
                auto [src_it, dst_it] = it;
                auto src_sample = eve::load[ignore](src_it);
                using simd_sample_t = std::remove_cvref_t<decltype(src_sample)>;

                auto centered = kumi::map(
                    [](auto x, auto mean) { return x - mean; },
                    src_sample,
                    dimension_mean
                );

                kumi::for_each(
                    [&](auto x) {
                        total_squared_centered_norm_w =
                            eve::fma[ignore](x, x, total_squared_centered_norm_w);
                    },
                    centered
                );

                eve::store[ignore](simd_sample_t{centered}, dst_it);
            }
        );

        if (tol == 0.0f) return 0.0f;

        const float total_squared_centered_norm = eve::reduce(total_squared_centered_norm_w);

        return tol
            * total_squared_centered_norm
            / static_cast<float>(samples.size())
            / static_cast<float>(kumi::size_v<SampleType>);
    }

    void subtract_dimension_mean_from_centroids() {
        for (auto& centroid : centroids) {
            kumi::for_each(
                [](auto& c, auto mean) { c -= mean; },
                centroid,
                dimension_mean
            );
        }
    }

    void add_dimension_mean_to_centroids() {
        for (auto& centroid : centroids) {
            kumi::for_each(
                [](auto& c, auto mean) { c += mean; },
                centroid,
                dimension_mean
            );
        }
    }

    float prepare_data_for_fit(float tol) {
        compute_dimension_mean_from_original();

        const float scaled_tol = copy_centered_samples_from_original_and_compute_scaled_tolerance(tol);

        subtract_dimension_mean_from_centroids();
        recompute_centroid_norms();

        return scaled_tol;
    }

    void finish_fit_after_final_assignment() {
        add_dimension_mean_to_centroids();
    }

    void check_cluster_count() const {
        kmeans::check_cluster_count(centroids.size(), original_samples.size());
    }

    std::size_t N() const {
        return samples.size();
    }

    assignment_vector make_assignment_vector(int initial_value) const {
        return assignment_vector(original_samples.size(), initial_value);
    }

    counts_vector make_counts_vector() const {
        return counts_vector(centroids.size(), 0);
    }


    void assign(assignment_vector& assignments) const {
        auto aligned_ptr = eve::as_aligned(assignments.data(), kmeans::cardinal{});
        auto unaligned_end = assignments.data() + assignments.size();
        auto assignments_range = eve::algo::as_range(aligned_ptr, unaligned_end);
        auto zipped_data = eve::views::zip(samples, assignments_range);
        (void)::assign_samples_to_centroids<false>(
            zipped_data,
            centroids,
            std::span<const float>(centroid_norms.data(), centroid_norms.size())
        );
    }
    bool assign_and_check_changed(assignment_vector& assignments) {
        incremental_update.snapshot_before_assignment(
            std::span<const int>{assignments.data(), assignments.size()}
        );

        auto aligned_ptr = eve::as_aligned(assignments.data(), kmeans::cardinal{});
        auto unaligned_end = assignments.data() + assignments.size();
        auto assignments_range = eve::algo::as_range(aligned_ptr, unaligned_end);
        auto zipped_data = eve::views::zip(samples, assignments_range);
        return ::assign_samples_to_centroids<true>(
            zipped_data,
            centroids,
            std::span<const float>(centroid_norms.data(), centroid_norms.size())
        );
    }

    void clear_dirty_clusters() {
        incremental_update.clear_dirty_clusters();
    }

    void mark_dirty_cluster(std::size_t k) {
        incremental_update.mark_dirty_cluster(k);
    }

    void add_sample_to_sum(std::size_t k, std::size_t n) {
        ::add_sample_to_sum(samples, sum, k, n);
    }

    void subtract_sample_from_sum(std::size_t k, std::size_t n) {
        ::subtract_sample_from_sum(samples, sum, k, n);
    }

    bool resolve_dead_centroids_and_mark_dirty(
        std::span<const int> assignments,
        std::span<int> counts
    ) {
        struct ops_t {
            kumi_kmeans_backend& backend;

            std::size_t N() const { return backend.samples.size(); }

            float distance_to_old_centroid(std::size_t n, std::size_t old_k) const {
                return kumi::inner_product(
                    backend.samples.get(n),
                    backend.centroids[old_k],
                    0.0f,
                    [](auto acc, auto x) { return acc + x; },
                    [](auto p, auto c) {
                        auto diff = p - c;
                        return diff * diff;
                    }
                );
            }

            void relocate_empty_cluster(
                std::size_t old_k,
                std::size_t new_k,
                std::size_t n
            ) {
                backend.subtract_sample_from_sum(old_k, n);
                backend.sum[new_k] = backend.samples.get(n);

                backend.mark_dirty_cluster(old_k);
                backend.mark_dirty_cluster(new_k);
            }
        };

        ops_t ops{ *this };

        return kmeans::resolve_dead_centroids_common(ops, assignments, counts);
    }

    float write_dirty_centroids(std::span<const int> counts) {
        float shift_sq = 0.0f;

        for (int dirty_k : incremental_update.dirty_clusters_span()) {
            const std::size_t k = static_cast<std::size_t>(dirty_k);

            if (counts[k] <= 0) continue;

            shift_sq += ::write_centroid_from_sum(centroids, sum, k, counts[k]);
        }

        return shift_sq;
    }

    void refresh_dirty_centroid_data() {
        recompute_dirty_centroid_norms();
    }

    kmeans::centroid_update_result update_centroids_full(
        const assignment_vector& assignments,
        counts_vector& counts
    ) {
        return ::update_centroids(
            samples,
            std::span<const int>(assignments.data(), assignments.size()),
            centroids,
            sum,
            counts
        );
    }

    void refresh_all_centroid_data() {
        recompute_centroid_norms();
    }

    kmeans::centroid_update_result update_centroids(
        assignment_vector& assignments,
        counts_vector& counts
    ) {
        return kmeans::update_centroids_incremental_or_full_common(
            *this,
            incremental_update,
            assignments,
            counts,
            samples.size() / 4
        );
    }
};

template <eve::product_type SampleType>
auto k_means(
    const eve::algo::soa_vector<SampleType>& samples,
    std::vector<SampleType>& centroids,
    int& out_algorithm_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    kumi_kmeans_backend<SampleType> backend{ samples, centroids };

    return kmeans::k_means_core(
        backend,
        out_algorithm_iterations,
        max_iterations,
        tol
    );
}
