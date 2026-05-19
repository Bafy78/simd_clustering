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

using wide_f = kmeans::wide_f;
using wide_i = kmeans::wide_i;

template <typename SimdPoint, typename PointType>
auto compute_simd_assignment_score(
    const SimdPoint& pt,
    const PointType& centroid,
    float centroid_norm_sq
) {
    auto dot = eve::zero(eve::as<wide_f>());

    kumi::for_each([&](auto p, auto c) {
        dot = eve::fma(p, wide_f(c), dot);
    }, pt, centroid);

    return eve::fma(dot, wide_f(-2.0f), wide_f(centroid_norm_sq));
}

template <typename SimdPoint, typename PointType>
wide_i compute_closest_centroid_labels(
    const SimdPoint& pt,
    const std::vector<PointType>& centroids,
    std::span<const float> centroid_norms
) {
    auto min_distances = eve::valmax(eve::as<wide_f>());
    auto closest_centroid_labels = eve::zero(eve::as<wide_i>());

    for (std::size_t k = 0; k < centroids.size(); ++k) {
        auto assignment_score = compute_simd_assignment_score(
            pt,
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

template <bool TrackChanges = false, eve::algo::relaxed_range R, typename PointType>
bool assign_points_to_centroids(
    R&& zipped_data,
    const std::vector<PointType>& centroids,
    std::span<const float> centroid_norms
) {
    bool any_changed = false;

    eve::algo::for_each(
        zipped_data,
        [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {

        auto [pt_it, assign_it] = it;
        auto pt = eve::load[ignore](pt_it);

        auto min_distances = eve::valmax(eve::as<wide_f>());
        auto closest_centroid_labels = compute_closest_centroid_labels(pt, centroids, centroid_norms);

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

template <eve::product_type PointType>
void resolve_dead_centroids(
    const eve::algo::soa_vector<PointType>& points,
    std::span<const int> assignments,
    const std::vector<PointType>& old_centroids,
    std::vector<PointType>& sum,
    std::span<int> counts
) {
    struct ops_t {
        const eve::algo::soa_vector<PointType>& points;
        const std::vector<PointType>& old_centroids;
        std::vector<PointType>& sum;

        std::size_t n_samples() const { return points.size(); }

        float distance_to_old_centroid(std::size_t sample_i, std::size_t old_label) const {
            return kumi::inner_product(
                points.get(sample_i),
                old_centroids[old_label],
                0.0f,
                [](auto acc, auto x) { return acc + x; },
                [](auto p, auto c) {
                    auto diff = p - c;
                    return diff * diff;
                }
            );
        }

        void relocate_empty_cluster(
            std::size_t old_cluster_id,
            std::size_t new_cluster_id,
            std::size_t sample_i
        ) {
            const auto pt = points.get(sample_i);
            kumi::for_each([](auto& dst, auto p) { dst -= p; }, sum[old_cluster_id], pt);
            sum[new_cluster_id] = pt;
        }
    };

    ops_t ops{ points, old_centroids, sum };

    kmeans::resolve_dead_centroids_common(
        ops,
        assignments,
        counts
    );
}

template <eve::product_type PointType>
void update_centroids(
    const eve::algo::soa_vector<PointType>& points,
    std::span<const int> assignments,
    std::vector<PointType>& centroids,
    std::vector<PointType>& sum,
    std::vector<int>& counts
) {
    struct ops_t {
        const eve::algo::soa_vector<PointType>& points;
        std::vector<PointType>& centroids;
        std::vector<PointType>& sum;

        std::size_t n_samples() const { return points.size(); }
        std::size_t n_clusters() const { return centroids.size(); }

        void reset_sums() {
            for (auto& row : sum) row = PointType{};
        }

        void add_point_to_sum(std::size_t cluster_idx, std::size_t sample_i) {
            const auto pt = points.get(sample_i);
            kumi::for_each([](auto& s, auto p) { s += p; }, sum[cluster_idx], pt);
        }

        void resolve_dead_centroids(std::span<const int> assignments, std::span<int> counts) {
            ::resolve_dead_centroids(points, assignments, centroids, sum, counts);
        }

        void write_centroid_from_sum(std::size_t cluster_idx, int count) {
            const float inv_count = 1.0f / static_cast<float>(count);
            centroids[cluster_idx] = kumi::map(
                    [inv_count](auto s) { return s * inv_count; },
                    sum[cluster_idx]
                );
        }
    };

    ops_t ops{ points, centroids, sum };

    kmeans::update_centroids_common(
        ops,
        assignments,
        std::span<int>(counts.data(), counts.size())
    );
}

template <eve::product_type PointType>
float calculate_centroid_shift_sq(
    const std::vector<PointType>& old_centroids,
    const std::vector<PointType>& new_centroids
) {
    float shift_sq = 0.0f;

    for (std::size_t k = 0; k < new_centroids.size(); ++k) {
        shift_sq = kumi::inner_product(
            old_centroids[k],
            new_centroids[k],
            shift_sq,
            [](auto acc, auto x) { return acc + x;},
            [](auto old_c, auto new_c) {
                auto diff = new_c - old_c;
                return diff * diff;
            }
        );
    }

    return shift_sq;
}


template <eve::product_type PointType>
struct kumi_kmeans_backend {
    using assignment_vector = std::vector<int, eve::aligned_allocator<int>>;
    using counts_vector = std::vector<int>;
    using centroid_snapshot = std::vector<PointType>;

    const eve::algo::soa_vector<PointType>& original_points;

    eve::algo::soa_vector<PointType> points;
    std::vector<PointType>& centroids;

    std::vector<PointType> sum;
    std::vector<float> centroid_norms;
    PointType feature_mean{};

    kumi_kmeans_backend(
        const eve::algo::soa_vector<PointType>& points_,
        std::vector<PointType>& centroids_
    )
        : original_points(points_),
        points(),
        centroids(centroids_),
        sum(centroids_.size()),
        centroid_norms(centroids_.size()) {}

    void recompute_centroid_norms() {
        centroid_norms.resize(centroids.size());

        for (std::size_t k = 0; k < centroids.size(); ++k) {
            centroid_norms[k] = kumi::inner_product(centroids[k], centroids[k], 0.0f);
        }
    }

    void compute_feature_mean_from_original() {
        if (original_points.size() == 0) {
            feature_mean = PointType{};
            return;
        }

        PointType feature_sum{};

        for (std::size_t i = 0; i < original_points.size(); ++i) {
            const auto pt = original_points.get(i);
            kumi::for_each([](auto& s, auto x) { s += x; }, feature_sum, pt);
        }

        const float inv_n = 1.0f / static_cast<float>(original_points.size());

        feature_mean = kumi::map([inv_n](auto s) { return s * inv_n; }, feature_sum);
    }

    float copy_centered_points_from_original_and_compute_scaled_tolerance(float tol) {
        points.resize(original_points.size());

        eve::algo::transform_to(
            original_points, 
            points,
            [&](auto src_pt) {
                using simd_point_t = std::remove_cvref_t<decltype(src_pt)>;
                auto centered = kumi::map(
                    [](auto x, auto mean) { return x - mean; },
                    src_pt,
                    feature_mean
                );
                return simd_point_t{centered};
            }
        );

        if (tol == 0.0f || points.size() == 0) return 0.0f;

        const float total_squared_centered_norm = 
            eve::algo::transform_reduce[eve::algo::fuse_operations]( 
                points,
                [](auto pt, auto acc) {
                    kumi::for_each([&](auto x) {
                        acc = eve::fma(x, x, acc);
                    }, pt); return acc; 
                },
                0.0f
            );

        return tol
            * total_squared_centered_norm
            / static_cast<float>(points.size())
            / static_cast<float>(kumi::size_v<PointType>);
    }

    void subtract_feature_mean_from_centroids() {
        for (auto& centroid : centroids) {
            kumi::for_each(
                [](auto& c, auto mean) { c -= mean; },
                centroid,
                feature_mean
            );
        }
    }

    void add_feature_mean_to_centroids() {
        for (auto& centroid : centroids) {
            kumi::for_each(
                [](auto& c, auto mean) { c += mean; },
                centroid,
                feature_mean
            );
        }
    }

    float prepare_data_for_fit(float tol) {
        compute_feature_mean_from_original();

        const float scaled_tol = copy_centered_points_from_original_and_compute_scaled_tolerance(tol);

        subtract_feature_mean_from_centroids();
        recompute_centroid_norms();

        return scaled_tol;
    }

    void finish_fit_after_final_assignment() {
        add_feature_mean_to_centroids();
    }

    void check_cluster_count() const {
        kmeans::check_cluster_count(centroids.size(), original_points.size());
    }

    assignment_vector make_assignment_vector(int initial_value) const {
        return assignment_vector(original_points.size(), initial_value);
    }

    counts_vector make_counts_vector() const {
        return counts_vector(centroids.size(), 0);
    }


    centroid_snapshot make_centroid_snapshot() const {
        return centroid_snapshot(centroids.size());
    }

    void save_centroids(centroid_snapshot& snapshot) const {
        snapshot = centroids;
    }

    void assign(assignment_vector& assignments) const {
        auto aligned_ptr = eve::as_aligned(assignments.data(), kmeans::cardinal{});
        auto unaligned_end = assignments.data() + assignments.size();
        auto assignments_range = eve::algo::as_range(aligned_ptr, unaligned_end);
        auto zipped_data = eve::views::zip(points, assignments_range);
        (void)::assign_points_to_centroids<false>(
            zipped_data,
            centroids,
            std::span<const float>(centroid_norms.data(), centroid_norms.size())
        );
    }
    bool assign_and_check_changed(assignment_vector& assignments) const {
        auto aligned_ptr = eve::as_aligned(assignments.data(), kmeans::cardinal{});
        auto unaligned_end = assignments.data() + assignments.size();
        auto assignments_range = eve::algo::as_range(aligned_ptr, unaligned_end);
        auto zipped_data = eve::views::zip(points, assignments_range);
        return ::assign_points_to_centroids<true>(
            zipped_data,
            centroids,
            std::span<const float>(centroid_norms.data(), centroid_norms.size())
        );
    }

    void update_centroids(
        assignment_vector& assignments,
        counts_vector& counts
    ) {
        ::update_centroids(
            points,
            std::span<const int>(assignments.data(), assignments.size()),
            centroids,
            sum,
            counts
        );

        recompute_centroid_norms();
    }

    float centroid_shift_sq(
        const centroid_snapshot& previous_centroids
    ) const {
        return ::calculate_centroid_shift_sq(
            previous_centroids,
            centroids
        );
    }
};

template <eve::product_type PointType>
auto k_means(
    const eve::algo::soa_vector<PointType>& points,
    std::vector<PointType>& centroids,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    kumi_kmeans_backend<PointType> backend{ points, centroids };

    return kmeans::k_means_core(
        backend,
        out_iterations,
        max_iterations,
        tol
    );
}
