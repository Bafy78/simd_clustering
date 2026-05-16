#pragma once

#include <ranges>
#include <algorithm>
#include <vector>
#include <numeric>
#include <span>
#include <type_traits>

#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "./k_means_core.hpp"


template <typename SimdPoint, typename PointType>
auto compute_simd_assignment_score(
    const SimdPoint& pt,
    const PointType& centroid,
    float centroid_norm_sq
) {
    using wide_dist = std::remove_cvref_t<decltype(get<0>(pt))>;

    auto dot = eve::zero(eve::as<wide_dist>());

    kumi::for_each_index([&](auto index, auto c) {
        constexpr std::size_t d = decltype(index)::value;

        dot = eve::fma(
            get<d>(pt),
            wide_dist(static_cast<float>(c)),
            dot
        );
    }, centroid);

    return wide_dist(centroid_norm_sq) - wide_dist(2.0f) * dot;
}

template <eve::product_type PointType>
float compute_sklearn_tolerance(
    const eve::algo::soa_vector<PointType>& points,
    float tol
) {
    constexpr std::size_t D = kumi::size_v<PointType>;

    auto sample_value = [&](auto feature_index, std::size_t sample_i) {
        constexpr std::size_t d = decltype(feature_index)::value;
        return kumi::get<d>(points.get(sample_i));
    };

    return kmeans::compute_sklearn_tolerance_common<D>(
        points.size(),
        sample_value,
        tol
    );
}

template <bool TrackChanges = false, class Label = int, eve::algo::relaxed_range R, typename PointType>
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

        using wide_dist = decltype(compute_simd_assignment_score(
            pt,
            centroids[0],
            centroid_norms[0]
        ));
        using wide_index = eve::wide<int, typename wide_dist::cardinal_type>;
        using wide_label = eve::wide<Label, typename wide_dist::cardinal_type>;

        auto min_distances = eve::valmax(eve::as<wide_dist>());
        auto closest_centroid_indices = eve::zero(eve::as<wide_index>());

        for (std::size_t k = 0; k < centroids.size(); ++k) {
            auto assignment_score = compute_simd_assignment_score(
                pt,
                centroids[k],
                centroid_norms[k]
            );

            auto is_closer = assignment_score < min_distances;
            min_distances = eve::min(min_distances, assignment_score);
            closest_centroid_indices = eve::if_else(is_closer, wide_index(static_cast<int>(k)), closest_centroid_indices);
        }

        const wide_label closest_centroid_labels = eve::convert(
            closest_centroid_indices,
            eve::as<Label>()
        );

        if constexpr (TrackChanges) {
            if (!any_changed) {
                const wide_label previous_centroid_labels = eve::load[ignore](assign_it);

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

template <eve::product_type PointType, class Label>
void resolve_dead_centroids(
    const eve::algo::soa_vector<PointType>& points,
    std::span<const Label> assignments,
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
            const auto pt = points.get(sample_i);
            const auto& centroid = old_centroids[old_label];

            float dist = 0.0f;
            kumi::for_each([&dist](auto p, auto c) {
                const float diff = p - c;
                dist += diff * diff;
            }, pt, centroid);

            return dist;
        }

        void relocate_empty_cluster(
            std::size_t old_cluster_id,
            std::size_t new_cluster_id,
            std::size_t sample_i
        ) {
            const auto pt = points.get(sample_i);
            kumi::for_each([](auto& dst, auto p) { dst -= p; }, sum[old_cluster_id], pt);
            kumi::for_each([](auto& dst, auto p) { dst = p; }, sum[new_cluster_id], pt);
        }
    };

    ops_t ops{ points, old_centroids, sum };

    kmeans::resolve_dead_centroids_common(
        ops,
        assignments,
        counts
    );
}

template <eve::product_type PointType, class Label>
void update_centroids(
    const eve::algo::soa_vector<PointType>& points,
    std::span<const Label> assignments,
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
            for (auto& row : sum) {
                kumi::for_each([](auto& v) { v = 0.0f; }, row);
            }
        }

        void add_point_to_sum(std::size_t cluster_idx, std::size_t sample_i) {
            const auto pt = points.get(sample_i);
            kumi::for_each([](auto& s, auto p) { s += p; }, sum[cluster_idx], pt);
        }

        void resolve_dead_centroids(std::span<const Label> assignments, std::span<int> counts) {
            ::resolve_dead_centroids(points, assignments, centroids, sum, counts);
        }

        void write_centroid_from_sum(std::size_t cluster_idx, int count) {
            kumi::for_each([count](auto& cent, auto s) {
                cent = s / static_cast<float>(count);
            }, centroids[cluster_idx], sum[cluster_idx]);
        }

        void after_centroids_updated() {}
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
    constexpr std::size_t D = kumi::size_v<PointType>;

    auto old_value = [&](std::size_t k, auto feature_index) {
        constexpr std::size_t d = decltype(feature_index)::value;
        return kumi::get<d>(old_centroids[k]);
    };

    auto new_value = [&](std::size_t k, auto feature_index) {
        constexpr std::size_t d = decltype(feature_index)::value;
        return kumi::get<d>(new_centroids[k]);
    };

    return kmeans::calculate_centroid_shift_sq_common<D>(
        new_centroids.size(),
        old_value,
        new_value
    );
}


template <eve::product_type PointType, class Label = kmeans::default_label_t>
struct kumi_kmeans_backend {
    using label_type = Label;
    using assignment_cardinal = typename eve::wide<float>::cardinal_type;
    using assignment_vector = std::vector<label_type, eve::aligned_allocator<label_type>>;
    using counts_vector = std::vector<int>;
    using centroid_snapshot = std::vector<PointType>;

    static constexpr label_type invalid_label = kmeans::invalid_label_v<label_type>;

    const eve::algo::soa_vector<PointType>& points;
    std::vector<PointType>& centroids;

    std::vector<PointType> sum;
    std::vector<float> centroid_norms;

    kumi_kmeans_backend(
        const eve::algo::soa_vector<PointType>& points_,
        std::vector<PointType>& centroids_
    )
        : points(points_),
        centroids(centroids_),
        sum(centroids_.size()),
        centroid_norms(centroids_.size()) {
        recompute_centroid_norms();
    }

    void recompute_centroid_norms() {
        centroid_norms.resize(centroids.size());

        for (std::size_t k = 0; k < centroids.size(); ++k) {
            float norm = 0.0f;

            kumi::for_each([&](auto c) {
                norm += c * c;
            }, centroids[k]);

            centroid_norms[k] = norm;
        }
    }

    void check_cluster_count() const {
        kmeans::check_cluster_count_fits<label_type>(centroids.size());
        kmeans::check_cluster_count(centroids.size(), points.size());
    }

    assignment_vector make_assignment_vector(label_type initial_value) const {
        return assignment_vector(points.size(), initial_value);
    }

    counts_vector make_counts_vector() const {
        return counts_vector(centroids.size(), 0);
    }

    float compute_tolerance(float tol) const {
        return ::compute_sklearn_tolerance(points, tol);
    }

    centroid_snapshot make_centroid_snapshot() const {
        return centroid_snapshot(centroids.size());
    }

    void save_centroids(centroid_snapshot& snapshot) const {
        snapshot = centroids;
    }

    void assign(assignment_vector& assignments) const {
        auto aligned_ptr = eve::as_aligned(assignments.data(), assignment_cardinal{});
        auto unaligned_end = assignments.data() + assignments.size();
        auto assignments_range = eve::algo::as_range(aligned_ptr, unaligned_end);
        auto zipped_data = eve::views::zip(points, assignments_range);
        (void)::assign_points_to_centroids<false, label_type>(
            zipped_data,
            centroids,
            std::span<const float>(centroid_norms.data(), centroid_norms.size())
        );
    }
    bool assign_and_check_changed(assignment_vector& assignments) const {
        auto aligned_ptr = eve::as_aligned(assignments.data(), assignment_cardinal{});
        auto unaligned_end = assignments.data() + assignments.size();
        auto assignments_range = eve::algo::as_range(aligned_ptr, unaligned_end);
        auto zipped_data = eve::views::zip(points, assignments_range);
        return ::assign_points_to_centroids<true, label_type>(
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
            std::span<const label_type>(assignments.data(), assignments.size()),
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

template <class Label = kmeans::default_label_t, eve::product_type PointType>
auto k_means(
    const eve::algo::soa_vector<PointType>& points,
    std::vector<PointType>& centroids,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    kumi_kmeans_backend<PointType, Label> backend{ points, centroids };

    return kmeans::k_means_core(
        backend,
        out_iterations,
        max_iterations,
        tol
    );
}
