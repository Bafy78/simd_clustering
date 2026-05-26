#pragma once

#include <algorithm>
#include <cstddef>
#include <new>
#include <utility>
#include <vector>

#include <eve/arch.hpp>
#include <eve/module/core.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "../core.hpp"

using wide_f = kmeans::wide_f;
using wide_i = kmeans::wide_i;

using aligned_float_vector = std::vector<float, eve::aligned_allocator<float>>;
using aligned_int_vector = std::vector<int, eve::aligned_allocator<int>>;

template<class T>
struct no_init_aligned_allocator : eve::aligned_allocator<T> {
    using value_type = T;

    no_init_aligned_allocator() noexcept = default;

    template<class U>
    no_init_aligned_allocator(const no_init_aligned_allocator<U>&) noexcept {}

    template<class U>
    struct rebind {
        using other = no_init_aligned_allocator<U>;
    };

    void construct(T* p) {
        ::new (static_cast<void*>(p)) T; // default-init, not value-init
    }

    template<class... Args>
    void construct(T* p, Args&&... args) {
        ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
    }
};

using no_init_aligned_float_vector = std::vector<float, no_init_aligned_allocator<float>>;

inline constexpr std::size_t simd_cardinal() {
    return static_cast<std::size_t>(wide_f::size());
}

inline std::size_t round_up_to_multiple(std::size_t n, std::size_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

// Static-D feature-major point view:
//     points[d][i]
//
// D is compile-time.
// stride is runtime because n_samples is runtime.
// stride is padded to SIMD cardinality so each feature column can be loaded safely in SIMD chunks.
template<std::size_t D>
struct points_soa_view {
    static constexpr std::size_t n_features = D;

    float* data = nullptr;

    std::size_t n_samples = 0;
    std::size_t stride = 0;

    float* feature(std::size_t d) const {
        return data + d * stride;
    }
};

template<std::size_t D>
struct points_soa_storage {
    static constexpr std::size_t n_features = D;

    no_init_aligned_float_vector data;

    std::size_t n_samples = 0;
    std::size_t stride = 0;

    points_soa_storage() = default;

    explicit points_soa_storage(std::size_t samples) {
        resize(samples);
    }

    void resize(std::size_t samples) {
        n_samples = samples;
        stride = round_up_to_multiple(samples, simd_cardinal());

        data.resize(D * stride);
    }

    float& operator()(std::size_t sample, std::size_t feature) {
        return data[feature * stride + sample];
    }

    float operator()(std::size_t sample, std::size_t feature) const {
        return data[feature * stride + sample];
    }

    const float* feature(std::size_t d) const {
        return data.data() + d * stride;
    }

    points_soa_view<D> view() {
        return points_soa_view<D>{
            .data = data.data(),
            .n_samples = n_samples,
            .stride = stride,
        };
    }

    points_soa_view<D> view() const {
        return points_soa_view<D>{
            .data = const_cast<float*>(data.data()),
            .n_samples = n_samples,
            .stride = stride,
        };
    }
};

// Static-D centroid storage.
//
// row_major is the single source of truth:
//     centroids[k][d]
//
// Assignment-specific layouts are owned and refreshed by the assignment
// backends when on_centroids_changed() is called.
template<std::size_t D>
struct centroids_storage {
    static constexpr std::size_t n_features = D;

    aligned_float_vector row_major;

    std::size_t n_clusters = 0;

    centroids_storage() = default;

    void resize(std::size_t clusters) {
        n_clusters = clusters;
        row_major.resize(n_clusters * D);
        std::fill(row_major.begin(), row_major.end(), 0.0f);
    }

    float& row(std::size_t k, std::size_t d) {
        return row_major[k * D + d];
    }

    float row(std::size_t k, std::size_t d) const {
        return row_major[k * D + d];
    }

    template<std::size_t Feature>
    float& row(std::size_t k) {
        static_assert(Feature < D);
        return row_major[k * D + Feature];
    }

    template<std::size_t Feature>
    float row(std::size_t k) const {
        static_assert(Feature < D);
        return row_major[k * D + Feature];
    }
};
