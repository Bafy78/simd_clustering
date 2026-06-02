#pragma once

#include <cstddef>
#include <new>
#include <utility>
#include <vector>

#include <eve/arch.hpp>
#include <eve/module/core.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "../simd.hpp"

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

// Compile-time-D dimension-major sample view:
//     samples[d][i]
//
// D is compile-time.
// stride is runtime because N is runtime.
// stride is padded to SIMD cardinality so each dimension column can be loaded safely in SIMD chunks.
template<std::size_t D>
struct samples_soa_view {
    float* data = nullptr;

    std::size_t N = 0;
    std::size_t stride = 0;

    float* dimension(std::size_t d) const {
        return data + d * stride;
    }
};

template<std::size_t D>
struct samples_soa_storage {
    no_init_aligned_float_vector data;

    std::size_t N = 0;
    std::size_t stride = 0;

    samples_soa_storage() = default;

    explicit samples_soa_storage(std::size_t N) {
        resize(N);
    }

    void resize(std::size_t new_N) {
        N = new_N;
        stride = round_up_to_multiple(new_N, simd_cardinal());

        data.resize(D * stride);
    }

    float& operator()(std::size_t n, std::size_t d) {
        return data[d * stride + n];
    }

    float operator()(std::size_t n, std::size_t d) const {
        return data[d * stride + n];
    }

    const float* dimension(std::size_t d) const {
        return data.data() + d * stride;
    }

    samples_soa_view<D> view() {
        return samples_soa_view<D>{
            .data = data.data(),
            .N = N,
            .stride = stride,
        };
    }

    samples_soa_view<D> view() const {
        return samples_soa_view<D>{
            .data = const_cast<float*>(data.data()),
            .N = N,
            .stride = stride,
        };
    }
};
