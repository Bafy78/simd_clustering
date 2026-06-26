#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <eve/arch.hpp>

#include "dynamic_d/backend.hpp"

namespace kmeans::lloyd_dispatch {

// Adaptive Lloyd dispatch between the static-D implementation and the
// dynamic-D micro-GEMM implementation.
//
// The decision boundary is empirical: it was fitted from a relatively dense
// benchmark grid over (K, N, D), measured on one development machine.  It is
// intentionally kept as a small runtime policy rather than baked into either
// backend because the two implementations are both useful baselines and may
// scale differently on other CPUs.
//
// Current rule:
//   * never compile or run static-D Lloyd above max_static_lloyd_d
//   * otherwise use static-D when D < dynamic_micro_gemm_threshold(K, N)
//   * otherwise use dynamic-D micro-GEMM
//
// Future multi-machine benchmark sweeps may either retune these constants
// toward a machine-independent middle ground, or introduce architecture-
// specific threshold constants if CPU families show materially different
// scaling behavior.
enum class lloyd_impl {
    static_d,
    dynamic_d_micro_gemm
};

inline constexpr std::size_t max_static_lloyd_d = 50;

template<std::size_t D>
inline constexpr bool static_lloyd_enabled_v = D <= max_static_lloyd_d;

// Rounded D threshold at which the dynamic-D micro-GEMM path is expected to
// beat the static-D path for the given cluster count and sample count.
//
//     threshold = round(D0 + max(0, A - B * log(N)) * C * K^-p)
//
// Larger N reduces the small-N penalty, and larger K discounts it sharply.
// The asymptotic threshold is D0=4.
inline int dynamic_micro_gemm_threshold(std::size_t K, std::size_t N) {
    const double D0 = 4.0;
    const double A  = 15.4205;
    const double B  = 0.89468;
    const double C  = 1258.93;
    const double p  = 4;

    const double excess_from_small_n = std::max(
        0.0,
        A - B * std::log(static_cast<double>(N))
    );

    const double cluster_discount = C * std::pow(static_cast<double>(K), -p);

    const double continuous_threshold = D0 + excess_from_small_n * cluster_discount;

    return static_cast<int>(std::floor(continuous_threshold + 0.5));
}

inline bool use_dynamic_micro_gemm(
    std::size_t K,
    std::size_t N,
    std::size_t D
) {
    return D > max_static_lloyd_d
        || static_cast<int>(D) >= dynamic_micro_gemm_threshold(K, N);
}

inline lloyd_impl choose_lloyd_impl(
    std::size_t K,
    std::size_t N,
    std::size_t D
) {
    return use_dynamic_micro_gemm(K, N, D)
        ? lloyd_impl::dynamic_d_micro_gemm
        : lloyd_impl::static_d;
}

// N_VECTORS is chosen from the SIMD register file size for the dispatched
// dynamic path.  K_TILE remains the benchmark/build-time tuning knob.
//
// This follows the same intent as the existing centroids-at-once style tuning:
// wider register files can keep more sample vectors live without creating as
// much spilling pressure.
inline constexpr std::size_t micro_gemm_n_vectors_for_register_file() {
    constexpr std::size_t regs = static_cast<std::size_t>(eve::register_count::simd);

    if constexpr (regs >= 32) {
        return 4;
    } else if constexpr (regs >= 16) {
        return 2;
    } else {
        return 1;
    }
}

template<std::size_t D, std::size_t K_TILE>
::aligned_int_vector k_means_micro_gemm_auto_n_vectors(
    ::samples_soa_view<D> samples,
    ::centroids_storage<D>& centroids,
    int& out_algorithm_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    return ::k_means_micro_gemm<
        D,
        micro_gemm_n_vectors_for_register_file(),
        K_TILE
    >(
        samples,
        centroids,
        out_algorithm_iterations,
        max_iterations,
        tol
    );
}

} // namespace kmeans::lloyd_dispatch
