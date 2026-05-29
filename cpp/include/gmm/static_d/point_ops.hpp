#pragma once

#include <eve/module/core.hpp>
#include <eve/wide.hpp>

#include "../../simd.hpp"


template <eve::product_type PointT>
float gmm_point_norm_sq(const PointT& point) {
    return kumi::inner_product(point, point, 0.0f);
}

template <eve::product_type SimdPointT, eve::product_type PointT>
wide_f gmm_dot_simd_point_with_mean(const SimdPointT& point, const PointT& mean) {
    auto dot = eve::zero(eve::as<wide_f>());

    kumi::for_each(
        [&](auto x, auto mu) {
            dot = eve::fma(x, wide_f(mu), dot);
        },
        point,
        mean
    );

    return dot;
}

template <eve::product_type SimdPointT>
wide_f gmm_simd_point_norm_sq(const SimdPointT& point) {
    auto norm_sq = eve::zero(eve::as<wide_f>());

    kumi::for_each(
        [&](auto x) {
            norm_sq = eve::fma(x, x, norm_sq);
        },
        point
    );

    return norm_sq;
}
