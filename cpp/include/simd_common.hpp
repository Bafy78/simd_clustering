#include <eve/module/core.hpp>
#include <eve/wide.hpp>

// Computes SIMD squared distance between a block of points and a centroid
constexpr auto compute_simd_dist_sq = [](const auto& pt, const auto& centroid) {
    auto dist_sq = eve::zero(eve::as<eve::wide<float>>());
    kumi::for_each([&](auto p, auto c) {
        auto diff = p - c;
        dist_sq = eve::fma(diff, diff, dist_sq); 
    }, pt, centroid);
    return dist_sq;
};