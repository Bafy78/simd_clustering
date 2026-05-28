#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>

#ifndef TUPLE_SIZE
#define TUPLE_SIZE 2
#endif

template<std::size_t D>
using static_point_type = kumi::result::fill_t<D, float>;

using PointType = static_point_type<TUPLE_SIZE>;

template<std::size_t D>
using static_points_soa_vector = eve::algo::soa_vector<static_point_type<D>>;

template<std::size_t D>
void check_static_aos_size(std::span<const float> aos, std::size_t rows, const char* label) {
    if (aos.size() != rows * D) {
        throw std::runtime_error(std::string(label) + " AoS buffer has unexpected size");
    }
}

template<std::size_t D>
void copy_aos_to_static_points(
    std::span<const float> aos,
    std::size_t n_samples,
    static_points_soa_vector<D>& points
) {
    check_static_aos_size<D>(aos, n_samples, "points");

    for (std::size_t i = 0; i < n_samples; ++i) {
        static_point_type<D> pt;

        kumi::for_each_index(
            [&](auto index, auto& element) {
                element = aos[i * D + index];
            },
            pt
        );

        points.set(i, pt);
    }
}

template<std::size_t D>
static_points_soa_vector<D> make_static_points_from_aos(
    std::span<const float> aos,
    std::size_t n_samples
) {
    static_points_soa_vector<D> points(eve::algo::no_init, n_samples);
    copy_aos_to_static_points<D>(aos, n_samples, points);
    return points;
}

template<std::size_t D>
std::vector<static_point_type<D>> make_static_vectors_from_aos(
    std::span<const float> aos,
    std::size_t rows,
    const char* label = "vectors"
) {
    check_static_aos_size<D>(aos, rows, label);

    std::vector<static_point_type<D>> vectors(rows);

    for (std::size_t row = 0; row < rows; ++row) {
        kumi::for_each_index(
            [&](auto index, auto& element) {
                element = aos[row * D + index];
            },
            vectors[row]
        );
    }

    return vectors;
}
