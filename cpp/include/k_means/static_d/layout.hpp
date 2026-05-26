#pragma once

#include <cstddef>

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
