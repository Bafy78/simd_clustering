#pragma once

#include <cstddef>

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>
#include <eve/wide.hpp>

using hdbscan_real = double;
using hdbscan_wide = eve::wide<hdbscan_real>;
inline const hdbscan_wide hdbscan_wide_zero = eve::zero(eve::as<hdbscan_wide>{});

using hdbscan_cardinal = typename hdbscan_wide::cardinal_type;
using hdbscan_wide_i = eve::wide<int, hdbscan_cardinal>;
inline const hdbscan_wide_i hdbscan_wide_zero_i = eve::zero(eve::as<hdbscan_wide_i>{});

template<std::size_t Dim>
using hdbscan_static_sample_type = kumi::result::fill_t<Dim, hdbscan_real>;

template<std::size_t Dim>
using hdbscan_static_samples_soa_vector = eve::algo::soa_vector<hdbscan_static_sample_type<Dim>>;
