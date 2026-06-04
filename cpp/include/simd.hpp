#pragma once

#include <eve/wide.hpp>

using wide_f = eve::wide<float>;
inline const wide_f wide_zero_f = eve::zero(eve::as<wide_f>{});

using cardinal = typename wide_f::cardinal_type;

using wide_i = eve::wide<int, cardinal>;
inline const wide_i wide_zero_i = eve::zero(eve::as<wide_i>{});