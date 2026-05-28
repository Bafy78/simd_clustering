#pragma once

#include <eve/wide.hpp>

using wide_f = eve::wide<float>;
using cardinal = typename wide_f::cardinal_type;
using wide_i = eve::wide<int, cardinal>;
