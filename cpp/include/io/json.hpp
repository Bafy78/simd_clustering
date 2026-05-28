#pragma once

#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <utility>

#include <eve/module/core.hpp>

template <eve::product_type PointT>
void write_point_json(std::ostream& out, const PointT& point) {
    out << "[";

    bool first = true;

    kumi::for_each(
        [&](auto const& value) {
            out << (first ? "" : ", ")
                << std::setprecision(std::numeric_limits<float>::max_digits10)
                << static_cast<float>(value);

            first = false;
        },
        point
    );

    out << "]";
}
