#pragma once

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<float> read_binary_f32(
    const std::string& filename,
    std::size_t value_count
) {
    std::vector<float> values(value_count);

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error: Could not open file " + filename);
    }

    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );

    if (!file) {
        throw std::runtime_error("Error: Could not read expected float data from " + filename);
    }

    return values;
}

inline std::vector<float> read_aos_f32(
    const std::string& filename,
    std::size_t rows,
    std::size_t cols
) {
    return read_binary_f32(filename, rows * cols);
}
