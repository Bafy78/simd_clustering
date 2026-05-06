#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <span>
#include <string>
#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>

#ifndef TUPLE_SIZE
#define TUPLE_SIZE 2
#endif

using PointType = kumi::result::fill_t<TUPLE_SIZE, float>;

// --- Helper: Read Raw Binary Data & Convert to SoA (For Setup) ---
inline eve::algo::soa_vector<PointType> read_dataset_soa(const std::string& filename, std::size_t n_samples) {
    std::vector<float> raw_aos_data(n_samples * TUPLE_SIZE);
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Error: Could not open file " + filename);
    
    file.read(reinterpret_cast<char*>(raw_aos_data.data()), raw_aos_data.size() * sizeof(float));
    
    eve::algo::soa_vector<PointType> points(n_samples);
    for (std::size_t i = 0; i < n_samples; ++i) {
        PointType pt;
        kumi::for_each_index([&](auto index, auto& element) {
            element = raw_aos_data[i * TUPLE_SIZE + index];
        }, pt);
        points.set(i, pt);
    }
    return points;
}

// --- Helper: Read Orchestrator's Initial Centroids ---
inline std::vector<PointType> read_initial_centroids_binary(const std::string& filename, int n_clusters) {
    std::vector<PointType> centroids(n_clusters);
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Error: Could not open initial centroids file " + filename);

    for (auto& c : centroids) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (..., in.read(reinterpret_cast<char*>(&get<I>(c)), sizeof(float)));
        }(std::make_index_sequence<TUPLE_SIZE>{});
    }
    return centroids;
}

// --- Helper: Write Final Results ---
template <eve::product_type PointType>
void write_results(const std::string& filename, 
                   const std::vector<PointType>& centroids, 
                   std::span<const int> assignments, 
                   int num_clusters,
                   int iterations) {
    std::ofstream out(filename);
    
    out << "[Lloyd Iterations]\n" << iterations << "\n";
    
    out << "[Centroids]\n";
    for (const auto& c : centroids) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((out << get<I>(c) << (I == TUPLE_SIZE - 1 ? "" : " ")), ...);
        }(std::make_index_sequence<TUPLE_SIZE>{});
        out << "\n";
    }

    out << "[Clusters]\n";
    std::vector<std::vector<int>> sets(num_clusters);
    for (std::size_t i = 0; i < assignments.size(); ++i) {
        sets[assignments[i]].push_back(i);
    }

    for (int k = 0; k < num_clusters; ++k) {
        out << k << ":";
        for (int idx : sets[k]) {
            out << " " << idx;
        }
        out << "\n";
    }
}