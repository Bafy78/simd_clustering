#include <iostream>
#include <string>
#include <vector>

#include <valgrind/callgrind.h>

#include "../include/io_utils.hpp"
#include "../include/k_means/dynamic_d/backend.hpp"

#ifndef KMEANS_K_TILE
#define KMEANS_K_TILE 5
#endif

#ifndef KMEANS_M_GROUP
#define KMEANS_M_GROUP 2
#endif

template<typename StaticPoints>
points_soa_storage<TUPLE_SIZE> make_dynamic_points(
    const StaticPoints& static_points
) {
    points_soa_storage<TUPLE_SIZE> out(static_points.size());

    for (std::size_t i = 0; i < static_points.size(); ++i) {
        auto pt = static_points.get(i);

        kumi::for_each_index(
            [&](auto index, auto value) {
                constexpr std::size_t d = decltype(index)::value;
                out(i, d) = value;
            },
            pt
        );
    }

    return out;
}

inline centroids_storage<TUPLE_SIZE> make_dynamic_centroids(
    const std::vector<PointType>& static_centroids
) {
    centroids_storage<TUPLE_SIZE> out;
    out.resize(static_centroids.size());

    for (std::size_t k = 0; k < static_centroids.size(); ++k) {
        kumi::for_each_index(
            [&](auto index, auto value) {
                constexpr std::size_t d = decltype(index)::value;
                out.row(k, d) = value;
            },
            static_centroids[k]
        );
    }

    return out;
}

inline std::vector<PointType> make_static_centroids(
    const centroids_storage<TUPLE_SIZE>& dynamic_centroids
) {
    std::vector<PointType> out(dynamic_centroids.n_clusters);

    for (std::size_t k = 0; k < dynamic_centroids.n_clusters; ++k) {
        PointType pt{};

        kumi::for_each_index(
            [&](auto index, auto& value) {
                constexpr std::size_t d = decltype(index)::value;
                value = dynamic_centroids.row(k, d);
            },
            pt
        );

        out[k] = pt;
    }

    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr
            << "Usage: " << argv[0]
            << " <dataset_bin> <n_samples> <n_clusters> <init_centroids_bin> <metrics_json_out>\n";
        return 1;
    }

    std::string dataset_bin = argv[1];
    std::size_t n_samples = std::stoull(argv[2]);
    int n_clusters = std::stoi(argv[3]);
    std::string init_centroids_bin = argv[4];
    std::string metrics_json_out = argv[5];

    // Setup is intentionally outside the measured region when Valgrind is run
    // with --instr-atstart=no.
    auto static_points = read_dataset_soa(dataset_bin, n_samples);
    auto static_initial_centroids = read_initial_centroids_binary(init_centroids_bin, n_clusters);
    auto dynamic_points_storage = make_dynamic_points(static_points);
    auto dynamic_points = dynamic_points_storage.view();
    auto dynamic_centroids = make_dynamic_centroids(static_initial_centroids);

    int iterations_to_converge = 0;

    CALLGRIND_START_INSTRUMENTATION;

    auto assignments = k_means_micro_gemm<TUPLE_SIZE, KMEANS_K_TILE, KMEANS_M_GROUP>(
        dynamic_points,
        dynamic_centroids,
        iterations_to_converge
    );

    CALLGRIND_STOP_INSTRUMENTATION;

    auto final_centroids = make_static_centroids(dynamic_centroids);

    write_lloyd_metrics(
        metrics_json_out,
        static_points,
        final_centroids,
        assignments,
        n_clusters,
        iterations_to_converge
    );

    return 0;
}
