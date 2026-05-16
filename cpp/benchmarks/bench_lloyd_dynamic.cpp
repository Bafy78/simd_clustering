#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <bit>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "../include/io_utils.hpp"
#include "../include/kmeans_lloyd_dynamic.hpp"

#ifndef KMEANS_K_TILE
#define KMEANS_K_TILE 4
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
    out.template resize_for_tile<KMEANS_K_TILE>(static_centroids.size());

    for (std::size_t k = 0; k < static_centroids.size(); ++k) {
        kumi::for_each_index(
            [&](auto index, auto value) {
            constexpr std::size_t d = decltype(index)::value;
            out.row(k, d) = value;
        },
            static_centroids[k]
        );
    }

    out.sync_feature_major_from_row_major();

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

template<std::size_t D>
inline void swap_centroids_storage(
    centroids_storage<D>& a,
    centroids_storage<D>& b
) noexcept {
    using std::swap;

    swap(a.row_major, b.row_major);
    swap(a.feature_major, b.feature_major);
    swap(a.n_clusters, b.n_clusters);
    swap(a.feature_major_stride, b.feature_major_stride);
}

int main(int argc, char* argv[]) {
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
            << " <binary_file> <n_samples> <n_clusters> <init_centroids_bin> <metrics_json_out> <nanobench_json_out> <bench_epochs> <min_epoch_seconds>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::size_t n_samples = std::stoull(argv[2]);
    int n_clusters = std::stoi(argv[3]);
    std::string init_centroids_bin = argv[4];
    std::string metrics_json_out = argv[5];
    std::string nanobench_out = argv[6];
    std::size_t bench_epochs = std::stoull(argv[7]);
    double min_epoch_seconds = std::stod(argv[8]);

    auto min_epoch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(min_epoch_seconds)
    );

    // Setup: read through the existing static path, then convert to the
    // static-D streamed/tiled SoA layout. This setup is not benchmarked.
    auto static_points = read_dataset_soa(filename, n_samples);
    auto static_initial_centroids = read_initial_centroids_binary(init_centroids_bin, n_clusters);
    auto dynamic_points_storage = make_dynamic_points(static_points);
    auto dynamic_points = dynamic_points_storage.view();
    auto dynamic_initial_centroids = make_dynamic_centroids(static_initial_centroids);

    ankerl::nanobench::Bench bench;

    bench.title(
        "EVE Static-D Streamed/Tiled K-Means "
        + std::to_string(TUPLE_SIZE)
        + "D K_TILE="
        + std::to_string(KMEANS_K_TILE)
        + " (Lloyd Iterations)"
    )
        .unit("run")
        .warmup(1)
        .epochs(bench_epochs)
        .minEpochTime(min_epoch_time)
        .performanceCounters(false)
        .output(nullptr);

    centroids_storage<TUPLE_SIZE> final_dynamic_centroids;
    using Label = kmeans::default_label_t;
    aligned_label_vector<Label> final_assignments;
    int iterations_to_converge = 0;

    bench.run("kmeans_lloyd_staticD_streamed_tiled", [&] {
        // Crucial: copy the initial state for every epoch.
        centroids_storage<TUPLE_SIZE> current_centroids =
            dynamic_initial_centroids;

        auto centroid_assignments = k_means_tiled<TUPLE_SIZE, KMEANS_K_TILE, Label>(
            dynamic_points,
            current_centroids,
            iterations_to_converge
        );

        ankerl::nanobench::doNotOptimizeAway(current_centroids.row_major.data());
        ankerl::nanobench::doNotOptimizeAway(current_centroids.feature_major.data());
        ankerl::nanobench::doNotOptimizeAway(centroid_assignments.data());

        swap_centroids_storage(final_dynamic_centroids, current_centroids);
        final_assignments.swap(centroid_assignments);
    });

    std::ofstream bench_out(nanobench_out);
    bench.render(ankerl::nanobench::templates::pyperf(), bench_out);

    // Convert final centroids back to the original static representation so we
    // can reuse the existing verification/metrics writer unchanged.
    auto final_centroids = make_static_centroids(final_dynamic_centroids);

    write_lloyd_metrics(
        metrics_json_out,
        static_points,
        final_centroids,
        final_assignments,
        n_clusters,
        iterations_to_converge
    );

    return 0;
}