#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef TUPLE_SIZE
#define TUPLE_SIZE 2
#endif

namespace diag {

constexpr std::size_t D = static_cast<std::size_t>(TUPLE_SIZE);

std::vector<float> read_f32_file(const std::string& path, std::size_t expected_count) {
    std::vector<float> out(expected_count);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("could not open " + path);
    }
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error("could not read expected float32 count from " + path);
    }
    return out;
}

struct MStepResult {
    std::string name;
    std::vector<double> N_k_raw;
    std::vector<double> N_k;
    std::vector<double> weights;
    std::vector<double> sum_x;       // K * D
    std::vector<double> sum_x2;      // K * D
    std::vector<double> means;       // K * D
    std::vector<double> avg_x2;      // K * D
    std::vector<double> mean_sq;     // K * D
    std::vector<double> covariances; // K * D
};

inline std::size_t kd(std::size_t k, std::size_t d) {
    return k * D + d;
}

MStepResult finalize_result(
    std::string name,
    std::vector<double> N_k_raw,
    std::vector<double> sum_x,
    std::vector<double> sum_x2,
    std::size_t N,
    std::size_t K,
    double reg_covar
) {
    const double eps10 = 10.0 * static_cast<double>(std::numeric_limits<float>::epsilon());

    MStepResult r;
    r.name = std::move(name);
    r.N_k_raw = std::move(N_k_raw);
    r.sum_x = std::move(sum_x);
    r.sum_x2 = std::move(sum_x2);
    r.N_k.resize(K);
    r.weights.resize(K);
    r.means.resize(K * D);
    r.avg_x2.resize(K * D);
    r.mean_sq.resize(K * D);
    r.covariances.resize(K * D);

    double N_k_sum = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        r.N_k[k] = r.N_k_raw[k] + eps10;
        N_k_sum += r.N_k[k];
    }

    // Match scikit/C++ behavior: weights are normalized by the summed N_k, not by N directly.
    (void)N;
    for (std::size_t k = 0; k < K; ++k) {
        r.weights[k] = r.N_k[k] / N_k_sum;
        const double inv_N_k = 1.0 / r.N_k[k];
        for (std::size_t d = 0; d < D; ++d) {
            const std::size_t o = kd(k, d);
            r.means[o] = r.sum_x[o] * inv_N_k;
            r.avg_x2[o] = r.sum_x2[o] * inv_N_k;
            r.mean_sq[o] = r.means[o] * r.means[o];
            r.covariances[o] = r.avg_x2[o] - r.mean_sq[o] + reg_covar;
        }
    }

    return r;
}

MStepResult compute_sample_major_float_mul_add(
    const std::vector<float>& x,
    const std::vector<float>& resp,
    std::size_t N,
    std::size_t K,
    double reg_covar
) {
    std::vector<float> N_k(K, 0.0f);
    std::vector<float> sum_x(K * D, 0.0f);
    std::vector<float> sum_x2(K * D, 0.0f);

    for (std::size_t n = 0; n < N; ++n) {
        std::array<float, D> xv{};
        std::array<float, D> x2v{};
        for (std::size_t d = 0; d < D; ++d) {
            xv[d] = x[n * D + d];
            x2v[d] = xv[d] * xv[d];
        }

        const float* resp_row = resp.data() + n * K;
        for (std::size_t k = 0; k < K; ++k) {
            const float r = resp_row[k];
            N_k[k] += r;
            for (std::size_t d = 0; d < D; ++d) {
                const std::size_t o = kd(k, d);
                sum_x[o] += r * xv[d];
                sum_x2[o] += r * x2v[d];
            }
        }
    }

    std::vector<double> N_k_d(K);
    std::vector<double> sx_d(K * D);
    std::vector<double> sx2_d(K * D);
    for (std::size_t k = 0; k < K; ++k) N_k_d[k] = N_k[k];
    for (std::size_t i = 0; i < K * D; ++i) {
        sx_d[i] = sum_x[i];
        sx2_d[i] = sum_x2[i];
    }

    return finalize_result("cpp_sample_major_float_mul_add", std::move(N_k_d), std::move(sx_d), std::move(sx2_d), N, K, reg_covar);
}

MStepResult compute_sample_major_float_fma(
    const std::vector<float>& x,
    const std::vector<float>& resp,
    std::size_t N,
    std::size_t K,
    double reg_covar
) {
    std::vector<float> N_k(K, 0.0f);
    std::vector<float> sum_x(K * D, 0.0f);
    std::vector<float> sum_x2(K * D, 0.0f);

    for (std::size_t n = 0; n < N; ++n) {
        std::array<float, D> xv{};
        std::array<float, D> x2v{};
        for (std::size_t d = 0; d < D; ++d) {
            xv[d] = x[n * D + d];
            x2v[d] = xv[d] * xv[d];
        }

        const float* resp_row = resp.data() + n * K;
        for (std::size_t k = 0; k < K; ++k) {
            const float r = resp_row[k];
            N_k[k] = std::fma(r, 1.0f, N_k[k]);
            for (std::size_t d = 0; d < D; ++d) {
                const std::size_t o = kd(k, d);
                sum_x[o] = std::fma(r, xv[d], sum_x[o]);
                sum_x2[o] = std::fma(r, x2v[d], sum_x2[o]);
            }
        }
    }

    std::vector<double> N_k_d(K);
    std::vector<double> sx_d(K * D);
    std::vector<double> sx2_d(K * D);
    for (std::size_t k = 0; k < K; ++k) N_k_d[k] = N_k[k];
    for (std::size_t i = 0; i < K * D; ++i) {
        sx_d[i] = sum_x[i];
        sx2_d[i] = sum_x2[i];
    }

    return finalize_result("cpp_sample_major_float_fma", std::move(N_k_d), std::move(sx_d), std::move(sx2_d), N, K, reg_covar);
}

template <std::size_t W>
MStepResult compute_lane_bucket_float_fma(
    const std::vector<float>& x,
    const std::vector<float>& resp,
    std::size_t N,
    std::size_t K,
    double reg_covar
) {
    std::vector<float> N_k_lanes(K * W, 0.0f);
    std::vector<float> sum_x_lanes(K * D * W, 0.0f);
    std::vector<float> sum_x2_lanes(K * D * W, 0.0f);

    auto lidx_k = [](std::size_t k, std::size_t lane) {
        return k * W + lane;
    };
    auto lidx_kd = [K](std::size_t k, std::size_t d, std::size_t lane) {
        (void)K;
        return (k * D + d) * W + lane;
    };

    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t lane = n % W;
        std::array<float, D> xv{};
        std::array<float, D> x2v{};
        for (std::size_t d = 0; d < D; ++d) {
            xv[d] = x[n * D + d];
            x2v[d] = xv[d] * xv[d];
        }

        const float* resp_row = resp.data() + n * K;
        for (std::size_t k = 0; k < K; ++k) {
            const float r = resp_row[k];
            N_k_lanes[lidx_k(k, lane)] = std::fma(r, 1.0f, N_k_lanes[lidx_k(k, lane)]);
            for (std::size_t d = 0; d < D; ++d) {
                const std::size_t o = lidx_kd(k, d, lane);
                sum_x_lanes[o] = std::fma(r, xv[d], sum_x_lanes[o]);
                sum_x2_lanes[o] = std::fma(r, x2v[d], sum_x2_lanes[o]);
            }
        }
    }

    std::vector<float> N_k(K, 0.0f);
    std::vector<float> sum_x(K * D, 0.0f);
    std::vector<float> sum_x2(K * D, 0.0f);

    // Linear lane reduction approximates eve::reduce enough to expose the lane-bucket accumulation effect.
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t lane = 0; lane < W; ++lane) {
            N_k[k] += N_k_lanes[lidx_k(k, lane)];
        }
        for (std::size_t d = 0; d < D; ++d) {
            const std::size_t o = kd(k, d);
            for (std::size_t lane = 0; lane < W; ++lane) {
                sum_x[o] += sum_x_lanes[lidx_kd(k, d, lane)];
                sum_x2[o] += sum_x2_lanes[lidx_kd(k, d, lane)];
            }
        }
    }

    std::vector<double> N_k_d(K);
    std::vector<double> sx_d(K * D);
    std::vector<double> sx2_d(K * D);
    for (std::size_t k = 0; k < K; ++k) N_k_d[k] = N_k[k];
    for (std::size_t i = 0; i < K * D; ++i) {
        sx_d[i] = sum_x[i];
        sx2_d[i] = sum_x2[i];
    }

    std::ostringstream name;
    name << "cpp_lane_bucket_float_fma_w" << W;
    return finalize_result(name.str(), std::move(N_k_d), std::move(sx_d), std::move(sx2_d), N, K, reg_covar);
}

MStepResult compute_cluster_major_float_fma(
    const std::vector<float>& x,
    const std::vector<float>& resp,
    std::size_t N,
    std::size_t K,
    double reg_covar
) {
    std::vector<float> N_k(K, 0.0f);
    std::vector<float> sum_x(K * D, 0.0f);
    std::vector<float> sum_x2(K * D, 0.0f);

    for (std::size_t k = 0; k < K; ++k) {
        float N_k_value = 0.0f;
        std::array<float, D> sx{};
        std::array<float, D> sx2{};

        for (std::size_t n = 0; n < N; ++n) {
            const float r = resp[n * K + k];
            N_k_value = std::fma(r, 1.0f, N_k_value);
            for (std::size_t d = 0; d < D; ++d) {
                const float xv = x[n * D + d];
                const float x2v = xv * xv;
                sx[d] = std::fma(r, xv, sx[d]);
                sx2[d] = std::fma(r, x2v, sx2[d]);
            }
        }

        N_k[k] = N_k_value;
        for (std::size_t d = 0; d < D; ++d) {
            sum_x[kd(k, d)] = sx[d];
            sum_x2[kd(k, d)] = sx2[d];
        }
    }

    std::vector<double> N_k_d(K);
    std::vector<double> sx_d(K * D);
    std::vector<double> sx2_d(K * D);
    for (std::size_t k = 0; k < K; ++k) N_k_d[k] = N_k[k];
    for (std::size_t i = 0; i < K * D; ++i) {
        sx_d[i] = sum_x[i];
        sx2_d[i] = sum_x2[i];
    }

    return finalize_result("cpp_cluster_major_float_fma", std::move(N_k_d), std::move(sx_d), std::move(sx2_d), N, K, reg_covar);
}

MStepResult compute_cluster_major_double(
    const std::vector<float>& x,
    const std::vector<float>& resp,
    std::size_t N,
    std::size_t K,
    double reg_covar
) {
    std::vector<double> N_k(K, 0.0);
    std::vector<double> sum_x(K * D, 0.0);
    std::vector<double> sum_x2(K * D, 0.0);

    for (std::size_t k = 0; k < K; ++k) {
        double N_k_value = 0.0;
        std::array<double, D> sx{};
        std::array<double, D> sx2{};

        for (std::size_t n = 0; n < N; ++n) {
            const double r = static_cast<double>(resp[n * K + k]);
            N_k_value += r;
            for (std::size_t d = 0; d < D; ++d) {
                const double xv = static_cast<double>(x[n * D + d]);
                sx[d] += r * xv;
                sx2[d] += r * xv * xv;
            }
        }

        N_k[k] = N_k_value;
        for (std::size_t d = 0; d < D; ++d) {
            sum_x[kd(k, d)] = sx[d];
            sum_x2[kd(k, d)] = sx2[d];
        }
    }

    return finalize_result("cpp_cluster_major_double", std::move(N_k), std::move(sum_x), std::move(sum_x2), N, K, reg_covar);
}

MStepResult compute_block1024_float_fma(
    const std::vector<float>& x,
    const std::vector<float>& resp,
    std::size_t N,
    std::size_t K,
    double reg_covar
) {
    constexpr std::size_t block_size = 1024;
    const std::size_t sample_block_count = (N + block_size - 1) / block_size;

    std::vector<float> N_k_blocks(sample_block_count * K, 0.0f);
    std::vector<float> sx_blocks(sample_block_count * K * D, 0.0f);
    std::vector<float> sx2_blocks(sample_block_count * K * D, 0.0f);

    for (std::size_t b = 0; b < sample_block_count; ++b) {
        const std::size_t begin = b * block_size;
        const std::size_t end = std::min(N, begin + block_size);
        for (std::size_t n = begin; n < end; ++n) {
            std::array<float, D> xv{};
            std::array<float, D> x2v{};
            for (std::size_t d = 0; d < D; ++d) {
                xv[d] = x[n * D + d];
                x2v[d] = xv[d] * xv[d];
            }

            const float* resp_row = resp.data() + n * K;
            for (std::size_t k = 0; k < K; ++k) {
                const float r = resp_row[k];
                N_k_blocks[b * K + k] = std::fma(r, 1.0f, N_k_blocks[b * K + k]);
                for (std::size_t d = 0; d < D; ++d) {
                    const std::size_t o = b * K * D + kd(k, d);
                    sx_blocks[o] = std::fma(r, xv[d], sx_blocks[o]);
                    sx2_blocks[o] = std::fma(r, x2v[d], sx2_blocks[o]);
                }
            }
        }
    }

    std::vector<float> N_k(K, 0.0f);
    std::vector<float> sum_x(K * D, 0.0f);
    std::vector<float> sum_x2(K * D, 0.0f);

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t b = 0; b < sample_block_count; ++b) {
            N_k[k] += N_k_blocks[b * K + k];
        }
        for (std::size_t d = 0; d < D; ++d) {
            const std::size_t o = kd(k, d);
            for (std::size_t b = 0; b < sample_block_count; ++b) {
                sum_x[o] += sx_blocks[b * K * D + o];
                sum_x2[o] += sx2_blocks[b * K * D + o];
            }
        }
    }

    std::vector<double> N_k_d(K);
    std::vector<double> sx_d(K * D);
    std::vector<double> sx2_d(K * D);
    for (std::size_t k = 0; k < K; ++k) N_k_d[k] = N_k[k];
    for (std::size_t i = 0; i < K * D; ++i) {
        sx_d[i] = sum_x[i];
        sx2_d[i] = sum_x2[i];
    }

    return finalize_result("cpp_block1024_float_fma", std::move(N_k_d), std::move(sx_d), std::move(sx2_d), N, K, reg_covar);
}

void write_1d(std::ostream& os, const std::vector<double>& v) {
    os << '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) os << ',';
        os << std::setprecision(17) << v[i];
    }
    os << ']';
}

void write_2d_kd(std::ostream& os, const std::vector<double>& v, std::size_t K) {
    os << '[';
    for (std::size_t k = 0; k < K; ++k) {
        if (k) os << ',';
        os << '[';
        for (std::size_t d = 0; d < D; ++d) {
            if (d) os << ',';
            os << std::setprecision(17) << v[kd(k, d)];
        }
        os << ']';
    }
    os << ']';
}

void write_result(std::ostream& os, const MStepResult& r, std::size_t K) {
    os << "{\n";
    os << "  \"N_k_raw\": "; write_1d(os, r.N_k_raw); os << ",\n";
    os << "  \"N_k\": "; write_1d(os, r.N_k); os << ",\n";
    os << "  \"weights\": "; write_1d(os, r.weights); os << ",\n";
    os << "  \"sum_x\": "; write_2d_kd(os, r.sum_x, K); os << ",\n";
    os << "  \"sum_x2\": "; write_2d_kd(os, r.sum_x2, K); os << ",\n";
    os << "  \"means\": "; write_2d_kd(os, r.means, K); os << ",\n";
    os << "  \"avg_x2\": "; write_2d_kd(os, r.avg_x2, K); os << ",\n";
    os << "  \"mean_sq\": "; write_2d_kd(os, r.mean_sq, K); os << ",\n";
    os << "  \"covariances\": "; write_2d_kd(os, r.covariances, K); os << "\n";
    os << "}";
}

void write_json(
    const std::string& output_path,
    const std::vector<MStepResult>& results,
    std::size_t N,
    std::size_t K,
    double reg_covar
) {
    std::ofstream os(output_path);
    if (!os) {
        throw std::runtime_error("could not open output json " + output_path);
    }

    os << "{\n";
    os << "  \"schema_version\": 1,\n";
    os << "  \"algorithm\": \"same_resp_diag_mstep\",\n";
    os << "  \"language\": \"cpp\",\n";
    os << "  \"D\": " << D << ",\n";
    os << "  \"N\": " << N << ",\n";
    os << "  \"K\": " << K << ",\n";
    os << "  \"reg_covar\": " << std::setprecision(17) << reg_covar << ",\n";
    os << "  \"eps10_float32\": " << std::setprecision(17) << 10.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) << ",\n";
    os << "  \"variants\": {\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i) os << ",\n";
        os << "    \"" << results[i].name << "\": ";
        write_result(os, results[i], K);
    }
    os << "\n  }\n";
    os << "}\n";
}

} // namespace diag

int main(int argc, char** argv) {
    try {
        if (argc < 6 || argc > 7) {
            std::cerr
                << "usage: " << argv[0]
                << " <data_f32_aos.bin> <N> <K> <resp_f32_row_major.bin> <out.json> [reg_covar]\n";
            return 2;
        }

        const std::string data_path = argv[1];
        const std::size_t N = static_cast<std::size_t>(std::stoull(argv[2]));
        const std::size_t K = static_cast<std::size_t>(std::stoull(argv[3]));
        const std::string resp_path = argv[4];
        const std::string out_path = argv[5];
        const double reg_covar = argc >= 7 ? std::stod(argv[6]) : 1e-6;

        const auto x = diag::read_f32_file(data_path, N * diag::D);
        const auto resp = diag::read_f32_file(resp_path, N * K);

        std::vector<diag::MStepResult> results;
        results.reserve(9);
        results.push_back(diag::compute_sample_major_float_mul_add(x, resp, N, K, reg_covar));
        results.push_back(diag::compute_sample_major_float_fma(x, resp, N, K, reg_covar));
        results.push_back(diag::compute_lane_bucket_float_fma<4>(x, resp, N, K, reg_covar));
        results.push_back(diag::compute_lane_bucket_float_fma<8>(x, resp, N, K, reg_covar));
        results.push_back(diag::compute_lane_bucket_float_fma<16>(x, resp, N, K, reg_covar));
        results.push_back(diag::compute_lane_bucket_float_fma<32>(x, resp, N, K, reg_covar));
        results.push_back(diag::compute_cluster_major_float_fma(x, resp, N, K, reg_covar));
        results.push_back(diag::compute_block1024_float_fma(x, resp, N, K, reg_covar));
        results.push_back(diag::compute_cluster_major_double(x, resp, N, K, reg_covar));

        diag::write_json(out_path, results, N, K, reg_covar);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
