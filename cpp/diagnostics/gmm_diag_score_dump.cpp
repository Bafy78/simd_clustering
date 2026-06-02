
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/wide.hpp>

#include "../include/io/binary.hpp"
#include "../include/simd.hpp"

#ifndef TUPLE_SIZE
#define TUPLE_SIZE 2
#endif

namespace {

constexpr std::size_t D = static_cast<std::size_t>(TUPLE_SIZE);

struct Args {
    std::string dataset_bin;
    std::size_t N{};
    std::size_t K{};
    std::string weights_bin;
    std::string means_bin;
    std::string precisions_bin;
    std::string sample_indices_file;
    std::string output_json;
};

[[noreturn]] void usage(const char* program) {
    std::cerr
        << "Usage:\n  " << program
        << " <dataset_bin> <N> <K>"
        << " <gmm_weights_bin> <gmm_means_bin> <gmm_precisions_diag_bin>"
        << " <sample_indices_file> <output_json>\n";
    std::exit(2);
}

Args parse_args(int argc, char** argv) {
    if (argc != 9) usage(argv[0]);
    Args a;
    a.dataset_bin = argv[1];
    a.N = static_cast<std::size_t>(std::stoull(argv[2]));
    a.K = static_cast<std::size_t>(std::stoull(argv[3]));
    a.weights_bin = argv[4];
    a.means_bin = argv[5];
    a.precisions_bin = argv[6];
    a.sample_indices_file = argv[7];
    a.output_json = argv[8];
    if (a.N == 0 || a.K == 0) {
        throw std::runtime_error("N and K must be positive");
    }
    return a;
}

std::vector<std::size_t> read_sample_indices(const std::string& filename, std::size_t N) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Could not open sample indices file: " + filename);
    std::stringstream buf;
    buf << f.rdbuf();
    std::string text = buf.str();
    std::replace(text.begin(), text.end(), ',', ' ');
    std::stringstream ss(text);
    std::vector<std::size_t> sample_indices;
    std::size_t n{};
    while (ss >> n) {
        if (n >= N) throw std::runtime_error("Sample index out of range");
        sample_indices.push_back(n);
    }
    if (sample_indices.empty()) throw std::runtime_error("Sample indices file is empty");
    return sample_indices;
}

void write_f64(std::ostream& out, double value) {
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
}

template <class T>
void write_array(std::ostream& out, const std::vector<T>& xs) {
    out << "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i) out << ", ";
        if constexpr (std::is_floating_point_v<T>) write_f64(out, static_cast<double>(xs[i]));
        else out << xs[i];
    }
    out << "]";
}

void write_matrix(std::ostream& out, const std::vector<float>& xs, std::size_t rows, std::size_t cols) {
    out << "[\n";
    for (std::size_t r = 0; r < rows; ++r) {
        out << "      [";
        for (std::size_t c = 0; c < cols; ++c) {
            if (c) out << ", ";
            write_f64(out, xs[r * cols + c]);
        }
        out << "]";
        if (r + 1 != rows) out << ",";
        out << "\n";
    }
    out << "    ]";
}

// Diagnostic-only rounded scalar ops. This makes scalar_split less likely to be
// optimized back into a contracted expression.
[[gnu::noinline]] float rounded_add(float a, float b) { volatile float r = a + b; return r; }
[[gnu::noinline]] float rounded_mul(float a, float b) { volatile float r = a * b; return r; }
[[gnu::noinline]] float rounded_log(float a) { volatile float aa = a; volatile float r = std::log(aa); return r; }

struct Precomputed {
    std::vector<float> log_weights;
    std::vector<float> log_precision_dets;
    std::vector<float> mean_quadratics;
    std::vector<float> current_constants;
};

Precomputed precompute(
    const std::vector<float>& weights,
    const std::vector<float>& means,
    const std::vector<float>& precisions,
    std::size_t K
) {
    const float log_2_pi = std::log(2.0f * std::numbers::pi_v<float>);
    const float half_D = 0.5f * static_cast<float>(D);
    Precomputed p;
    p.log_weights.resize(K);
    p.log_precision_dets.resize(K);
    p.mean_quadratics.resize(K);
    p.current_constants.resize(K);

    for (std::size_t k = 0; k < K; ++k) {
        float logdet = 0.0f;
        float meanq = 0.0f;
        for (std::size_t d = 0; d < D; ++d) {
            const float mu = means[k * D + d];
            const float prec = precisions[k * D + d];
            logdet += std::log(prec);
            meanq += mu * mu * prec;
        }
        p.log_weights[k] = std::log(weights[k]);
        p.log_precision_dets[k] = logdet;
        p.mean_quadratics[k] = meanq;
        p.current_constants[k] =
            p.log_weights[k] + 0.5f * logdet - half_D * log_2_pi - 0.5f * meanq;
    }
    return p;
}

float score_scalar_fma(const float* x, const float* mean, const float* prec, float constant) {
    float score = constant;
    for (std::size_t d = 0; d < D; ++d) {
        const float x2 = x[d] * x[d];
        const float mu_p = mean[d] * prec[d];
        score = std::fma(-0.5f * prec[d], x2, std::fma(mu_p, x[d], score));
    }
    return score;
}

float score_scalar_split(const float* x, const float* mean, const float* prec, float constant) {
    float score = constant;
    for (std::size_t d = 0; d < D; ++d) {
        const float x2 = rounded_mul(x[d], x[d]);
        const float mu_p = rounded_mul(mean[d], prec[d]);
        score = rounded_add(score, rounded_mul(mu_p, x[d]));
        score = rounded_add(score, rounded_mul(rounded_mul(-0.5f, prec[d]), x2));
    }
    return score;
}

float score_grouped_float(const float* x, const float* mean, const float* prec, float log_weight) {
    const float log_2_pi = rounded_log(2.0f * std::numbers::pi_v<float>);
    float logdet = 0.0f;
    float meanq = 0.0f;
    float cross = 0.0f;
    float x2p = 0.0f;
    for (std::size_t d = 0; d < D; ++d) {
        const float x2 = rounded_mul(x[d], x[d]);
        const float mu2 = rounded_mul(mean[d], mean[d]);
        const float mu_p = rounded_mul(mean[d], prec[d]);
        logdet = rounded_add(logdet, rounded_log(prec[d]));
        meanq = rounded_add(meanq, rounded_mul(mu2, prec[d]));
        cross = rounded_add(cross, rounded_mul(x[d], mu_p));
        x2p = rounded_add(x2p, rounded_mul(x2, prec[d]));
    }
    float maha = rounded_add(meanq, rounded_mul(-2.0f, cross));
    maha = rounded_add(maha, x2p);
    float s = log_weight;
    s = rounded_add(s, rounded_mul(0.5f, logdet));
    s = rounded_add(s, rounded_mul(-0.5f, rounded_mul(static_cast<float>(D), log_2_pi)));
    s = rounded_add(s, rounded_mul(-0.5f, maha));
    return s;
}

float score_double_grouped(const float* x, const float* mean, const float* prec, float weight) {
    const double log_2_pi = std::log(2.0 * std::numbers::pi_v<double>);
    double logdet = 0.0;
    double meanq = 0.0;
    double cross = 0.0;
    double x2p = 0.0;
    for (std::size_t d = 0; d < D; ++d) {
        const double xd = x[d];
        const double mu = mean[d];
        const double pr = prec[d];
        logdet += std::log(pr);
        meanq += mu * mu * pr;
        cross += xd * mu * pr;
        x2p += xd * xd * pr;
    }
    const double maha = meanq - 2.0 * cross + x2p;
    const double s = std::log(static_cast<double>(weight))
        + 0.5 * logdet
        - 0.5 * static_cast<double>(D) * log_2_pi
        - 0.5 * maha;
    return static_cast<float>(s);
}

void compute_eve_current(
    const std::vector<float>& X,
    const std::vector<float>& means,
    const std::vector<float>& precs,
    const std::vector<float>& constants,
    const std::vector<std::size_t>& sample_indices,
    std::size_t K,
    std::vector<float>& out
) {
    using cardinal_t = typename wide_f::cardinal_type;
    constexpr std::size_t lanes = cardinal_t::value;

    out.assign(sample_indices.size() * K, 0.0f);
    alignas(64) std::array<float, lanes> lane_values{};
    std::array<wide_f, D> sample{};
    std::array<wide_f, D> sample2{};

    for (std::size_t start = 0; start < sample_indices.size(); start += lanes) {
        const std::size_t active = std::min(lanes, sample_indices.size() - start);
        for (std::size_t d = 0; d < D; ++d) {
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                const std::size_t pos = start + std::min(lane, active - 1);
                lane_values[lane] = X[sample_indices[pos] * D + d];
            }
            sample[d] = eve::load(lane_values.data(), eve::as<wide_f>());
            sample2[d] = sample[d] * sample[d];
        }

        for (std::size_t k = 0; k < K; ++k) {
            auto score = wide_f(constants[k]);
            for (std::size_t d = 0; d < D; ++d) {
                const float mu = means[k * D + d];
                const float pr = precs[k * D + d];
                const float mu_p = mu * pr;
                score = eve::fma(
                    wide_f(-0.5f) * wide_f(pr),
                    sample2[d],
                    eve::fma(wide_f(mu_p), sample[d], score)
                );
            }
            eve::store(score, lane_values.data());
            for (std::size_t lane = 0; lane < active; ++lane) {
                out[(start + lane) * K + k] = lane_values[lane];
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args a = parse_args(argc, argv);
        using cardinal_t = typename wide_f::cardinal_type;

        const auto X = read_aos_f32(a.dataset_bin, a.N, D);
        const auto weights = read_binary_f32(a.weights_bin, a.K);
        const auto means = read_binary_f32(a.means_bin, a.K * D);
        const auto precs = read_binary_f32(a.precisions_bin, a.K * D);
        const auto sample_indices = read_sample_indices(a.sample_indices_file, a.N);
        const auto pc = precompute(weights, means, precs, a.K);

        std::vector<float> eve_current;
        compute_eve_current(X, means, precs, pc.current_constants, sample_indices, a.K, eve_current);

        std::vector<float> scalar_fma(sample_indices.size() * a.K);
        std::vector<float> scalar_split(sample_indices.size() * a.K);
        std::vector<float> grouped_float(sample_indices.size() * a.K);
        std::vector<float> grouped_double(sample_indices.size() * a.K);

        for (std::size_t selected_n = 0; selected_n < sample_indices.size(); ++selected_n) {
            const float* x = &X[sample_indices[selected_n] * D];
            for (std::size_t k = 0; k < a.K; ++k) {
                const float* mean = &means[k * D];
                const float* prec = &precs[k * D];
                const std::size_t pos = selected_n * a.K + k;
                scalar_fma[pos] = score_scalar_fma(x, mean, prec, pc.current_constants[k]);
                scalar_split[pos] = score_scalar_split(x, mean, prec, pc.current_constants[k]);
                grouped_float[pos] = score_grouped_float(x, mean, prec, pc.log_weights[k]);
                grouped_double[pos] = score_double_grouped(x, mean, prec, weights[k]);
            }
        }

        std::ofstream out(a.output_json);
        if (!out) throw std::runtime_error("Could not open output JSON: " + a.output_json);

        out << "{\n";
        out << "  \"schema_version\": 1,\n";
        out << "  \"phase\": \"gmm\",\n";
        out << "  \"diagnostic\": \"gmm_diag_weighted_log_prob_score_dump\",\n";
        out << "  \"language\": \"cpp\",\n";
        out << "  \"D\": " << D << ",\n";
        out << "  \"N\": " << a.N << ",\n";
        out << "  \"K\": " << a.K << ",\n";
        out << "  \"eve_cardinal\": " << cardinal_t::value << ",\n";
        out << "  \"sample_indices\": ";
        write_array(out, sample_indices);
        out << ",\n";
        out << "  \"score_constants_current\": ";
        write_array(out, pc.current_constants);
        out << ",\n";
        out << "  \"log_weights\": ";
        write_array(out, pc.log_weights);
        out << ",\n";
        out << "  \"log_precision_dets\": ";
        write_array(out, pc.log_precision_dets);
        out << ",\n";
        out << "  \"mean_quadratics\": ";
        write_array(out, pc.mean_quadratics);
        out << ",\n";
        out << "  \"scores\": {\n";
        out << "    \"eve_current\": ";
        write_matrix(out, eve_current, sample_indices.size(), a.K);
        out << ",\n";
        out << "    \"scalar_fma_current\": ";
        write_matrix(out, scalar_fma, sample_indices.size(), a.K);
        out << ",\n";
        out << "    \"scalar_split_current\": ";
        write_matrix(out, scalar_split, sample_indices.size(), a.K);
        out << ",\n";
        out << "    \"scalar_grouped_float\": ";
        write_matrix(out, grouped_float, sample_indices.size(), a.K);
        out << ",\n";
        out << "    \"scalar_double_grouped\": ";
        write_matrix(out, grouped_double, sample_indices.size(), a.K);
        out << "\n";
        out << "  }\n";
        out << "}\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gmm_diag_score_dump failed: " << e.what() << "\n";
        return 1;
    }
}
