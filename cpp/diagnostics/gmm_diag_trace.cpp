#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/gmm/static_d/diagonal_covariance.hpp"
#include "../include/gmm/static_d/em.hpp"
#include "../include/gmm/static_d/input.hpp"
#include "../include/io/binary.hpp"
#include "../include/io/json.hpp"

namespace {

struct args_t {
    std::string dataset_bin;
    std::size_t N = 0;
    std::size_t K = 0;
    std::string weights_bin;
    std::string means_bin;
    std::string precisions_bin;
    std::string output_json;
    int max_iter = 100;
    float tol = 1e-3f;
    float reg_covar = 1e-6f;
};

[[noreturn]] void usage(const char* program) {
    std::cerr
        << "Usage:\n  " << program
        << " <dataset_bin> <N> <K>"
        << " <gmm_weights_bin> <gmm_means_bin> <gmm_precisions_bin>"
        << " <output_json> [max_iter=100] [tol=1e-3] [reg_covar=1e-6]\n";
    std::exit(2);
}

args_t parse_args(int argc, char** argv) {
    if (argc < 8 || argc > 11) {
        usage(argv[0]);
    }

    args_t args;
    args.dataset_bin = argv[1];
    args.N = static_cast<std::size_t>(std::stoull(argv[2]));
    args.K = static_cast<std::size_t>(std::stoull(argv[3]));
    args.weights_bin = argv[4];
    args.means_bin = argv[5];
    args.precisions_bin = argv[6];
    args.output_json = argv[7];

    if (argc >= 9) {
        args.max_iter = std::stoi(argv[8]);
    }
    if (argc >= 10) {
        args.tol = std::stof(argv[9]);
    }
    if (argc >= 11) {
        args.reg_covar = std::stof(argv[10]);
    }

    if (args.N == 0 || args.K == 0 || args.max_iter < 0) {
        throw std::runtime_error("Invalid N, K, or max_iter");
    }

    return args;
}

void write_f64(std::ostream& out, double value) {
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
}

void write_scalar_array_json(std::ostream& out, const std::vector<float>& values) {
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        write_f64(out, static_cast<double>(values[i]));
    }
    out << "]";
}

template <eve::product_type SampleT>
void write_mean_vector_json(std::ostream& out, const std::vector<SampleT>& means) {
    out << "[\n";
    for (std::size_t k = 0; k < means.size(); ++k) {
        out << "        ";
        write_sample_json(out, means[k]);
        if (k + 1 != means.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "      ]";
}

template <eve::product_type SampleT>
void write_flat_diag_matrix_json(
    std::ostream& out,
    const std::vector<float>& values,
    std::size_t K
) {
    constexpr std::size_t D = kumi::size_v<SampleT>;
    out << "[\n";
    for (std::size_t k = 0; k < K; ++k) {
        out << "        [";
        for (std::size_t d = 0; d < D; ++d) {
            if (d != 0) {
                out << ", ";
            }
            write_f64(out, static_cast<double>(values[k * D + d]));
        }
        out << "]";
        if (k + 1 != K) {
            out << ",";
        }
        out << "\n";
    }
    out << "      ]";
}

void write_vector_key(
    std::ostream& out,
    const char* key,
    const std::vector<float>& values,
    const char* suffix
) {
    out << "      \"" << key << "\": ";
    write_scalar_array_json(out, values);
    out << suffix;
}

template <eve::product_type SampleT>
void write_matrix_key(
    std::ostream& out,
    const char* key,
    const std::vector<float>& values,
    std::size_t K,
    const char* suffix
) {
    out << "      \"" << key << "\": ";
    write_flat_diag_matrix_json<SampleT>(out, values, K);
    out << suffix;
}

template <eve::product_type SampleT, class StateT>
std::vector<float> reduced_sum_x(const StateT& state) {
    constexpr std::size_t D = kumi::size_v<SampleT>;
    std::vector<float> out(state.K() * D);

    for (std::size_t k = 0; k < state.K(); ++k) {
        for (std::size_t d = 0; d < D; ++d) {
            out[k * D + d] = eve::reduce(state.sum_x_w[k][d]);
        }
    }

    return out;
}

template <eve::product_type SampleT, class StateT>
std::vector<float> reduced_sum_x2(const StateT& state) {
    constexpr std::size_t D = kumi::size_v<SampleT>;
    std::vector<float> out(state.K() * D);

    for (std::size_t k = 0; k < state.K(); ++k) {
        for (std::size_t d = 0; d < D; ++d) {
            out[k * D + d] = eve::reduce(state.covariance.sum_x2_w[k][d]);
        }
    }

    return out;
}

template <class StateT>
std::vector<float> reduced_N_k_raw(const StateT& state) {
    std::vector<float> out(state.K());

    for (std::size_t k = 0; k < state.K(); ++k) {
        out[k] = eve::reduce(state.N_k_w[k]);
    }

    return out;
}

template <eve::product_type SampleT>
std::vector<float> means_to_flat(const std::vector<SampleT>& means) {
    constexpr std::size_t D = kumi::size_v<SampleT>;
    std::vector<float> out(means.size() * D);

    for (std::size_t k = 0; k < means.size(); ++k) {
        kumi::for_each_index(
            [&](auto index, auto value) {
                out[k * D + index] = static_cast<float>(value);
            },
            means[k]
        );
    }

    return out;
}

template <eve::product_type SampleT>
void write_trace_json(
    std::ostream& out,
    const args_t& args,
    const std::vector<float>& initial_weights,
    const std::vector<SampleT>& initial_means,
    const std::vector<float>& initial_precisions,
    const std::vector<std::string>& iteration_jsons,
    const std::vector<float>& lower_bounds,
    int iterations,
    bool converged,
    float lower_bound,
    const std::vector<float>& final_weights,
    const std::vector<SampleT>& final_means,
    const std::vector<float>& final_covariances,
    const std::vector<float>& final_precisions
) {
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"algorithm\": \"gmm_diag_sufficient_statistics_trace\",\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"D\": " << kumi::size_v<SampleType> << ",\n";
    out << "  \"N\": " << args.N << ",\n";
    out << "  \"K\": " << args.K << ",\n";
    out << "  \"tol\": "; write_f64(out, args.tol); out << ",\n";
    out << "  \"reg_covar\": "; write_f64(out, args.reg_covar); out << ",\n";
    out << "  \"max_iter\": " << args.max_iter << ",\n";

    out << "  \"initial\": {\n";
    write_vector_key(out, "weights", initial_weights, ",\n");
    out << "      \"means\": ";
    write_mean_vector_json(out, initial_means);
    out << ",\n";
    write_matrix_key<SampleT>(out, "precisions", initial_precisions, args.K, "\n");
    out << "  },\n";

    out << "  \"iterations\": [\n";
    for (std::size_t i = 0; i < iteration_jsons.size(); ++i) {
        out << iteration_jsons[i];
        if (i + 1 != iteration_jsons.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"final\": {\n";
    out << "    \"iterations\": " << iterations << ",\n";
    out << "    \"converged\": " << (converged ? "true" : "false") << ",\n";
    out << "    \"lower_bound\": "; write_f64(out, lower_bound); out << ",\n";
    out << "    \"lower_bounds\": "; write_scalar_array_json(out, lower_bounds); out << ",\n";
    out << "    \"weights\": "; write_scalar_array_json(out, final_weights); out << ",\n";
    out << "    \"means\": "; write_mean_vector_json(out, final_means); out << ",\n";
    out << "    \"covariances\": ";
    write_flat_diag_matrix_json<SampleT>(out, final_covariances, args.K);
    out << ",\n";
    out << "    \"precisions\": ";
    write_flat_diag_matrix_json<SampleT>(out, final_precisions, args.K);
    out << "\n";
    out << "  }\n";
    out << "}\n";
}

template <eve::product_type SampleT, class StateT>
std::string make_iteration_json(
    int iter,
    float lower_bound,
    const StateT& state,
    float eps10
) {
    constexpr std::size_t D = kumi::size_v<SampleT>;
    const std::size_t K = state.K();

    const auto N_k_raw = reduced_N_k_raw(state);
    const auto sum_x = reduced_sum_x<SampleT>(state);
    const auto sum_x2 = reduced_sum_x2<SampleT>(state);

    std::vector<float> N_k(K);
    std::vector<float> avg_x(K * D);
    std::vector<float> avg_x2(K * D);
    std::vector<float> mean_sq(K * D);
    std::vector<float> cov_from_stats(K * D);
    std::vector<float> weights_from_N_k(K);

    float N_k_sum = 0.0f;
    for (std::size_t k = 0; k < K; ++k) {
        N_k[k] = N_k_raw[k] + eps10;
        N_k_sum += N_k[k];
    }

    for (std::size_t k = 0; k < K; ++k) {
        const float inv_N_k = 1.0f / N_k[k];
        weights_from_N_k[k] = N_k[k] / N_k_sum;

        for (std::size_t d = 0; d < D; ++d) {
            const std::size_t idx = k * D + d;
            avg_x[idx] = sum_x[idx] * inv_N_k;
            avg_x2[idx] = sum_x2[idx] * inv_N_k;
            mean_sq[idx] = avg_x[idx] * avg_x[idx];
            cov_from_stats[idx] = avg_x2[idx] - mean_sq[idx] + state.covariance.reg_covar;
        }
    }

    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "    {\n";
    out << "      \"iter\": " << iter << ",\n";
    out << "      \"lower_bound\": "; write_f64(out, lower_bound); out << ",\n";
    write_vector_key(out, "weights_before_m_step", state.weights, ",\n");
    out << "      \"means_before_m_step\": "; write_mean_vector_json(out, state.means); out << ",\n";
    write_matrix_key<SampleT>(out, "covariances_before_m_step", state.covariance.covariances, K, ",\n");
    write_matrix_key<SampleT>(out, "precisions_before_m_step", state.covariance.precisions, K, ",\n");
    write_vector_key(out, "N_k_raw", N_k_raw, ",\n");
    write_vector_key(out, "N_k", N_k, ",\n");
    out << "      \"N_k_sum\": "; write_f64(out, N_k_sum); out << ",\n";
    write_vector_key(out, "weights_from_N_k", weights_from_N_k, ",\n");
    write_matrix_key<SampleT>(out, "sum_x", sum_x, K, ",\n");
    write_matrix_key<SampleT>(out, "sum_x2", sum_x2, K, ",\n");
    write_matrix_key<SampleT>(out, "avg_x", avg_x, K, ",\n");
    write_matrix_key<SampleT>(out, "avg_x2", avg_x2, K, ",\n");
    write_matrix_key<SampleT>(out, "mean_sq", mean_sq, K, ",\n");
    write_matrix_key<SampleT>(out, "cov_from_stats", cov_from_stats, K, "\n");
    out << "    }";
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const args_t args = parse_args(argc, argv);
        constexpr float eps10 = 10.0f * std::numeric_limits<float>::epsilon();

        auto samples = read_static_gmm_samples_binary(args.dataset_bin, args.N);
        auto weights = read_binary_f32(args.weights_bin, args.K);
        auto means = read_static_gmm_means_binary(
            args.means_bin,
            static_cast<int>(args.K)
        );
        auto precisions = read_binary_f32(
            args.precisions_bin,
            args.K * static_cast<std::size_t>(D)
        );

        const auto initial_weights = weights;
        const auto initial_means = means;
        const auto initial_precisions = precisions;

        static_gmm_em_state<SampleType, diagonal_covariance_model<SampleType>> state{
            samples,
            std::move(weights),
            std::move(means),
            diagonal_covariance_model<SampleType>{
                std::move(precisions),
                args.K,
                args.reg_covar
            }
        };

        std::vector<std::string> iteration_jsons;
        std::vector<float> lower_bounds;
        iteration_jsons.reserve(static_cast<std::size_t>(args.max_iter));
        lower_bounds.reserve(static_cast<std::size_t>(args.max_iter));

        float lower_bound = -std::numeric_limits<float>::infinity();
        bool converged = false;
        int iterations = 0;

        for (int iter = 1; iter <= args.max_iter; ++iter) {
            const float previous_lower_bound = lower_bound;

            lower_bound = state.e_step_and_accumulate_sufficient_statistics();
            iteration_jsons.push_back(
                make_iteration_json<SampleType>(iter, lower_bound, state, eps10)
            );

            state.m_step_from_accumulators();

            lower_bounds.push_back(lower_bound);
            iterations = iter;

            if (std::abs(lower_bound - previous_lower_bound) < args.tol) {
                converged = true;
                break;
            }
        }

        std::ofstream out(args.output_json);
        if (!out) {
            throw std::runtime_error("Could not open output JSON: " + args.output_json);
        }

        write_trace_json<SampleType>(
            out,
            args,
            initial_weights,
            initial_means,
            initial_precisions,
            iteration_jsons,
            lower_bounds,
            iterations,
            converged,
            lower_bound,
            state.weights,
            state.means,
            state.covariance.covariances,
            state.covariance.precisions
        );

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "gmm_diag_trace failed: " << exc.what() << "\n";
        return 1;
    }
}
