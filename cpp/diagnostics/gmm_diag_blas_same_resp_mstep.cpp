#include <cblas.h>

#include <algorithm>
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

inline std::size_t kd(std::size_t k, std::size_t d) {
    return k * D + d;
}

struct MStepResult {
    std::string name;
    std::vector<double> N_k_raw;
    std::vector<double> N_k;
    std::vector<double> weights;
    std::vector<double> sum_x;
    std::vector<double> sum_x2;
    std::vector<double> means;
    std::vector<double> avg_x2;
    std::vector<double> mean_sq;
    std::vector<double> covariances;
};

void write_json_array(std::ostream& out, const std::vector<double>& v) {
    out << '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out << ',';
        out << std::setprecision(17) << v[i];
    }
    out << ']';
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

    for (std::size_t k = 0; k < K; ++k) {
        r.weights[k] = r.N_k[k] / N_k_sum;
        for (std::size_t d = 0; d < D; ++d) {
            const std::size_t idx = kd(k, d);
            const double mean = r.sum_x[idx] / r.N_k[k];
            const double avg_x2 = r.sum_x2[idx] / r.N_k[k];
            const double mean_sq = mean * mean;
            r.means[idx] = mean;
            r.avg_x2[idx] = avg_x2;
            r.mean_sq[idx] = mean_sq;
            r.covariances[idx] = avg_x2 - mean_sq + reg_covar;
        }
    }

    return r;
}

std::vector<double> N_k_loop_float(const std::vector<float>& resp, std::size_t N, std::size_t K) {
    std::vector<float> N_k_f(K, 0.0f);
    for (std::size_t n = 0; n < N; ++n) {
        const float* resp_sample = resp.data() + n * K;
        for (std::size_t k = 0; k < K; ++k) {
            N_k_f[k] += resp_sample[k];
        }
    }
    std::vector<double> N_k(K);
    for (std::size_t k = 0; k < K; ++k) N_k[k] = static_cast<double>(N_k_f[k]);
    return N_k;
}

std::vector<double> N_k_loop_double(const std::vector<float>& resp, std::size_t N, std::size_t K) {
    std::vector<double> N_k(K, 0.0);
    for (std::size_t n = 0; n < N; ++n) {
        const float* resp_sample = resp.data() + n * K;
        for (std::size_t k = 0; k < K; ++k) {
            N_k[k] += static_cast<double>(resp_sample[k]);
        }
    }
    return N_k;
}

std::vector<double> N_k_blas_sgemv(const std::vector<float>& resp, std::size_t N, std::size_t K) {
    std::vector<float> ones(N, 1.0f);
    std::vector<float> N_k_f(K, 0.0f);

    cblas_sgemv(
        CblasRowMajor,
        CblasTrans,
        static_cast<int>(N),
        static_cast<int>(K),
        1.0f,
        resp.data(),
        static_cast<int>(K),
        ones.data(),
        1,
        0.0f,
        N_k_f.data(),
        1
    );

    std::vector<double> N_k(K);
    for (std::size_t k = 0; k < K; ++k) N_k[k] = static_cast<double>(N_k_f[k]);
    return N_k;
}

std::vector<double> N_k_blas_sgemm_ones(const std::vector<float>& resp, std::size_t N, std::size_t K) {
    std::vector<float> ones(N, 1.0f);
    std::vector<float> N_k_f(K, 0.0f);

    // N_k = resp.T @ ones, reducing sample rows into one value per cluster.
    cblas_sgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        static_cast<int>(K),
        1,
        static_cast<int>(N),
        1.0f,
        resp.data(),
        static_cast<int>(K),
        ones.data(),
        1,
        0.0f,
        N_k_f.data(),
        1
    );

    std::vector<double> N_k(K);
    for (std::size_t k = 0; k < K; ++k) N_k[k] = static_cast<double>(N_k_f[k]);
    return N_k;
}

std::vector<double> to_double(const std::vector<float>& v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v[i]);
    return out;
}

MStepResult mstep_cblas_sgemm(
    const std::string& name,
    const std::vector<float>& x,
    const std::vector<float>& resp,
    std::size_t N,
    std::size_t K,
    double reg_covar,
    const std::string& N_k_mode
) {
    std::vector<float> x2(N * D);
    for (std::size_t i = 0; i < N * D; ++i) x2[i] = x[i] * x[i];

    std::vector<float> sum_x_f(K * D, 0.0f);
    std::vector<float> sum_x2_f(K * D, 0.0f);

    // sum_x = resp.T @ X; X contributes D columns, both matrices share N sample rows, and resp contributes K cluster columns.
    cblas_sgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        static_cast<int>(K),
        static_cast<int>(D),
        static_cast<int>(N),
        1.0f,
        resp.data(),
        static_cast<int>(K),
        x.data(),
        static_cast<int>(D),
        0.0f,
        sum_x_f.data(),
        static_cast<int>(D)
    );

    // sum_x2 = resp.T @ (X * X).
    cblas_sgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        static_cast<int>(K),
        static_cast<int>(D),
        static_cast<int>(N),
        1.0f,
        resp.data(),
        static_cast<int>(K),
        x2.data(),
        static_cast<int>(D),
        0.0f,
        sum_x2_f.data(),
        static_cast<int>(D)
    );

    std::vector<double> N_k;
    if (N_k_mode == "loop_float") {
        N_k = N_k_loop_float(resp, N, K);
    } else if (N_k_mode == "loop_double") {
        N_k = N_k_loop_double(resp, N, K);
    } else if (N_k_mode == "sgemv") {
        N_k = N_k_blas_sgemv(resp, N, K);
    } else if (N_k_mode == "sgemm_ones") {
        N_k = N_k_blas_sgemm_ones(resp, N, K);
    } else {
        throw std::runtime_error("unknown N_k_mode " + N_k_mode);
    }

    return finalize_result(name, std::move(N_k), to_double(sum_x_f), to_double(sum_x2_f), N, K, reg_covar);
}

MStepResult mstep_cblas_dgemm(
    const std::string& name,
    const std::vector<float>& x_f,
    const std::vector<float>& resp_f,
    std::size_t N,
    std::size_t K,
    double reg_covar
) {
    std::vector<double> x(N * D);
    std::vector<double> x2(N * D);
    std::vector<double> resp(N * K);

    for (std::size_t i = 0; i < N * D; ++i) {
        x[i] = static_cast<double>(x_f[i]);
        x2[i] = x[i] * x[i];
    }
    for (std::size_t i = 0; i < N * K; ++i) {
        resp[i] = static_cast<double>(resp_f[i]);
    }

    std::vector<double> sum_x(K * D, 0.0);
    std::vector<double> sum_x2(K * D, 0.0);
    std::vector<double> N_k(K, 0.0);
    std::vector<double> ones(N, 1.0);

    cblas_dgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        static_cast<int>(K),
        static_cast<int>(D),
        static_cast<int>(N),
        1.0,
        resp.data(),
        static_cast<int>(K),
        x.data(),
        static_cast<int>(D),
        0.0,
        sum_x.data(),
        static_cast<int>(D)
    );

    cblas_dgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        static_cast<int>(K),
        static_cast<int>(D),
        static_cast<int>(N),
        1.0,
        resp.data(),
        static_cast<int>(K),
        x2.data(),
        static_cast<int>(D),
        0.0,
        sum_x2.data(),
        static_cast<int>(D)
    );

    cblas_dgemv(
        CblasRowMajor,
        CblasTrans,
        static_cast<int>(N),
        static_cast<int>(K),
        1.0,
        resp.data(),
        static_cast<int>(K),
        ones.data(),
        1,
        0.0,
        N_k.data(),
        1
    );

    return finalize_result(name, std::move(N_k), std::move(sum_x), std::move(sum_x2), N, K, reg_covar);
}

void write_output(
    const std::string& output_path,
    std::size_t N,
    std::size_t K,
    double reg_covar,
    const std::vector<MStepResult>& results
) {
    std::ofstream out(output_path);
    if (!out) throw std::runtime_error("could not open output " + output_path);

    out << std::setprecision(17);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"algorithm\": \"gmm_diag_blas_same_resp_mstep\",\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"D\": " << D << ",\n";
    out << "  \"N\": " << N << ",\n";
    out << "  \"K\": " << K << ",\n";
    out << "  \"reg_covar\": " << reg_covar << ",\n";
    out << "  \"variants\": {\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    \"" << r.name << "\": {\n";
        out << "      \"N_k_raw\": "; write_json_array(out, r.N_k_raw); out << ",\n";
        out << "      \"N_k\": "; write_json_array(out, r.N_k); out << ",\n";
        out << "      \"weights\": "; write_json_array(out, r.weights); out << ",\n";
        out << "      \"sum_x\": "; write_json_array(out, r.sum_x); out << ",\n";
        out << "      \"sum_x2\": "; write_json_array(out, r.sum_x2); out << ",\n";
        out << "      \"means\": "; write_json_array(out, r.means); out << ",\n";
        out << "      \"avg_x2\": "; write_json_array(out, r.avg_x2); out << ",\n";
        out << "      \"mean_sq\": "; write_json_array(out, r.mean_sq); out << ",\n";
        out << "      \"covariances\": "; write_json_array(out, r.covariances); out << "\n";
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }

    out << "  }\n";
    out << "}\n";
}

} // namespace diag

int main(int argc, char** argv) {
    try {
        if (argc != 7) {
            std::cerr
                << "usage: " << argv[0]
                << " <data.bin> <N> <K> <resp.bin> <output.json> <reg_covar>\n";
            return 2;
        }

        const std::string data_path = argv[1];
        const std::size_t N = static_cast<std::size_t>(std::stoull(argv[2]));
        const std::size_t K = static_cast<std::size_t>(std::stoull(argv[3]));
        const std::string resp_path = argv[4];
        const std::string output_path = argv[5];
        const double reg_covar = std::stod(argv[6]);

        const auto x = diag::read_f32_file(data_path, N * diag::D);
        const auto resp = diag::read_f32_file(resp_path, N * K);

        std::vector<diag::MStepResult> results;
        results.push_back(diag::mstep_cblas_sgemm("cpp_cblas_sgemm_N_k_loop_float", x, resp, N, K, reg_covar, "loop_float"));
        results.push_back(diag::mstep_cblas_sgemm("cpp_cblas_sgemm_N_k_sgemv", x, resp, N, K, reg_covar, "sgemv"));
        results.push_back(diag::mstep_cblas_sgemm("cpp_cblas_sgemm_N_k_sgemm_ones", x, resp, N, K, reg_covar, "sgemm_ones"));
        results.push_back(diag::mstep_cblas_sgemm("cpp_cblas_sgemm_N_k_loop_double", x, resp, N, K, reg_covar, "loop_double"));
        results.push_back(diag::mstep_cblas_dgemm("cpp_cblas_dgemm", x, resp, N, K, reg_covar));

        diag::write_output(output_path, N, K, reg_covar, results);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
