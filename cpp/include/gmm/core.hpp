#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace gmm {

struct em_trace {
    std::vector<float> lower_bounds;
    int algorithm_iterations = 0;
    float lower_bound = -std::numeric_limits<float>::infinity();
};

template<class State>
void m_step_from_accumulators_common(State& state) {
    constexpr float eps10 = 10.0f * std::numeric_limits<float>::epsilon();

    float N_k_sum = 0.0f;

    for (std::size_t k = 0; k < state.K(); ++k) {
        const float N_k = state.reduced_component_count(k) + eps10;
        state.set_component_weight(k, N_k);
        N_k_sum += N_k;
    }

    for (std::size_t k = 0; k < state.K(); ++k) {
        const float N_k = state.component_weight(k);
        const float inv_N_k = 1.0f / N_k;

        state.write_mean_from_accumulators(k, inv_N_k);
        state.update_covariance_from_sufficient_statistics(k, N_k);
    }

    // Full-covariance implementations may need to recompute selected covariance
    // statistics from responsibilities produced by the just-finished E-step.
    // Score data has intentionally not been refreshed yet, so it still represents
    // the parameters that generated those responsibilities.
    state.recompute_unstable_covariances_from_current_responsibilities();

    for (std::size_t k = 0; k < state.K(); ++k) {
        state.set_component_weight(k, state.component_weight(k) / N_k_sum);
    }

    state.refresh_score_data();
}

template<class State>
em_trace em_core(State& state, int max_iterations, float tol) {
    em_trace trace;
    trace.lower_bounds.reserve(static_cast<std::size_t>(max_iterations));

    float lower_bound = -std::numeric_limits<float>::infinity();

    for (int iter = 1; iter <= max_iterations; ++iter) {
        const float previous_lower_bound = lower_bound;

        lower_bound = state.e_step_and_accumulate_sufficient_statistics();
        m_step_from_accumulators_common(state);

        trace.lower_bounds.push_back(lower_bound);
        trace.algorithm_iterations = iter;
        trace.lower_bound = lower_bound;

        if (std::abs(lower_bound - previous_lower_bound) < tol) {
            break;
        }
    }

    return trace;
}

} // namespace gmm
