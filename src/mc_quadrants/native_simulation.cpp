#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace {

class RandomStream {
  public:
    RandomStream(std::uint64_t seed, std::uint64_t stream) {
        std::uint64_t value = seed ^ (0x9e3779b97f4a7c15ULL * (stream + 1));
        for (auto &word : state_) word = splitmix64(value);
    }

    double uniform() {
        return static_cast<double>(next() >> 11) * 0x1.0p-53;
    }

    double normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        const double first = std::max(uniform(), std::numeric_limits<double>::min());
        const double second = uniform();
        const double radius = std::sqrt(-2.0 * std::log(first));
        const double angle = 6.283185307179586476925286766559 * second;
        spare_ = radius * std::sin(angle);
        has_spare_ = true;
        return radius * std::cos(angle);
    }

    double exponential() {
        return -std::log(std::max(uniform(), std::numeric_limits<double>::min()));
    }

  private:
    std::uint64_t state_[4]{};
    bool has_spare_ = false;
    double spare_ = 0.0;

    static std::uint64_t splitmix64(std::uint64_t &value) {
        std::uint64_t result = (value += 0x9e3779b97f4a7c15ULL);
        result = (result ^ (result >> 30)) * 0xbf58476d1ce4e5b9ULL;
        result = (result ^ (result >> 27)) * 0x94d049bb133111ebULL;
        return result ^ (result >> 31);
    }

    static std::uint64_t rotate_left(std::uint64_t value, int shift) {
        return (value << shift) | (value >> (64 - shift));
    }

    std::uint64_t next() {
        const std::uint64_t result = rotate_left(state_[1] * 5, 7) * 9;
        const std::uint64_t temporary = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= temporary;
        state_[3] = rotate_left(state_[3], 45);
        return result;
    }

};

inline const double *state_matrix(const double *values, int state, int assets) {
    return values + static_cast<std::size_t>(state) * assets * assets;
}

inline const double *state_vector(const double *values, int state, int assets) {
    return values + static_cast<std::size_t>(state) * assets;
}

void cholesky(const std::vector<double> &matrix, std::vector<double> &factor, int assets) {
    std::fill(factor.begin(), factor.end(), 0.0);
    for (int column = 0; column < assets; ++column) {
        double diagonal = matrix[column * assets + column];
        for (int previous = 0; previous < column; ++previous) {
            const double value = factor[column * assets + previous];
            diagonal -= value * value;
        }
        factor[column * assets + column] = std::sqrt(std::max(diagonal, 1e-10));
        for (int row = column + 1; row < assets; ++row) {
            double value = matrix[row * assets + column];
            for (int previous = 0; previous < column; ++previous) {
                value -= factor[row * assets + previous] * factor[column * assets + previous];
            }
            factor[row * assets + column] = value / factor[column * assets + column];
        }
    }
}

double sinc(double value) {
    if (std::abs(value) < 1e-6) {
        const double square = value * value;
        return 1.0 - square / 6.0 + square * square / 120.0;
    }
    return std::sin(value) / value;
}

double zolotarev_b_ratio(double value, double alpha) {
    const double complement = 1.0 - alpha;
    const double log_ratio = std::log(std::max(sinc(value), 1e-300))
        - alpha * std::log(std::max(sinc(alpha * value), 1e-300))
        - complement * std::log(std::max(sinc(complement * value), 1e-300));
    return std::exp(log_ratio);
}

double zolotarev_a(double value, double alpha, double b_ratio) {
    const double complement = 1.0 - alpha;
    const double log_b_zero = -alpha * std::log(alpha)
        - complement * std::log(complement);
    return std::exp(-(log_b_zero + std::log(std::max(b_ratio, 1e-300))) / complement);
}

// Devroye's uniformly fast double-rejection generator for the exponentially
// tilted unilateral stable law S_{alpha, lambda}. Its Laplace transform is
// exp(lambda^alpha - (lambda + s)^alpha). The implementation follows the
// full algorithm in ACM TOMACS 19(4), 2009, using sinc ratios near zero.
struct TiltedStableParameters {
    double alpha;
    double lambda;
    double complement;
    double lambda_alpha;
    double gamma_parameter;
    double sqrt_gamma;
    double xi;
    double psi;
    double w1;
    double w2;
    double w3;
    double b;
};

TiltedStableParameters make_tilted_stable_parameters(double alpha, double lambda) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double sqrt_pi_over_two = 1.253314137315500251207882642406;
    TiltedStableParameters parameters{};
    parameters.alpha = alpha;
    parameters.lambda = lambda;
    parameters.complement = 1.0 - alpha;
    parameters.lambda_alpha = std::pow(lambda, alpha);
    parameters.gamma_parameter = parameters.lambda_alpha * alpha * parameters.complement;
    parameters.sqrt_gamma = std::sqrt(parameters.gamma_parameter);
    parameters.xi = (2.0 + sqrt_pi_over_two)
        * std::sqrt(2.0 * parameters.gamma_parameter + 1.0) / pi;
    parameters.psi = std::exp(-parameters.gamma_parameter * pi * pi / 8.0)
        * (2.0 + sqrt_pi_over_two) * std::sqrt(parameters.gamma_parameter * pi) / pi;
    parameters.w1 = parameters.xi * std::sqrt(pi / (2.0 * parameters.gamma_parameter));
    parameters.w2 = 2.0 * parameters.psi * std::sqrt(pi);
    parameters.w3 = parameters.xi * pi;
    parameters.b = parameters.complement / alpha;
    return parameters;
}

double exponentially_tilted_stable(
    RandomStream &random,
    const TiltedStableParameters &parameters
) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double sqrt_pi_over_two = 1.253314137315500251207882642406;
    const double alpha = parameters.alpha;
    const double lambda = parameters.lambda;
    const double lambda_alpha = parameters.lambda_alpha;
    const double gamma_parameter = parameters.gamma_parameter;
    const double sqrt_gamma = parameters.sqrt_gamma;
    const double xi = parameters.xi;
    const double psi = parameters.psi;
    const double w1 = parameters.w1;
    const double w2 = parameters.w2;
    const double w3 = parameters.w3;
    const double b = parameters.b;

    while (true) {
        double u = 0.0;
        double z_uniform = 0.0;
        double a_value = 0.0;
        double z_value = 0.0;

        while (true) {
            const double selector = random.uniform();
            const double mixture_uniform = random.uniform();
            if (gamma_parameter >= 1.0) {
                if (selector < w1 / (w1 + w2)) {
                    u = std::abs(random.normal()) / sqrt_gamma;
                } else {
                    u = pi * (1.0 - mixture_uniform * mixture_uniform);
                }
            } else if (selector < w3 / (w3 + w2)) {
                u = pi * mixture_uniform;
            } else {
                u = pi * (1.0 - mixture_uniform * mixture_uniform);
            }
            if (!(u > 0.0 && u < pi)) continue;

            const double b_ratio = zolotarev_b_ratio(u, alpha);
            const double zeta = std::sqrt(std::max(b_ratio, 1e-300));
            const double phi = std::pow(sqrt_gamma + alpha * zeta, 1.0 / alpha);
            const double gamma_power = std::pow(sqrt_gamma, 1.0 / alpha);
            const double denominator = std::max(phi - gamma_power, 1e-300);
            z_value = phi / denominator;

            double envelope = psi / std::sqrt(std::max(pi - u, 1e-300));
            if (gamma_parameter >= 1.0) {
                envelope += xi * std::exp(-gamma_parameter * u * u / 2.0);
            } else {
                envelope += xi;
            }
            const double rho_denominator =
                (1.0 + sqrt_pi_over_two) * sqrt_gamma / zeta + z_value;
            const double log_rho = std::log(pi)
                - lambda_alpha * (1.0 - 1.0 / (zeta * zeta))
                + std::log(std::max(envelope, 1e-300))
                - std::log(std::max(rho_denominator, 1e-300));
            const double log_uniform = std::log(
                std::max(random.uniform(), std::numeric_limits<double>::min())
            );
            if (log_uniform + log_rho > 0.0) continue;

            z_uniform = std::exp(log_uniform + log_rho);
            a_value = zolotarev_a(u, alpha, b_ratio);
            break;
        }

        const double m = std::pow(b * lambda / a_value, alpha);
        const double delta = std::sqrt(m * alpha / a_value);
        const double a1 = delta * sqrt_pi_over_two;
        const double a2 = delta;
        const double a3 = z_value / a_value;
        const double sum = a1 + a2 + a3;
        const double selector = random.uniform();
        double x = 0.0;
        double normal_draw = 0.0;
        double exponential_draw = 0.0;
        if (selector < a1 / sum) {
            normal_draw = random.normal();
            x = m - delta * std::abs(normal_draw);
        } else if (selector < (a1 + a2) / sum) {
            x = m + delta * random.uniform();
        } else {
            exponential_draw = random.exponential();
            x = m + delta + exponential_draw * a3;
        }
        if (x < 0.0 || !std::isfinite(x)) continue;

        const double energy = -std::log(
            std::max(z_uniform, std::numeric_limits<double>::min())
        );
        double acceptance = a_value * (x - m)
            + lambda * (std::pow(x, -b) - std::pow(m, -b));
        if (x < m) acceptance -= normal_draw * normal_draw / 2.0;
        if (x > m + delta) acceptance -= exponential_draw;
        if (acceptance <= energy) return std::pow(x, -b);
    }
}

double exponentially_tilted_stable_simple_rejection(
    RandomStream &random,
    const TiltedStableParameters &parameters
) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double alpha = parameters.alpha;
    const double complement = parameters.complement;
    const double stable_power = complement / alpha;
    while (true) {
        const double angle = pi * random.uniform();
        if (!(angle > 0.0 && angle < pi)) continue;
        const double b_ratio = zolotarev_b_ratio(angle, alpha);
        const double a_value = zolotarev_a(angle, alpha, b_ratio);
        const double exponential = random.exponential();
        const double stable = std::pow(
            a_value / std::max(exponential, std::numeric_limits<double>::min()),
            stable_power
        );
        if (!std::isfinite(stable) || stable <= 0.0) continue;
        const double log_uniform = std::log(
            std::max(random.uniform(), std::numeric_limits<double>::min())
        );
        if (log_uniform <= -parameters.lambda * stable) return stable;
    }
}

struct NTSSubordinatorParameters {
    double scale;
    TiltedStableParameters tilted;
};

// Kanter stable rejection wins comfortably in this range; Devroye's uniformly
// bounded double-rejection sampler remains the fallback for stronger tilting.
constexpr double kSimpleRejectionMaxLambdaAlpha = 1.5;

NTSSubordinatorParameters make_nts_subordinator_parameters(
    double tail_index,
    double tempering
) {
    const double alpha = tail_index / 2.0;
    const double log_scale = (
        (1.0 - alpha) * std::log(tempering) - std::log(alpha)
    ) / alpha;
    const double scale = std::exp(log_scale);
    return {scale, make_tilted_stable_parameters(alpha, tempering * scale)};
}

double nts_subordinator(
    RandomStream &random,
    const NTSSubordinatorParameters &parameters
) {
    const double tilted = parameters.tilted.lambda_alpha
            <= kSimpleRejectionMaxLambdaAlpha
        ? exponentially_tilted_stable_simple_rejection(random, parameters.tilted)
        : exponentially_tilted_stable(random, parameters.tilted);
    return parameters.scale * tilted;
}

template <typename Function>
void parallel_paths(int paths, int requested_threads, Function function) {
    const int thread_count = std::max(1, std::min(paths, requested_threads));
    if (thread_count == 1) {
        function(0, paths);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(thread_count));
    for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
        const int begin = paths * thread_index / thread_count;
        const int end = paths * (thread_index + 1) / thread_count;
        threads.emplace_back(function, begin, end);
    }
    for (auto &thread : threads) thread.join();
}

}  // namespace

extern "C" int mc_sample_mnts_subordinators(
    int samples,
    double tail_index,
    double tempering,
    std::uint64_t seed,
    double *output
) {
    if (samples <= 0 || !(tail_index > 0.0 && tail_index < 2.0) ||
        !(tempering > 0.0) || !output) return 1;
    RandomStream random(seed, 0);
    const NTSSubordinatorParameters parameters = make_nts_subordinator_parameters(
        tail_index,
        tempering
    );
    for (int index = 0; index < samples; ++index) {
        output[index] = nts_subordinator(random, parameters);
        if (!std::isfinite(output[index]) || output[index] <= 0.0) return 2;
    }
    return 0;
}

extern "C" int mc_sample_exponentially_tilted_stable(
    int samples,
    double alpha,
    double lambda,
    std::uint64_t seed,
    double *output
) {
    if (samples <= 0 || !(alpha > 0.0 && alpha < 1.0) || !(lambda > 0.0) || !output) {
        return 1;
    }
    RandomStream random(seed, 0);
    const TiltedStableParameters parameters = make_tilted_stable_parameters(alpha, lambda);
    for (int index = 0; index < samples; ++index) {
        output[index] = exponentially_tilted_stable(random, parameters);
        if (!std::isfinite(output[index]) || output[index] <= 0.0) return 2;
    }
    return 0;
}

extern "C" int mc_simulate_parametric(
    int periods,
    int paths,
    int assets,
    int states,
    int macro_dimensions,
    const std::uint8_t *regimes,
    const double *means,
    const double *gaussian_correlation_cholesky,
    const double *gaussian_correlations,
    const double *volatilities,
    const double *tail_indexes,
    const double *temperings,
    const double *skewness,
    const double *gaussian_scales,
    const double *macro_shocks,
    const double *macro_betas,
    std::uint64_t seed,
    int garch,
    double garch_alpha,
    double garch_beta,
    int dynamic_correlation,
    double dcc_alpha,
    double dcc_beta,
    double dcc_asymmetry,
    int requested_threads,
    double *output
) {
    if (periods <= 0 || paths <= 0 || assets <= 0 || states <= 0 || !regimes || !means ||
        !gaussian_correlation_cholesky || !gaussian_correlations || !volatilities ||
        !tail_indexes || !temperings || !skewness || !gaussian_scales || !output) return 1;
    thread_local std::vector<NTSSubordinatorParameters> state_subordinators;
    thread_local std::vector<double> cached_tail_indexes;
    thread_local std::vector<double> cached_temperings;
    bool refresh_subordinators = state_subordinators.size()
        != static_cast<std::size_t>(states);
    for (int state = 0; state < states; ++state) {
        if (!(tail_indexes[state] > 0.0 && tail_indexes[state] < 2.0) ||
            !(temperings[state] > 0.0)) return 2;
        refresh_subordinators = refresh_subordinators
            || cached_tail_indexes[state] != tail_indexes[state]
            || cached_temperings[state] != temperings[state];
        for (int asset = 0; asset < assets; ++asset) {
            const std::size_t index = static_cast<std::size_t>(state) * assets + asset;
            if (!std::isfinite(skewness[index]) || !(gaussian_scales[index] > 0.0)) return 3;
        }
    }
    if (refresh_subordinators) {
        state_subordinators.clear();
        state_subordinators.reserve(static_cast<std::size_t>(states));
        cached_tail_indexes.assign(tail_indexes, tail_indexes + states);
        cached_temperings.assign(temperings, temperings + states);
        for (int state = 0; state < states; ++state) {
            state_subordinators.push_back(make_nts_subordinator_parameters(
                tail_indexes[state],
                temperings[state]
            ));
        }
    }
    // Capture the caller thread's read-only storage explicitly: referring to a
    // thread_local vector from worker threads would select each worker's empty
    // instance instead.
    const NTSSubordinatorParameters *subordinator_parameters =
        state_subordinators.data();

    parallel_paths(paths, requested_threads, [&](int begin, int end) {
        thread_local std::vector<double> q;
        thread_local std::vector<double> factor;
        thread_local std::vector<double> previous;
        thread_local std::vector<double> independent;
        thread_local std::vector<double> standardized;
        thread_local std::vector<double> conditional_variance;
        q.resize(static_cast<std::size_t>(assets) * assets);
        factor.resize(static_cast<std::size_t>(assets) * assets);
        previous.resize(static_cast<std::size_t>(assets));
        independent.resize(static_cast<std::size_t>(assets));
        standardized.resize(static_cast<std::size_t>(assets));
        conditional_variance.resize(static_cast<std::size_t>(assets));
        for (int path = begin; path < end; ++path) {
            RandomStream random(seed, static_cast<std::uint64_t>(path));
            std::fill(previous.begin(), previous.end(), 0.0);
            std::fill(conditional_variance.begin(), conditional_variance.end(), 0.0);
            int previous_state = -1;

        for (int period = 0; period < periods; ++period) {
            const std::size_t path_offset = static_cast<std::size_t>(period) * paths + path;
            const int state = static_cast<int>(regimes[path_offset]);
            if (state < 0 || state >= states) continue;
            const bool reanchored = previous_state != state;
            const double *base = state_matrix(gaussian_correlations, state, assets);
            const double *correlation_factor = state_matrix(
                gaussian_correlation_cholesky,
                state,
                assets
            );
            const double *state_mean = state_vector(means, state, assets);
            const double *state_volatility = state_vector(volatilities, state, assets);
            const double *state_skewness = state_vector(skewness, state, assets);
            const double *state_gaussian_scale = state_vector(gaussian_scales, state, assets);

            for (int asset = 0; asset < assets; ++asset) independent[asset] = random.normal();

            if (dynamic_correlation) {
                if (reanchored) {
                    std::copy(base, base + assets * assets, q.begin());
                } else {
                    const double base_weight = 1.0 - dcc_alpha - dcc_beta - dcc_asymmetry;
                    for (int row = 0; row < assets; ++row) {
                        const double negative_row = std::min(previous[row], 0.0);
                        for (int column = 0; column < assets; ++column) {
                            const int index = row * assets + column;
                            q[index] = base_weight * base[index]
                                + dcc_alpha * previous[row] * previous[column]
                                + dcc_beta * q[index]
                                + dcc_asymmetry * negative_row * std::min(previous[column], 0.0);
                        }
                    }
                }
                cholesky(q, factor, assets);
                for (int row = 0; row < assets; ++row) {
                    double value = 0.0;
                    for (int column = 0; column <= row; ++column) {
                        value += factor[row * assets + column] * independent[column];
                    }
                    standardized[row] = value / std::sqrt(std::max(q[row * assets + row], 1e-10));
                }
            } else {
                for (int row = 0; row < assets; ++row) {
                    double value = 0.0;
                    for (int column = 0; column <= row; ++column) {
                        value += correlation_factor[row * assets + column] * independent[column];
                    }
                    standardized[row] = value;
                }
            }

            const double subordinator = nts_subordinator(
                random,
                subordinator_parameters[state]
            );
            const double root_subordinator = std::sqrt(subordinator);
            for (int asset = 0; asset < assets; ++asset) {
                standardized[asset] = state_skewness[asset] * (subordinator - 1.0)
                    + root_subordinator * state_gaussian_scale[asset] * standardized[asset];
            }

            for (int asset = 0; asset < assets; ++asset) {
                double residual = standardized[asset];
                if (garch) {
                    const double level = state_volatility[asset] * state_volatility[asset];
                    if (reanchored) conditional_variance[asset] = level;
                    residual *= std::sqrt(std::max(conditional_variance[asset], 0.0));
                    conditional_variance[asset] =
                        (1.0 - garch_alpha - garch_beta) * level
                        + garch_alpha * residual * residual
                        + garch_beta * conditional_variance[asset];
                } else {
                    residual *= state_volatility[asset];
                }

                double macro_effect = 0.0;
                if (macro_dimensions > 0 && macro_shocks && macro_betas) {
                    const double *shock = macro_shocks
                        + path_offset * static_cast<std::size_t>(macro_dimensions);
                    for (int dimension = 0; dimension < macro_dimensions; ++dimension) {
                        macro_effect += shock[dimension]
                            * macro_betas[static_cast<std::size_t>(dimension) * assets + asset];
                    }
                }
                output[(path_offset * assets) + asset] = state_mean[asset] + macro_effect + residual;
            }
            previous = standardized;
            previous_state = state;
        }
        }
    });
    return 0;
}

namespace {

constexpr double kItalianTaxRate = 0.26;
constexpr int kLossBuckets = 5;

enum TaxStat {
    kCapitalGainsTax = 0,
    kInvestmentIncomeTax,
    kForeignWithholdingTax,
    kFinancialTransactionTax,
    kWealthTax,
    kStampDuty,
    kIvafe,
    kTerminalTax,
    kTaxesPaid,
    kRealizedGains,
    kRealizedLosses,
    kLossCarryforward,
    kExpiredLosses,
    kTransactionCosts,
    kTaxStatCount,
};

enum YearStat {
    kYearCapitalGainsTax = 0,
    kYearManagedResultTax,
    kYearDeferredTaxPayment,
    kYearExpiredLosses,
    kYearFinancialTransactionTax,
    kYearStampDuty,
    kYearIvafe,
    kYearTerminalTax,
    kYearGrossSalesForSpending,
    kYearNetSpending,
    kYearStatCount,
};

struct SaleResult {
    double tax = 0.0;
    double gains = 0.0;
    double losses = 0.0;
};

double sum_values(const std::vector<double> &values) {
    double result = 0.0;
    for (double value : values) result += value;
    return result;
}

double consume_losses(std::vector<double> &losses, double amount) {
    double remaining = std::max(amount, 0.0);
    for (double &bucket : losses) {
        const double used = std::min(bucket, remaining);
        bucket -= used;
        remaining -= used;
    }
    return remaining;
}

double advance_year(std::vector<double> &losses) {
    const double expired = losses.front();
    for (int bucket = 0; bucket < kLossBuckets - 1; ++bucket) {
        losses[bucket] = losses[bucket + 1];
    }
    losses.back() = 0.0;
    return expired;
}

void allocate_contribution(
    const std::vector<double> &holdings,
    const double *weights,
    double contribution,
    int mode,
    std::vector<double> &allocation
) {
    const int assets = static_cast<int>(holdings.size());
    std::fill(allocation.begin(), allocation.end(), 0.0);
    if (contribution <= 0.0) return;
    if (mode == 0) {
        for (int asset = 0; asset < assets; ++asset) {
            allocation[asset] = contribution * weights[asset];
        }
        return;
    }
    const double target_value = sum_values(holdings) + contribution;
    double deficit_total = 0.0;
    for (int asset = 0; asset < assets; ++asset) {
        allocation[asset] = std::max(target_value * weights[asset] - holdings[asset], 0.0);
        deficit_total += allocation[asset];
    }
    const double applied = std::min(deficit_total, contribution);
    if (deficit_total > 0.0) {
        for (double &value : allocation) value *= applied / deficit_total;
    }
    const double residual = contribution - applied;
    for (int asset = 0; asset < assets; ++asset) {
        allocation[asset] += residual * weights[asset];
    }
}

void deduct_charge(
    std::vector<double> &holdings,
    std::vector<double> &basis,
    double charge
) {
    const double value = sum_values(holdings);
    const double scale = value > 0.0
        ? std::max(value - std::max(charge, 0.0), 0.0) / value
        : 0.0;
    for (std::size_t asset = 0; asset < holdings.size(); ++asset) {
        holdings[asset] *= scale;
        basis[asset] *= scale;
    }
}

void deduct_wrapper_charge(
    std::vector<double> &holdings,
    double &basis,
    double charge
) {
    const double value = sum_values(holdings);
    const double scale = value > 0.0
        ? std::max(value - std::max(charge, 0.0), 0.0) / value
        : 0.0;
    for (double &holding : holdings) holding *= scale;
    basis *= scale;
}

double preview_sales_tax(
    const std::vector<double> &sales,
    const std::vector<double> &holdings,
    const std::vector<double> &basis,
    const double *taxable_fraction,
    const std::uint8_t *offsettable,
    const std::vector<double> &loss_buckets,
    double transaction_cost_rate
) {
    double non_offsettable = 0.0;
    double offsettable_gains = 0.0;
    double available_losses = sum_values(loss_buckets);
    for (std::size_t asset = 0; asset < holdings.size(); ++asset) {
        const double basis_sold = holdings[asset] > 0.0
            ? basis[asset] * sales[asset] / holdings[asset]
            : 0.0;
        const double taxable = (
            sales[asset] * (1.0 - transaction_cost_rate) - basis_sold
        ) * taxable_fraction[asset];
        if (taxable < 0.0) {
            available_losses += -taxable;
        } else if (offsettable[asset]) {
            offsettable_gains += taxable;
        } else {
            non_offsettable += taxable;
        }
    }
    return (non_offsettable + std::max(offsettable_gains - available_losses, 0.0))
        * kItalianTaxRate;
}

SaleResult settle_sales(
    const std::vector<double> &sales,
    std::vector<double> &holdings,
    std::vector<double> &basis,
    const double *taxable_fraction,
    const std::uint8_t *offsettable,
    std::vector<double> &loss_buckets,
    double transaction_cost_rate,
    bool apply_sales
) {
    SaleResult result;
    double non_offsettable = 0.0;
    double offsettable_gains = 0.0;
    double new_losses = 0.0;
    for (std::size_t asset = 0; asset < holdings.size(); ++asset) {
        const double basis_sold = holdings[asset] > 0.0
            ? basis[asset] * sales[asset] / holdings[asset]
            : 0.0;
        const double realized = sales[asset] * (1.0 - transaction_cost_rate)
            - basis_sold;
        const double taxable = realized * taxable_fraction[asset];
        result.gains += std::max(realized, 0.0);
        result.losses += std::max(-realized, 0.0);
        if (taxable < 0.0) {
            new_losses += -taxable;
        } else if (offsettable[asset]) {
            offsettable_gains += taxable;
        } else {
            non_offsettable += taxable;
        }
    }
    loss_buckets.back() += new_losses;
    result.tax = (
        non_offsettable + consume_losses(loss_buckets, offsettable_gains)
    ) * kItalianTaxRate;
    if (apply_sales) {
        for (std::size_t asset = 0; asset < holdings.size(); ++asset) {
            const double basis_sold = holdings[asset] > 0.0
                ? basis[asset] * sales[asset] / holdings[asset]
                : 0.0;
            basis[asset] = std::max(basis[asset] - basis_sold, 0.0);
            holdings[asset] = std::max(holdings[asset] - sales[asset], 0.0);
        }
    }
    return result;
}

SaleResult raise_cash(
    double requested,
    bool immediate,
    std::vector<double> &holdings,
    std::vector<double> &basis,
    const double *taxable_fraction,
    const std::uint8_t *offsettable,
    std::vector<double> &loss_buckets,
    std::vector<double> &sales
) {
    SaleResult total;
    double cash_required = std::max(requested, 0.0);
    const int attempts = immediate ? 12 : 1;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        const double value = sum_values(holdings);
        const double gross_sale = std::min(cash_required, value);
        if (gross_sale <= 1e-10 || value <= 0.0) break;
        for (std::size_t asset = 0; asset < holdings.size(); ++asset) {
            sales[asset] = holdings[asset] * gross_sale / value;
        }
        const SaleResult event = settle_sales(
            sales,
            holdings,
            basis,
            taxable_fraction,
            offsettable,
            loss_buckets,
            0.0,
            true
        );
        total.tax += event.tax;
        total.gains += event.gains;
        total.losses += event.losses;
        cash_required = event.tax;
    }
    return total;
}

struct RebalanceResult {
    SaleResult sale;
    double ftt = 0.0;
    double transaction_cost = 0.0;
};

RebalanceResult rebalance_taxed(
    std::vector<double> &holdings,
    std::vector<double> &basis,
    const double *weights,
    double cost_rate,
    const double *taxable_fraction,
    const std::uint8_t *offsettable,
    const double *ftt_rates,
    std::vector<double> &loss_buckets,
    bool immediate,
    std::vector<double> &sales,
    std::vector<double> &purchases
) {
    const int assets = static_cast<int>(holdings.size());
    const double value = sum_values(holdings);
    double post_cost_value = value;
    std::fill(sales.begin(), sales.end(), 0.0);
    std::fill(purchases.begin(), purchases.end(), 0.0);
    for (int iteration = 0; iteration < 8; ++iteration) {
        double turnover = 0.0;
        double ftt = 0.0;
        for (int asset = 0; asset < assets; ++asset) {
            const double target = post_cost_value * weights[asset];
            sales[asset] = std::max(holdings[asset] - target, 0.0);
            purchases[asset] = std::max(target - holdings[asset], 0.0);
            turnover += sales[asset] + purchases[asset];
            ftt += purchases[asset] * ftt_rates[asset];
        }
        const double tax = immediate
            ? preview_sales_tax(
                sales,
                holdings,
                basis,
                taxable_fraction,
                offsettable,
                loss_buckets,
                cost_rate
            )
            : 0.0;
        const double updated = std::max(value - tax - turnover * cost_rate - ftt, 0.0);
        const double tolerance = 1e-10 + 1e-11 * std::abs(post_cost_value);
        const bool converged = std::abs(updated - post_cost_value) <= tolerance;
        post_cost_value = updated;
        if (converged) break;
    }

    RebalanceResult result;
    for (int asset = 0; asset < assets; ++asset) {
        const double target = post_cost_value * weights[asset];
        sales[asset] = std::max(holdings[asset] - target, 0.0);
        purchases[asset] = std::max(target - holdings[asset], 0.0);
        result.transaction_cost += (sales[asset] + purchases[asset]) * cost_rate;
        result.ftt += purchases[asset] * ftt_rates[asset];
    }
    result.sale = settle_sales(
        sales,
        holdings,
        basis,
        taxable_fraction,
        offsettable,
        loss_buckets,
        cost_rate,
        true
    );
    for (int asset = 0; asset < assets; ++asset) {
        basis[asset] += purchases[asset] * (1.0 + cost_rate + ftt_rates[asset]);
        holdings[asset] += purchases[asset];
    }
    return result;
}

double masked_value(const std::vector<double> &holdings, const std::uint8_t *mask) {
    double result = 0.0;
    for (std::size_t asset = 0; asset < holdings.size(); ++asset) {
        if (mask[asset]) result += holdings[asset];
    }
    return result;
}

double wrapper_taxable_fraction(
    const std::vector<double> &holdings,
    const double *taxable_fraction
) {
    const double value = sum_values(holdings);
    if (value <= 0.0) return 1.0;
    double weighted = 0.0;
    for (std::size_t asset = 0; asset < holdings.size(); ++asset) {
        weighted += holdings[asset] * taxable_fraction[asset];
    }
    return weighted / value;
}

double sell_wrapper(
    double requested,
    std::vector<double> &holdings,
    double &basis,
    const double *taxable_fraction
) {
    const double value = sum_values(holdings);
    const double gross_sale = std::min(std::max(requested, 0.0), value);
    const double fraction = value > 0.0 ? gross_sale / value : 0.0;
    const double basis_sold = basis * fraction;
    const double gain = std::max(gross_sale - basis_sold, 0.0);
    const double tax = gain * wrapper_taxable_fraction(holdings, taxable_fraction)
        * kItalianTaxRate;
    for (double &holding : holdings) holding *= 1.0 - fraction;
    basis = std::max(basis - basis_sold, 0.0);
    return tax;
}

void add_year_stat(
    std::vector<double> &year_stats,
    int slot,
    int metric,
    double value
) {
    if (slot >= 0) {
        year_stats[static_cast<std::size_t>(slot) * kYearStatCount + metric] += value;
    }
}

}  // namespace

extern "C" int mc_simulate_italian_portfolios(
    int periods,
    int paths,
    int assets,
    int requested_threads,
    const double *growth,
    const double *weights,
    double initial_value,
    int rebalance_frequency,
    const double *transaction_cost_rate_paths,
    double default_transaction_cost_rate,
    double contribution,
    int contribution_mode,
    double withdrawal,
    int withdrawal_start_period,
    int advanced_decumulation,
    int phase_count,
    const int *phase_starts,
    const int *phase_ends,
    const int *phase_frequencies,
    const double *phase_amounts,
    const double *phase_multipliers,
    int safe_rate_mode,
    double safe_withdrawal_rate,
    const double *one_time_expenses,
    const double *withdrawal_cpi,
    int guardrail_policy,
    int guardrail_review_months,
    double upper_guardrail,
    double lower_guardrail,
    double guardrail_adjustment,
    double spending_floor,
    double spending_ceiling,
    int skip_inflation_after_loss,
    int tax_regime,
    const double *taxable_fraction,
    const std::uint8_t *offsettable,
    const double *ftt_rates,
    const std::uint8_t *stamp_mask,
    const std::uint8_t *ivafe_mask,
    double annual_wealth_tax,
    int terminal_liquidation,
    int wrapper_benchmark,
    const int *year_slots,
    int year_count,
    double *gross_wealth,
    double *diy_wealth,
    double *wrapper_terminal,
    double *wrapper_annualized,
    double *tax_stats,
    double *gross_transaction_costs,
    double *year_stats,
    double *requested_spending,
    double *funded_spending,
    std::int8_t *guardrail_events
) {
    if (periods <= 0 || paths <= 0 || assets <= 0 || !growth || !weights ||
        !taxable_fraction || !offsettable || !ftt_rates || !stamp_mask || !ivafe_mask ||
        !year_slots || year_count <= 0 || !gross_wealth || !diy_wealth || !tax_stats ||
        !gross_transaction_costs || !year_stats) return 1;
    if (tax_regime < 0 || tax_regime > 2 || contribution_mode < 0 || contribution_mode > 1 ||
        withdrawal_start_period < 1 || withdrawal_start_period > periods) {
        return 2;
    }
    if (advanced_decumulation && (
        phase_count < 0 || !one_time_expenses || !withdrawal_cpi ||
        !requested_spending || !funded_spending || !guardrail_events ||
        guardrail_review_months <= 0 || lower_guardrail >= upper_guardrail ||
        spending_floor < 0.0 || spending_ceiling < spending_floor ||
        (phase_count > 0 && (
            !phase_starts || !phase_ends || !phase_frequencies ||
            !phase_amounts || !phase_multipliers
        ))
    )) return 3;

    std::fill(year_stats, year_stats + static_cast<std::size_t>(year_count) * kYearStatCount, 0.0);
    std::mutex year_mutex;
    parallel_paths(paths, requested_threads, [&](int begin, int end) {
        thread_local std::vector<double> local_year_stats;
        thread_local std::vector<double> gross;
        thread_local std::vector<double> holdings;
        thread_local std::vector<double> basis;
        thread_local std::vector<double> wrapper;
        thread_local std::vector<double> losses;
        thread_local std::vector<double> rebalance_sales;
        thread_local std::vector<double> rebalance_purchases;
        thread_local std::vector<double> contribution_allocation;
        local_year_stats.assign(
            static_cast<std::size_t>(year_count) * kYearStatCount,
            0.0
        );
        gross.resize(static_cast<std::size_t>(assets));
        holdings.resize(static_cast<std::size_t>(assets));
        basis.resize(static_cast<std::size_t>(assets));
        wrapper.resize(static_cast<std::size_t>(assets));
        losses.resize(kLossBuckets);
        rebalance_sales.resize(static_cast<std::size_t>(assets));
        rebalance_purchases.resize(static_cast<std::size_t>(assets));
        contribution_allocation.resize(static_cast<std::size_t>(assets));
        for (int path = begin; path < end; ++path) {
            std::fill(gross.begin(), gross.end(), 0.0);
            std::fill(holdings.begin(), holdings.end(), 0.0);
            std::fill(basis.begin(), basis.end(), 0.0);
            std::fill(wrapper.begin(), wrapper.end(), 0.0);
            std::fill(losses.begin(), losses.end(), 0.0);
            for (int asset = 0; asset < assets; ++asset) {
                gross[asset] = initial_value * weights[asset];
                holdings[asset] = gross[asset];
                basis[asset] = gross[asset];
                wrapper[asset] = gross[asset];
            }

            double capital_tax = 0.0;
            double ftt_total = 0.0;
            double wealth_tax = 0.0;
            double stamp_total = 0.0;
            double ivafe_total = 0.0;
            double terminal_tax = 0.0;
            double realized_gains = 0.0;
            double realized_losses = 0.0;
            double expired_losses = 0.0;
            double transaction_cost_total = 0.0;
            double gross_cost_total = 0.0;
            double pending_tax = 0.0;
            double year_start_value = initial_value;
            double year_contributions = 0.0;
            double year_withdrawals = 0.0;
            double wrapper_basis = initial_value;
            double wrapper_pending_tax = 0.0;
            double wrapper_previous_value = initial_value;
            double wrapper_return_product = 1.0;
            int wrapper_return_count = 0;
            const bool managed = tax_regime == 2;
            const bool declarative = tax_regime == 1;
            const double wealth_tax_rate = annual_wealth_tax / 12.0;

            struct SpendingState {
                int phase = -1;
                double annual_nominal = 0.0;
                double reference_rate = 0.0;
                double last_review_wealth = 0.0;
                double last_review_cpi = 1.0;
            };
            SpendingState diy_spending;
            SpendingState gross_spending;
            SpendingState wrapper_spending;
            diy_spending.last_review_wealth = initial_value;
            gross_spending.last_review_wealth = initial_value;
            wrapper_spending.last_review_wealth = initial_value;

            auto spending_request = [&](double current_wealth, SpendingState &state, int period,
                                        std::int8_t *event_output) {
                if (!advanced_decumulation) {
                    return period + 1 >= withdrawal_start_period ? withdrawal : 0.0;
                }
                const int month = period + 1;
                const std::size_t path_index = static_cast<std::size_t>(period) * paths + path;
                const double cpi = withdrawal_cpi[path_index];
                int active_phase = -1;
                for (int phase = 0; phase < phase_count; ++phase) {
                    if (month >= phase_starts[phase] && month <= phase_ends[phase]) {
                        active_phase = phase;
                        break;
                    }
                }
                double requested = one_time_expenses[period] * cpi;
                std::int8_t event = 0;
                if (active_phase >= 0) {
                    const double base_real = safe_rate_mode
                        ? initial_value * safe_withdrawal_rate * phase_multipliers[active_phase]
                        : phase_amounts[active_phase];
                    const bool reset = state.phase != active_phase;
                    const bool review = reset ||
                        (month - phase_starts[active_phase]) % guardrail_review_months == 0;
                    if (reset) {
                        state.phase = active_phase;
                        state.annual_nominal = base_real * cpi;
                        state.reference_rate = current_wealth > 0.0
                            ? state.annual_nominal / current_wealth
                            : std::numeric_limits<double>::infinity();
                        state.last_review_wealth = current_wealth;
                        state.last_review_cpi = cpi;
                    } else if (review && guardrail_policy == 1) {
                        const double previous_real = state.last_review_wealth /
                            std::max(state.last_review_cpi, 1e-300);
                        const double current_real = current_wealth / std::max(cpi, 1e-300);
                        const double real_return = previous_real > 0.0
                            ? current_real / previous_real - 1.0
                            : -1.0;
                        if (!skip_inflation_after_loss || real_return >= 0.0) {
                            state.annual_nominal *= cpi / std::max(state.last_review_cpi, 1e-300);
                        }
                        const double current_rate = current_wealth > 0.0
                            ? state.annual_nominal / current_wealth
                            : std::numeric_limits<double>::infinity();
                        const double before = state.annual_nominal;
                        if (current_rate > upper_guardrail * state.reference_rate) {
                            state.annual_nominal *= 1.0 - guardrail_adjustment;
                            event = -1;
                        } else if (current_rate < lower_guardrail * state.reference_rate) {
                            state.annual_nominal *= 1.0 + guardrail_adjustment;
                            event = 1;
                        }
                        state.annual_nominal = std::clamp(
                            state.annual_nominal,
                            base_real * spending_floor * cpi,
                            base_real * spending_ceiling * cpi
                        );
                        if (std::abs(before - state.annual_nominal) <= 1e-12) event = 0;
                        state.last_review_wealth = current_wealth;
                        state.last_review_cpi = cpi;
                    } else if (guardrail_policy == 0) {
                        state.annual_nominal = base_real * cpi;
                    }
                    if ((month - phase_starts[active_phase]) % phase_frequencies[active_phase] == 0) {
                        requested += state.annual_nominal * phase_frequencies[active_phase] / 12.0;
                    }
                }
                if (event_output) *event_output = event;
                return std::max(requested, 0.0);
            };

            auto settle_managed_year = [&](int slot) {
                const double result = sum_values(holdings) + year_withdrawals
                    - year_contributions - year_start_value;
                double weighted_fraction = 0.0;
                for (int asset = 0; asset < assets; ++asset) {
                    weighted_fraction += weights[asset] * taxable_fraction[asset];
                }
                const double base = result * weighted_fraction;
                losses.back() += std::max(-base, 0.0);
                const double tax = consume_losses(losses, std::max(base, 0.0))
                    * kItalianTaxRate;
                deduct_charge(holdings, basis, tax);
                capital_tax += tax;
                add_year_stat(local_year_stats, slot, kYearManagedResultTax, tax);
                year_start_value = sum_values(holdings);
                year_contributions = 0.0;
                year_withdrawals = 0.0;
            };

            for (int period = 0; period < periods; ++period) {
                const int slot = year_slots[period];
                if (period > 0 && slot != year_slots[period - 1]) {
                    const int previous_slot = year_slots[period - 1];
                    if (managed) {
                        settle_managed_year(previous_slot);
                    } else if (declarative && pending_tax > 0.0) {
                        deduct_charge(holdings, basis, pending_tax);
                        add_year_stat(
                            local_year_stats,
                            previous_slot,
                            kYearDeferredTaxPayment,
                            pending_tax
                        );
                        pending_tax = 0.0;
                    }
                    const double expired = advance_year(losses);
                    expired_losses += expired;
                    add_year_stat(
                        local_year_stats,
                        previous_slot,
                        kYearExpiredLosses,
                        expired
                    );
                    if (wrapper_benchmark && declarative && wrapper_pending_tax > 0.0) {
                        deduct_wrapper_charge(wrapper, wrapper_basis, wrapper_pending_tax);
                        wrapper_pending_tax = 0.0;
                    }
                }

                if (contribution > 0.0) {
                    allocate_contribution(
                        gross,
                        weights,
                        contribution,
                        contribution_mode,
                        contribution_allocation
                    );
                    for (int asset = 0; asset < assets; ++asset) {
                        gross[asset] += contribution_allocation[asset];
                    }
                    allocate_contribution(
                        holdings,
                        weights,
                        contribution,
                        contribution_mode,
                        contribution_allocation
                    );
                    for (int asset = 0; asset < assets; ++asset) {
                        const double purchase = contribution_allocation[asset]
                            / (1.0 + ftt_rates[asset]);
                        holdings[asset] += purchase;
                        basis[asset] += contribution_allocation[asset];
                        const double ftt = contribution_allocation[asset] - purchase;
                        ftt_total += ftt;
                        add_year_stat(
                            local_year_stats,
                            slot,
                            kYearFinancialTransactionTax,
                            ftt
                        );
                    }
                    year_contributions += contribution;

                    if (wrapper_benchmark) {
                        allocate_contribution(
                            wrapper,
                            weights,
                            contribution,
                            contribution_mode,
                            contribution_allocation
                        );
                        for (int asset = 0; asset < assets; ++asset) {
                            wrapper[asset] += contribution_allocation[asset]
                                / (1.0 + ftt_rates[asset]);
                        }
                        wrapper_basis += contribution;
                    }
                }

                const double opening_value = sum_values(holdings);
                for (int asset = 0; asset < assets; ++asset) {
                    const std::size_t index = (
                        (static_cast<std::size_t>(period) * paths + path) * assets + asset
                    );
                    gross[asset] *= growth[index];
                    holdings[asset] *= growth[index];
                    if (wrapper_benchmark) wrapper[asset] *= growth[index];
                }

                std::int8_t policy_event = 0;
                const double active_withdrawal = spending_request(
                    sum_values(holdings), diy_spending, period, &policy_event
                );
                const double gross_active_withdrawal = spending_request(
                    sum_values(gross), gross_spending, period, nullptr
                );
                const double wrapper_active_withdrawal = wrapper_benchmark
                    ? spending_request(sum_values(wrapper), wrapper_spending, period, nullptr)
                    : 0.0;
                double wrapper_funded_withdrawal = 0.0;
                const std::size_t spending_index = static_cast<std::size_t>(period) * paths + path;
                if (advanced_decumulation) {
                    requested_spending[spending_index] = active_withdrawal;
                    guardrail_events[spending_index] = policy_event;
                }

                if (active_withdrawal > 0.0) {
                    const double diy_before_sale = sum_values(holdings);
                    double immediate_withdrawal_tax = 0.0;
                    const double gross_value = sum_values(gross);
                    const double gross_sale = std::min(gross_active_withdrawal, gross_value);
                    const double gross_scale = gross_value > 0.0
                        ? std::max(gross_value - gross_sale, 0.0) / gross_value
                        : 0.0;
                    for (double &value : gross) value *= gross_scale;

                    if (managed) {
                        const double value = sum_values(holdings);
                        const double sale = std::min(active_withdrawal, value);
                        const double scale = value > 0.0 ? 1.0 - sale / value : 0.0;
                        for (int asset = 0; asset < assets; ++asset) {
                            holdings[asset] *= scale;
                            basis[asset] *= scale;
                        }
                    } else {
                        const SaleResult sale = raise_cash(
                            active_withdrawal,
                            !declarative,
                            holdings,
                            basis,
                            taxable_fraction,
                            offsettable,
                            losses,
                            rebalance_sales
                        );
                        immediate_withdrawal_tax = declarative ? 0.0 : sale.tax;
                        capital_tax += sale.tax;
                        if (declarative) pending_tax += sale.tax;
                        realized_gains += sale.gains;
                        realized_losses += sale.losses;
                        add_year_stat(
                            local_year_stats,
                            slot,
                            kYearCapitalGainsTax,
                            sale.tax
                        );
                    }
                    const double gross_sales_for_spending = std::max(
                        diy_before_sale - sum_values(holdings), 0.0
                    );
                    const double spendable_cash = (managed || declarative)
                        ? gross_sales_for_spending
                        : std::max(
                            gross_sales_for_spending - immediate_withdrawal_tax,
                            0.0
                        );
                    const double funded = std::min(active_withdrawal, spendable_cash);
                    if (advanced_decumulation) {
                        funded_spending[spending_index] = funded;
                    }
                    year_withdrawals += advanced_decumulation
                        ? funded
                        : std::min(active_withdrawal, opening_value);
                    add_year_stat(
                        local_year_stats,
                        slot,
                        kYearGrossSalesForSpending,
                        gross_sales_for_spending
                    );
                    add_year_stat(
                        local_year_stats,
                        slot,
                        kYearNetSpending,
                        funded
                    );

                    if (wrapper_benchmark) {
                        const double wrapper_before_sale = sum_values(wrapper);
                        if (declarative) {
                            wrapper_pending_tax += sell_wrapper(
                                wrapper_active_withdrawal,
                                wrapper,
                                wrapper_basis,
                                taxable_fraction
                            );
                            wrapper_funded_withdrawal = std::min(
                                wrapper_active_withdrawal,
                                std::max(wrapper_before_sale - sum_values(wrapper), 0.0)
                            );
                        } else {
                            double requested = wrapper_active_withdrawal;
                            double wrapper_tax_total = 0.0;
                            for (int attempt = 0; attempt < 12; ++attempt) {
                                const double before = sum_values(wrapper);
                                if (std::min(requested, before) <= 1e-10) break;
                                requested = sell_wrapper(
                                    requested,
                                    wrapper,
                                    wrapper_basis,
                                    taxable_fraction
                                );
                                wrapper_tax_total += requested;
                            }
                            const double wrapper_gross_sales = std::max(
                                wrapper_before_sale - sum_values(wrapper), 0.0
                            );
                            wrapper_funded_withdrawal = std::min(
                                wrapper_active_withdrawal,
                                std::max(wrapper_gross_sales - wrapper_tax_total, 0.0)
                            );
                        }
                    }
                }

                const double active_cost_rate = transaction_cost_rate_paths
                    ? transaction_cost_rate_paths[static_cast<std::size_t>(period) * paths + path]
                    : default_transaction_cost_rate;
                if (rebalance_frequency > 0 && (period + 1) % rebalance_frequency == 0) {
                    const double gross_value = sum_values(gross);
                    double gross_turnover = 0.0;
                    for (int asset = 0; asset < assets; ++asset) {
                        gross_turnover += std::abs(gross_value * weights[asset] - gross[asset]);
                    }
                    const double gross_cost = gross_turnover * active_cost_rate;
                    gross_cost_total += gross_cost;
                    const double gross_after_cost = gross_value - gross_cost;
                    for (int asset = 0; asset < assets; ++asset) {
                        gross[asset] = gross_after_cost * weights[asset];
                    }

                    if (managed) {
                        const double value = sum_values(holdings);
                        double turnover = 0.0;
                        double ftt = 0.0;
                        for (int asset = 0; asset < assets; ++asset) {
                            const double target = value * weights[asset];
                            turnover += std::abs(target - holdings[asset]);
                            ftt += std::max(target - holdings[asset], 0.0) * ftt_rates[asset];
                        }
                        const double transaction_cost = turnover * active_cost_rate;
                        transaction_cost_total += transaction_cost;
                        ftt_total += ftt;
                        add_year_stat(
                            local_year_stats,
                            slot,
                            kYearFinancialTransactionTax,
                            ftt
                        );
                        deduct_charge(holdings, basis, transaction_cost + ftt);
                        const double remaining = sum_values(holdings);
                        for (int asset = 0; asset < assets; ++asset) {
                            holdings[asset] = remaining * weights[asset];
                            basis[asset] = holdings[asset];
                        }
                    } else {
                        const RebalanceResult rebalanced = rebalance_taxed(
                            holdings,
                            basis,
                            weights,
                            active_cost_rate,
                            taxable_fraction,
                            offsettable,
                            ftt_rates,
                            losses,
                            !declarative,
                            rebalance_sales,
                            rebalance_purchases
                        );
                        capital_tax += rebalanced.sale.tax;
                        if (declarative) pending_tax += rebalanced.sale.tax;
                        realized_gains += rebalanced.sale.gains;
                        realized_losses += rebalanced.sale.losses;
                        ftt_total += rebalanced.ftt;
                        transaction_cost_total += rebalanced.transaction_cost;
                        add_year_stat(
                            local_year_stats,
                            slot,
                            kYearCapitalGainsTax,
                            rebalanced.sale.tax
                        );
                        add_year_stat(
                            local_year_stats,
                            slot,
                            kYearFinancialTransactionTax,
                            rebalanced.ftt
                        );
                    }

                    if (wrapper_benchmark) {
                        const double value = sum_values(wrapper);
                        double turnover = 0.0;
                        double ftt = 0.0;
                        for (int asset = 0; asset < assets; ++asset) {
                            const double target = value * weights[asset];
                            turnover += std::abs(target - wrapper[asset]);
                            ftt += std::max(target - wrapper[asset], 0.0) * ftt_rates[asset];
                        }
                        const double remaining = std::max(
                            value - turnover * active_cost_rate - ftt,
                            0.0
                        );
                        for (int asset = 0; asset < assets; ++asset) {
                            wrapper[asset] = remaining * weights[asset];
                        }
                    }
                }

                if (wealth_tax_rate > 0.0) {
                    const double stamp = masked_value(holdings, stamp_mask) * wealth_tax_rate;
                    const double ivafe = masked_value(holdings, ivafe_mask) * wealth_tax_rate;
                    deduct_charge(holdings, basis, stamp + ivafe);
                    wealth_tax += stamp + ivafe;
                    stamp_total += stamp;
                    ivafe_total += ivafe;
                    add_year_stat(local_year_stats, slot, kYearStampDuty, stamp);
                    add_year_stat(local_year_stats, slot, kYearIvafe, ivafe);

                    if (wrapper_benchmark) {
                        const double wrapper_charge = (
                            masked_value(wrapper, stamp_mask) + masked_value(wrapper, ivafe_mask)
                        ) * wealth_tax_rate;
                        deduct_wrapper_charge(wrapper, wrapper_basis, wrapper_charge);
                    }
                }

                gross_wealth[static_cast<std::size_t>(period) * paths + path] = sum_values(gross);
                diy_wealth[static_cast<std::size_t>(period) * paths + path] = sum_values(holdings);
                if (wrapper_benchmark) {
                    const double wrapper_value = sum_values(wrapper);
                    const double denominator = wrapper_previous_value + contribution;
                    const double numerator = wrapper_value + wrapper_funded_withdrawal;
                    if (denominator > 0.0 && numerator > 0.0) {
                        wrapper_return_product *= numerator / denominator;
                        ++wrapper_return_count;
                    }
                    wrapper_previous_value = wrapper_value;
                }
            }

            const int final_slot = year_slots[periods - 1];
            if (managed) {
                settle_managed_year(final_slot);
                diy_wealth[static_cast<std::size_t>(periods - 1) * paths + path] = sum_values(holdings);
            } else if (terminal_liquidation) {
                rebalance_sales = holdings;
                const SaleResult event = settle_sales(
                    rebalance_sales,
                    holdings,
                    basis,
                    taxable_fraction,
                    offsettable,
                    losses,
                    0.0,
                    false
                );
                terminal_tax += event.tax;
                realized_gains += event.gains;
                realized_losses += event.losses;
                diy_wealth[static_cast<std::size_t>(periods - 1) * paths + path] = std::max(
                    diy_wealth[static_cast<std::size_t>(periods - 1) * paths + path]
                        - event.tax - (declarative ? pending_tax : 0.0),
                    0.0
                );
                pending_tax = 0.0;
                add_year_stat(local_year_stats, final_slot, kYearTerminalTax, event.tax);
            } else if (declarative && pending_tax > 0.0) {
                deduct_charge(holdings, basis, pending_tax);
                add_year_stat(
                    local_year_stats,
                    final_slot,
                    kYearDeferredTaxPayment,
                    pending_tax
                );
                pending_tax = 0.0;
                diy_wealth[static_cast<std::size_t>(periods - 1) * paths + path] = sum_values(holdings);
            }

            if (wrapper_benchmark) {
                const double before_tax = sum_values(wrapper);
                const double gain = std::max(before_tax - wrapper_basis, 0.0);
                const double tax = gain * wrapper_taxable_fraction(wrapper, taxable_fraction)
                    * kItalianTaxRate;
                const double terminal = std::max(
                    before_tax - tax - wrapper_pending_tax,
                    0.0
                );
                if (before_tax > 0.0 && terminal > 0.0) {
                    wrapper_return_product *= terminal / before_tax;
                }
                wrapper_terminal[path] = terminal;
                wrapper_annualized[path] = wrapper_return_count > 0
                    ? std::pow(
                        wrapper_return_product,
                        12.0 / wrapper_return_count
                    ) - 1.0
                    : 0.0;
            }

            const double loss_carry = sum_values(losses);
            const double taxes_paid = capital_tax + ftt_total + wealth_tax + terminal_tax;
            tax_stats[static_cast<std::size_t>(kCapitalGainsTax) * paths + path] = capital_tax;
            tax_stats[static_cast<std::size_t>(kInvestmentIncomeTax) * paths + path] = 0.0;
            tax_stats[static_cast<std::size_t>(kForeignWithholdingTax) * paths + path] = 0.0;
            tax_stats[static_cast<std::size_t>(kFinancialTransactionTax) * paths + path] = ftt_total;
            tax_stats[static_cast<std::size_t>(kWealthTax) * paths + path] = wealth_tax;
            tax_stats[static_cast<std::size_t>(kStampDuty) * paths + path] = stamp_total;
            tax_stats[static_cast<std::size_t>(kIvafe) * paths + path] = ivafe_total;
            tax_stats[static_cast<std::size_t>(kTerminalTax) * paths + path] = terminal_tax;
            tax_stats[static_cast<std::size_t>(kTaxesPaid) * paths + path] = taxes_paid;
            tax_stats[static_cast<std::size_t>(kRealizedGains) * paths + path] = realized_gains;
            tax_stats[static_cast<std::size_t>(kRealizedLosses) * paths + path] = realized_losses;
            tax_stats[static_cast<std::size_t>(kLossCarryforward) * paths + path] = loss_carry;
            tax_stats[static_cast<std::size_t>(kExpiredLosses) * paths + path] = expired_losses;
            tax_stats[static_cast<std::size_t>(kTransactionCosts) * paths + path] = transaction_cost_total;
            gross_transaction_costs[path] = gross_cost_total;
        }
        if (paths == 1) {
            std::copy(local_year_stats.begin(), local_year_stats.end(), year_stats);
        } else {
            std::lock_guard<std::mutex> lock(year_mutex);
            for (std::size_t index = 0; index < local_year_stats.size(); ++index) {
                year_stats[index] += local_year_stats[index];
            }
        }
    });
    return 0;
}

struct MCParametricPortfolioConfig {
    int periods;
    int paths;
    int assets;
    int states;
    int macro_dimensions;
    int requested_threads;
    const std::uint8_t *regimes;
    const double *means;
    const double *gaussian_correlation_cholesky;
    const double *gaussian_correlations;
    const double *volatilities;
    const double *tail_indexes;
    const double *temperings;
    const double *skewness;
    const double *gaussian_scales;
    const double *macro_shocks;
    const double *macro_betas;
    std::uint64_t seed;
    int garch;
    double garch_alpha;
    double garch_beta;
    int dynamic_correlation;
    double dcc_alpha;
    double dcc_beta;
    double dcc_asymmetry;
    const double *monthly_fee_log;
    int simple_returns;
};

struct MCItalianPortfolioConfig {
    const double *weights;
    double initial_value;
    int rebalance_frequency;
    const double *transaction_cost_rate_paths;
    double default_transaction_cost_rate;
    double contribution;
    int contribution_mode;
    double withdrawal;
    int withdrawal_start_period;
    int tax_regime;
    const double *taxable_fraction;
    const std::uint8_t *offsettable;
    const double *ftt_rates;
    const std::uint8_t *stamp_mask;
    const std::uint8_t *ivafe_mask;
    double annual_wealth_tax;
    int terminal_liquidation;
    int wrapper_benchmark;
    const int *year_slots;
    int year_count;
};

extern "C" int mc_simulate_parametric_italian_portfolios(
    const MCParametricPortfolioConfig *parametric,
    const MCItalianPortfolioConfig *tax,
    double *gross_wealth,
    double *diy_wealth,
    double *wrapper_terminal,
    double *wrapper_annualized,
    double *tax_stats,
    double *gross_transaction_costs,
    double *year_stats
) {
    if (!parametric || !tax || !parametric->regimes || !parametric->means ||
        !parametric->gaussian_correlation_cholesky || !parametric->gaussian_correlations ||
        !parametric->volatilities || !parametric->tail_indexes || !parametric->temperings ||
        !parametric->skewness || !parametric->gaussian_scales ||
        !parametric->monthly_fee_log || !tax->weights || !tax->taxable_fraction ||
        !tax->offsettable || !tax->ftt_rates || !tax->stamp_mask || !tax->ivafe_mask ||
        !tax->year_slots || !gross_wealth || !diy_wealth || !tax_stats ||
        !gross_transaction_costs || !year_stats) return 1;
    const int periods = parametric->periods;
    const int paths = parametric->paths;
    const int assets = parametric->assets;
    if (periods <= 0 || paths <= 0 || assets <= 0 || parametric->states <= 0 ||
        tax->year_count <= 0) return 2;

    std::fill(
        year_stats,
        year_stats + static_cast<std::size_t>(tax->year_count) * kYearStatCount,
        0.0
    );
    std::atomic<int> failure{0};
    std::mutex year_mutex;
    parallel_paths(paths, parametric->requested_threads, [&](int begin, int end) {
        std::vector<double> local_year_stats(
            static_cast<std::size_t>(tax->year_count) * kYearStatCount,
            0.0
        );
        std::vector<std::uint8_t> path_regimes(static_cast<std::size_t>(periods));
        std::vector<double> path_macro(
            static_cast<std::size_t>(periods) * parametric->macro_dimensions
        );
        std::vector<double> path_returns(
            static_cast<std::size_t>(periods) * assets,
            0.0
        );
        std::vector<double> path_growth(path_returns.size(), 0.0);
        std::vector<double> path_cost_rates(
            tax->transaction_cost_rate_paths ? static_cast<std::size_t>(periods) : 0U
        );
        std::vector<double> path_gross(static_cast<std::size_t>(periods), 0.0);
        std::vector<double> path_diy(static_cast<std::size_t>(periods), 0.0);
        std::vector<double> path_wrapper_terminal(tax->wrapper_benchmark ? 1U : 0U);
        std::vector<double> path_wrapper_annualized(tax->wrapper_benchmark ? 1U : 0U);
        std::vector<double> path_tax_stats(kTaxStatCount, 0.0);
        std::vector<double> path_year_stats(
            static_cast<std::size_t>(tax->year_count) * kYearStatCount,
            0.0
        );
        for (int path = begin; path < end; ++path) {
            if (failure.load(std::memory_order_relaxed) != 0) continue;
            for (int period = 0; period < periods; ++period) {
                const std::size_t source = static_cast<std::size_t>(period) * paths + path;
                path_regimes[period] = parametric->regimes[source];
                for (int dimension = 0; dimension < parametric->macro_dimensions; ++dimension) {
                    path_macro[
                        static_cast<std::size_t>(period) * parametric->macro_dimensions + dimension
                    ] = parametric->macro_shocks[
                        source * parametric->macro_dimensions + dimension
                    ];
                }
            }

            constexpr std::uint64_t stream_constant = 0x9e3779b97f4a7c15ULL;
            const std::uint64_t adjusted_seed = parametric->seed
                ^ (stream_constant * (static_cast<std::uint64_t>(path) + 1ULL))
                ^ stream_constant;
            const int return_status = mc_simulate_parametric(
                periods,
                1,
                assets,
                parametric->states,
                parametric->macro_dimensions,
                path_regimes.data(),
                parametric->means,
                parametric->gaussian_correlation_cholesky,
                parametric->gaussian_correlations,
                parametric->volatilities,
                parametric->tail_indexes,
                parametric->temperings,
                parametric->skewness,
                parametric->gaussian_scales,
                path_macro.empty() ? nullptr : path_macro.data(),
                parametric->macro_betas,
                adjusted_seed,
                parametric->garch,
                parametric->garch_alpha,
                parametric->garch_beta,
                parametric->dynamic_correlation,
                parametric->dcc_alpha,
                parametric->dcc_beta,
                parametric->dcc_asymmetry,
                1,
                path_returns.data()
            );
            if (return_status != 0) {
                failure.store(10 + return_status, std::memory_order_relaxed);
                continue;
            }

            bool valid_growth = true;
            for (int period = 0; period < periods; ++period) {
                for (int asset = 0; asset < assets; ++asset) {
                    const std::size_t index = static_cast<std::size_t>(period) * assets + asset;
                    const double growth = parametric->simple_returns
                        ? (1.0 + path_returns[index])
                            * std::exp(parametric->monthly_fee_log[asset])
                        : std::exp(path_returns[index] + parametric->monthly_fee_log[asset]);
                    path_growth[index] = growth;
                    valid_growth = valid_growth && std::isfinite(growth) && growth > 0.0;
                }
            }
            if (!valid_growth) {
                failure.store(20, std::memory_order_relaxed);
                continue;
            }

            if (tax->transaction_cost_rate_paths) {
                for (int period = 0; period < periods; ++period) {
                    path_cost_rates[period] = tax->transaction_cost_rate_paths[
                        static_cast<std::size_t>(period) * paths + path
                    ];
                }
            }
            double path_gross_cost = 0.0;
            const int ledger_status = mc_simulate_italian_portfolios(
                periods,
                1,
                assets,
                1,
                path_growth.data(),
                tax->weights,
                tax->initial_value,
                tax->rebalance_frequency,
                path_cost_rates.empty() ? nullptr : path_cost_rates.data(),
                tax->default_transaction_cost_rate,
                tax->contribution,
                tax->contribution_mode,
                tax->withdrawal,
                tax->withdrawal_start_period,
                0,
                0,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                0,
                0.0,
                nullptr,
                nullptr,
                0,
                12,
                1.20,
                0.80,
                0.10,
                0.70,
                1.30,
                1,
                tax->tax_regime,
                tax->taxable_fraction,
                tax->offsettable,
                tax->ftt_rates,
                tax->stamp_mask,
                tax->ivafe_mask,
                tax->annual_wealth_tax,
                tax->terminal_liquidation,
                tax->wrapper_benchmark,
                tax->year_slots,
                tax->year_count,
                path_gross.data(),
                path_diy.data(),
                path_wrapper_terminal.empty() ? nullptr : path_wrapper_terminal.data(),
                path_wrapper_annualized.empty() ? nullptr : path_wrapper_annualized.data(),
                path_tax_stats.data(),
                &path_gross_cost,
                path_year_stats.data(),
                nullptr,
                nullptr,
                nullptr
            );
            if (ledger_status != 0) {
                failure.store(30 + ledger_status, std::memory_order_relaxed);
                continue;
            }

            for (int period = 0; period < periods; ++period) {
                const std::size_t destination = static_cast<std::size_t>(period) * paths + path;
                gross_wealth[destination] = path_gross[period];
                diy_wealth[destination] = path_diy[period];
            }
            for (int stat = 0; stat < kTaxStatCount; ++stat) {
                tax_stats[static_cast<std::size_t>(stat) * paths + path] = path_tax_stats[stat];
            }
            gross_transaction_costs[path] = path_gross_cost;
            if (tax->wrapper_benchmark) {
                wrapper_terminal[path] = path_wrapper_terminal[0];
                wrapper_annualized[path] = path_wrapper_annualized[0];
            }
            for (std::size_t index = 0; index < path_year_stats.size(); ++index) {
                local_year_stats[index] += path_year_stats[index];
            }
        }
        std::lock_guard<std::mutex> lock(year_mutex);
        for (std::size_t index = 0; index < local_year_stats.size(); ++index) {
            year_stats[index] += local_year_stats[index];
        }
    });
    return failure.load(std::memory_order_relaxed);
}

extern "C" int mc_native_version() { return 4; }
