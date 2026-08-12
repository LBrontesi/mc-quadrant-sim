#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

    double chi_squared(double degrees_of_freedom) {
        return 2.0 * gamma(degrees_of_freedom / 2.0);
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

    double gamma(double shape) {
        if (shape < 1.0) {
            const double draw = std::max(uniform(), std::numeric_limits<double>::min());
            return gamma(shape + 1.0) * std::pow(draw, 1.0 / shape);
        }
        const double d = shape - 1.0 / 3.0;
        const double c = 1.0 / std::sqrt(9.0 * d);
        while (true) {
            const double x = normal();
            const double base = 1.0 + c * x;
            if (base <= 0.0) continue;
            const double v = base * base * base;
            const double u = uniform();
            if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
            if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
        }
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

}  // namespace

extern "C" int mc_simulate_parametric(
    int periods,
    int paths,
    int assets,
    int states,
    int macro_dimensions,
    const std::uint8_t *regimes,
    const double *means,
    const double *covariance_cholesky,
    const double *correlation_cholesky,
    const double *base_correlations,
    const double *volatilities,
    const double *macro_shocks,
    const double *macro_betas,
    std::uint64_t seed,
    int student_t,
    double degrees_of_freedom,
    int garch,
    double garch_alpha,
    double garch_beta,
    int dynamic_correlation,
    double dcc_alpha,
    double dcc_beta,
    double dcc_asymmetry,
    double *output
) {
    if (periods <= 0 || paths <= 0 || assets <= 0 || states <= 0 || !regimes || !means ||
        !covariance_cholesky || !correlation_cholesky || !base_correlations ||
        !volatilities || !output) return 1;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int path = 0; path < paths; ++path) {
        RandomStream random(seed, static_cast<std::uint64_t>(path));
        std::vector<double> q(assets * assets, 0.0);
        std::vector<double> factor(assets * assets, 0.0);
        std::vector<double> previous(assets, 0.0);
        std::vector<double> independent(assets, 0.0);
        std::vector<double> standardized(assets, 0.0);
        std::vector<double> conditional_variance(assets, 0.0);
        int previous_state = -1;

        for (int period = 0; period < periods; ++period) {
            const std::size_t path_offset = static_cast<std::size_t>(period) * paths + path;
            const int state = static_cast<int>(regimes[path_offset]);
            if (state < 0 || state >= states) continue;
            const bool reanchored = previous_state != state;
            const double *base = state_matrix(base_correlations, state, assets);
            const double *covariance_factor = state_matrix(covariance_cholesky, state, assets);
            const double *correlation_factor = state_matrix(correlation_cholesky, state, assets);
            const double *state_mean = state_vector(means, state, assets);
            const double *state_volatility = state_vector(volatilities, state, assets);

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
            } else if (garch) {
                for (int row = 0; row < assets; ++row) {
                    double value = 0.0;
                    for (int column = 0; column <= row; ++column) {
                        value += correlation_factor[row * assets + column] * independent[column];
                    }
                    standardized[row] = value;
                }
            } else {
                for (int row = 0; row < assets; ++row) {
                    double value = 0.0;
                    for (int column = 0; column <= row; ++column) {
                        value += covariance_factor[row * assets + column] * independent[column];
                    }
                    standardized[row] = value;
                }
            }

            if (student_t) {
                const double scale = std::sqrt(
                    (degrees_of_freedom - 2.0) / random.chi_squared(degrees_of_freedom)
                );
                for (double &value : standardized) value *= scale;
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
                } else if (dynamic_correlation) {
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
    return 0;
}

extern "C" int mc_native_version() { return 1; }
