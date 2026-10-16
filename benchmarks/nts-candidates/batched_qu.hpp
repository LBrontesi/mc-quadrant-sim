#pragma once

#include <array>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// Benchmark-only buffered Qu generator. Proposal generation remains scalar;
// independent transcendental evaluations are batched, optionally via vForce.
// Uses the same envelope and acceptance law as the production implementation.
template <int Size, bool VectorMath>
class BatchedQu {
    RandomStream &random;
    const NTSSubordinatorParameters &parameters;
    std::array<double, Size> output{}, angle{}, gamma{}, log_gamma{}, uniform{};
    std::array<double, Size> angles[3]{}, sines[3]{}, log_sinc[3]{};
    std::array<double, Size> log_b{}, exponents{}, exponentials{};
    int cursor = 0, accepted = 0;

    static void logarithms(double *out, const double *in) {
#ifdef __APPLE__
        if constexpr (VectorMath) {
            const int count = Size;
            vvlog(out, in, &count);
            return;
        }
#else
        static_assert(!VectorMath, "vForce requires macOS");
#endif
        for (int i = 0; i < Size; ++i) out[i] = std::log(in[i]);
    }

    static void sinuses(double *out, const double *in) {
#ifdef __APPLE__
        if constexpr (VectorMath) {
            const int count = Size;
            vvsin(out, in, &count);
            return;
        }
#endif
        for (int i = 0; i < Size; ++i) out[i] = std::sin(in[i]);
    }

    static void exponentiate(double *out, const double *in) {
#ifdef __APPLE__
        if constexpr (VectorMath) {
            const int count = Size;
            vvexp(out, in, &count);
            return;
        }
#endif
        for (int i = 0; i < Size; ++i) out[i] = std::exp(in[i]);
    }

    [[gnu::noinline]] void refill() {
        const auto &p = parameters.qu;
        const bool normal_angle = p.envelope >= 2;
        const bool x_envelope = (p.envelope % 2) == 0;
        const auto &shape = x_envelope ? p.gamma_x : p.gamma_z;
        cursor = accepted = 0;
        do {
            for (int i = 0; i < Size; ++i) {
                angle[i] = normal_angle
                    ? truncated_zero_mean_normal(random, p.sqrt_normal_precision, p.standardized_angle_upper)
                    : 3.14159265358979323846 * random.uniform();
                gamma[i] = gamma_shape(random, shape);
                uniform[i] = std::max(random.uniform(), std::numeric_limits<double>::min());
                angles[0][i] = angle[i];
                angles[1][i] = p.alpha * angle[i];
                angles[2][i] = p.complement * angle[i];
            }
            for (int part = 0; part < 3; ++part) {
                sinuses(sines[part].data(), angles[part].data());
                for (int i = 0; i < Size; ++i) {
                    const double value = angles[part][i];
                    const double square = value * value;
                    sines[part][i] = std::max(std::abs(value) < 1e-6
                        ? 1.0 - square / 6.0 + square * square / 120.0
                        : sines[part][i] / value, 1e-300);
                }
                logarithms(log_sinc[part].data(), sines[part].data());
            }
            logarithms(log_gamma.data(), gamma.data());
            // vForce supports in-place inputs/outputs for these functions.
            logarithms(uniform.data(), uniform.data());
            for (int i = 0; i < Size; ++i) {
                log_b[i] = p.log_b_zero - (log_sinc[0][i]
                    - p.alpha * log_sinc[1][i] - p.complement * log_sinc[2][i]);
                exponents[i] = x_envelope
                    ? log_b[i] / p.complement + p.x_power * p.log_lambda - p.x_power * log_gamma[i]
                    : log_b[i] / p.alpha - p.complement * log_gamma[i] / p.alpha;
            }
            exponentiate(exponentials.data(), exponents.data());
            for (int i = 0; i < Size; ++i) {
                double ratio = x_envelope
                    ? p.log_x_ratio_constant + log_b[i] / p.complement
                        - (p.x_power + p.gamma_shape_x) * log_gamma[i] - exponentials[i]
                    : p.log_z_ratio_constant - p.z_power * log_gamma[i] - p.lambda * exponentials[i];
                if (normal_angle) ratio += p.log_normal_adjustment
                    + 0.5 * p.normal_precision * angle[i] * angle[i];
                const double result = x_envelope ? gamma[i] / p.lambda : exponentials[i];
                if (uniform[i] <= std::min(0.0, ratio - p.log_constants[p.envelope])
                        && result > 0.0 && std::isfinite(result)) {
                    output[accepted++] = parameters.scale * result;
                }
            }
        } while (accepted == 0);
    }

  public:
    BatchedQu(RandomStream &rng, const NTSSubordinatorParameters &p) : random(rng), parameters(p) {}
    double draw() {
        if (cursor == accepted) refill();
        return output[cursor++];
    }
};
