#include <iostream>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <string>

constexpr double pi = 3.1415926535897932;

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

struct GammaShapeParameters {
    double shape;
    double d;
    double c;
};

GammaShapeParameters make_gamma_shape_parameters(double shape) {
    const double d = (shape < 1.0 ? shape + 1.0 : shape) - 1.0 / 3.0;
    return {shape, d, 1.0 / std::sqrt(9.0 * d)};
}

double gamma_proposal(RandomStream &random, double d, double c) {
    while (true) {
        const double normal = random.normal();
        const double base = 1.0 + c * normal;
        if (base <= 0.0) continue;
        const double proposal = base * base * base;
        const double uniform = random.uniform();
        const double normal_square = normal * normal;
        if (uniform < 1.0 - 0.0331 * normal_square * normal_square) {
            return d * proposal;
        }
        if (std::log(std::max(uniform, std::numeric_limits<double>::min()))
                < 0.5 * normal_square + d * (1.0 - proposal + std::log(proposal))) {
            return d * proposal;
        }
    }
}

double gamma_shape(RandomStream &random, const GammaShapeParameters &parameters) {
    // Marsaglia and Tsang's generator with the same shape-boost transformation
    // and random draw order. Shape-dependent setup is done once per regime.
    if (parameters.shape < 1.0) {
        while (true) {
            const double boosted = gamma_proposal(random, parameters.d, parameters.c);
            const double log_scale = std::log(
                std::max(random.uniform(), std::numeric_limits<double>::min())
            ) / parameters.shape;
            const double value = boosted * std::exp(log_scale);
            if (value > 0.0 && std::isfinite(value)) return value;
        }
    }
    return gamma_proposal(random, parameters.d, parameters.c);
}

struct Devroye {
    double alpha;
    double one_minus_alpha;
    double lambda;
    double inv_alpha;
    double inv_one_minus_alpha;
    double neg_inv_one_minus_alpha;
    double lambda_alpha;
    double gamma;
    double gamma_sqrt;
    double gamma_power;
    double eps;
    double phi;
    double w1;
    double w2;
    double w3;
    double prob_w1_w2;
    double prob_w2_w3;
    double b;
    double b_lambda;
    double inv_B0;
    double meanT;
    double inv_meanT;
    double sqrt_pi_over_2;
    double rho_gamma_coeff;
    double mix12_coeff;
    bool gamma_ge_one;
};

// Immutable sampler parameters. RandomStream owns all mutable RNG state.
struct Qu {
    double alpha, lambda, lambda_alpha, x, y;
    double complement, inv_alpha, inv_complement, power;
    double inv_meanT, sigma, precision, lambda_power;
    double log_constant, log_x_factor, log_z_factor, log_normal_adjustment;
    int envelope;
    GammaShapeParameters gamma;
};

/* computes values which can be precomputed for Devroje algorithm sued to generate the tilted stable */
void initDevroye(Devroye& p, double lambda, double alpha) {
    p.alpha = alpha;
    p.one_minus_alpha = 1.0 - alpha;
    p.lambda = lambda;

    p.inv_alpha = 1.0 / alpha;
    p.inv_one_minus_alpha = 1.0 / p.one_minus_alpha;
    p.neg_inv_one_minus_alpha = -p.inv_one_minus_alpha;

    p.lambda_alpha = std::pow(lambda, alpha);
    p.gamma = p.lambda_alpha * alpha * p.one_minus_alpha;
    p.gamma_sqrt = std::sqrt(p.gamma);
    p.gamma_power = std::pow(p.gamma_sqrt, p.inv_alpha);

    p.sqrt_pi_over_2 = std::sqrt(pi / 2.0);

    p.eps = ((2.0 + p.sqrt_pi_over_2) * std::sqrt(2.0 * p.gamma) + 1.0) / pi;
    p.phi = (1.0 / pi) * std::exp(-(p.gamma * pi * pi) / 8.0) * (2.0 + p.sqrt_pi_over_2) * std::sqrt(p.gamma * pi);

    p.w1 = p.eps * std::sqrt(pi / (2.0 * p.gamma));
    p.w2 = 2.0 * p.phi * std::sqrt(pi);
    p.w3 = p.eps * pi;

    p.prob_w1_w2 = p.w1 / (p.w1 + p.w2);
    p.prob_w2_w3 = p.w3 / (p.w2 + p.w3);

    p.b = p.one_minus_alpha / alpha;
    p.b_lambda = p.b * lambda;

    p.inv_B0 = std::pow(alpha, alpha) * std::pow(p.one_minus_alpha, p.one_minus_alpha);

    p.meanT = alpha * std::pow(lambda, alpha - 1.0);
    p.inv_meanT = 1.0 / p.meanT;

    p.rho_gamma_coeff = (1.0 + p.sqrt_pi_over_2) * p.gamma_sqrt;
    p.mix12_coeff = p.sqrt_pi_over_2 + 1.0;

    p.gamma_ge_one = p.gamma >= 1.0;
}

/* computes values which can be precomputed for Qu algorithm sued to generate the tilted stable */
void initQu(Qu& p, double lambda, double a) {
    if (!(a > 0 && a < 1) || !(lambda > 0) || !std::isfinite(lambda))
        throw std::invalid_argument("Require 0 < alpha < 1 and finite lambda > 0");
    p.alpha = a;
    p.lambda = lambda;
    p.complement = 1 - a;
    p.inv_alpha = 1 / a;
    p.inv_complement = 1 / p.complement;
    p.power = a / p.complement;
    p.lambda_alpha = std::pow(lambda, a);
    p.x = a * p.lambda_alpha;
    p.y = p.complement * p.lambda_alpha;
    p.precision = a * p.complement * p.lambda_alpha;
    p.sigma = 1 / std::sqrt(p.precision);
    p.inv_meanT = 1 / (a * std::pow(lambda, a - 1));
    const double log_denominator = .5 * std::log(2 * pi * p.precision);
    // Log envelopes avoid overflow from tgamma and exp at large tempering.
    const double c[] = {
        std::lgamma(p.x) + p.x - 1 - p.lambda_alpha * std::log(p.x)
            + (p.y + 1) * std::log(p.power + p.x),
        std::lgamma(p.y + 1) + p.y - p.y * std::log(p.y),
        std::lgamma(p.x + 1) + p.x - 1 - p.x * std::log(p.x)
            - log_denominator + (p.y + 1) * std::log1p(1 / p.y),
        std::lgamma(p.y + 1) + p.y - p.y * std::log(p.y) - log_denominator
    };
    p.envelope = static_cast<int>(std::min_element(c, c + 4) - c);
    p.log_constant = c[p.envelope];
    p.lambda_power = p.power * std::log(lambda);
    p.log_x_factor = std::log(a / p.complement) + p.lambda_alpha
        + std::lgamma(p.x) + p.lambda_power;
    p.log_z_factor = p.lambda_alpha + std::lgamma(p.y + 1);
    p.log_normal_adjustment = std::log(std::erf(pi / (p.sigma * std::sqrt(2.0))))
        - log_denominator;
    p.gamma = make_gamma_shape_parameters(
        p.envelope == 0 || p.envelope == 2 ? p.x : p.y + 1);
    if (!std::isfinite(p.log_constant) || !std::isfinite(p.inv_meanT)
        || !std::isfinite(p.log_x_factor) || !std::isfinite(p.log_z_factor))
        throw std::invalid_argument("Parameters exceed supported floating-point range");
}

// Exclude zero where uniforms feed logarithms.
inline double uniform01(RandomStream& s) {
    return std::max(s.uniform(), std::numeric_limits<double>::min());
}

/* Qu algorithm used to generate the T in the NTS distribution as discussed in the paper RandomVariate Generationfor Exponential and Gamma
Tilted Stable Distributions Qu 2021. Algorithm is based on 2 dimensional Single Rejection*/
double tilted_tempered_stable_Qu(RandomStream& s, const Qu& p) {
    for (;;) {
        double angle;
        if (p.envelope < 2) {
            angle = pi * uniform01(s);
        } else {
            // Folding the symmetric normal avoids rejecting negative proposals.
            do { angle = std::abs(s.normal()) * p.sigma; }
            while (angle <= 0 || angle >= pi);
        }
        const double log_b = p.alpha * std::log(std::sin(p.alpha * angle))
            + p.complement * std::log(std::sin(p.complement * angle))
            - std::log(std::sin(angle));
        const double proposal = gamma_shape(s, p.gamma);
        if (!(proposal > 0) || !std::isfinite(proposal)) continue;
        const double log_proposal = std::log(proposal);
        double result, log_ratio;
        if (p.envelope == 0 || p.envelope == 2) {
            const double log_term = log_b * p.inv_complement;
            result = proposal / p.lambda;
            log_ratio = p.log_x_factor + log_term
                - (p.power + p.x) * log_proposal
                - std::exp(log_term + p.lambda_power - p.power * log_proposal);
        } else {
            result = std::exp((log_b - p.complement * log_proposal) * p.inv_alpha);
            log_ratio = p.log_z_factor - p.y * log_proposal - p.lambda * result;
        }
        if (p.envelope >= 2)
            log_ratio += p.log_normal_adjustment + .5 * p.precision * angle * angle;
        if (result > 0 && std::isfinite(result)
            && std::log(uniform01(s)) <= std::min(0.0, log_ratio - p.log_constant))
            return result;
    }
}

/* Devroje algorithm used to generate the T in the NTS distribution as discussed in the paper RANDOM VARIATE GENERATION 
FOR EXPONENTIALLY AND POLYNOMIALLY TILTED STABLE DISTRIBUTIONS Devroje . Algorithm is based on Double Rejection algorithm.
This is the main reason Qu is faster*/
double tilted_tempered_stable_Devroye(RandomStream& s, const Devroye& p) {
    double U;
    double z;
    double Z;
    double BU;

    for (;;) {

        // First rejection stage
        for (;;) {
            double V = uniform01(s);
            double W1 = uniform01(s);

            if (p.gamma_ge_one) {
                if (V < p.prob_w1_w2) U = std::abs(s.normal()) / p.gamma_sqrt;
                else U = pi * (1.0 - W1 * W1);
            } else {
                if (V < p.prob_w2_w3) U = pi * W1;
                else U = pi * (1.0 - W1 * W1);
            }

            if (U >= pi) continue;

            double W = uniform01(s);

            BU = std::sin(U) /
                 (std::pow(std::sin(p.alpha * U), p.alpha) *
                  std::pow(std::sin(p.one_minus_alpha * U), p.one_minus_alpha));

            double c = std::sqrt(BU * p.inv_B0);
            double vi = std::pow(p.gamma_sqrt + p.alpha * c, p.inv_alpha);

            z = vi / (vi - p.gamma_power);

            double gss;

            if (p.gamma_ge_one) gss = p.eps * std::exp(-p.gamma * U * U / 2.0);
            else gss = p.eps;

            if (U > 0.0) gss += p.phi / std::sqrt(pi - U);

            double rho = pi * std::exp(-p.lambda_alpha * (1.0 - 1.0 / (c * c))) * gss /
                         (p.rho_gamma_coeff / c + z);

            Z = W * rho;

            if (Z <= 1.0) break;
        }

        double a = std::pow(BU, p.neg_inv_one_minus_alpha);
        double m = std::pow(p.b_lambda / a, p.alpha);
        double h1 = std::sqrt((m * p.alpha) / a);

        double a1 = h1 * p.sqrt_pi_over_2;
        double a3 = z / a;
        double s1 = h1 * p.mix12_coeff + a3;

        double V1 = uniform01(s);
        double E1 = 0.0;
        double N1 = 0.0;
        double X;

        if (V1 < a1 / s1) {
            N1 = s.normal();
            X = m - h1 * std::abs(N1);
        } else if (V1 < (h1 * p.mix12_coeff) / s1) {
            X = m + h1 * uniform01(s);
        } else {
            E1 = -std::log(uniform01(s));
            X = m + h1 + a3 * E1;
        }

        if (X <= 0.0) continue;

        double E = -std::log(Z);
        double X_neg_b = std::pow(X, -p.b);
        double m_neg_b = std::pow(m, -p.b);
        double gf = a * (X - m) + p.lambda * (X_neg_b - m_neg_b);

        if (X < m) gf -= (N1 * N1) / 2.0;
        if (X > m + h1) gf -= E1;

        if (gf <= E) return X_neg_b;
    }
}

/*Union of tilted simulation Devroje and a cached Box–Muller normal draw. T is normalized to have expected value 1 */
double NTSDevroye(RandomStream& state, const Devroye& p, double beta, double mu, double sigma) {
    double T = tilted_tempered_stable_Devroye(state, p);
    double T_scaled = T * p.inv_meanT;
    double Z = state.normal();

    return mu + beta * (T_scaled - 1.0) + sigma * std::sqrt(T_scaled) * Z;
}

/*Union of tilted simulation Qu and a cached Box–Muller normal draw. T is normalized to have expected value 1 */
double NTSQu(RandomStream& state, const Qu& p, double beta, double mu, double sigma) {
    double T = tilted_tempered_stable_Qu(state, p);
    double T_scaled = T * p.inv_meanT;
    double Z = state.normal();
    return mu + beta * (T_scaled - 1.0) + sigma * std::sqrt(T_scaled) * Z;
}

// Standalone comparison: setup and statistics are outside the timed loops.
int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: ./nts <seed> [samples=1000000] [alpha=0.5] [lambda=0.5] [repeats=5]\n";
            return 1;
        }
        const uint64_t seed = std::stoull(argv[1]);
        const int count = argc > 2 ? std::stoi(argv[2]) : 1000000;
        const double alpha = argc > 3 ? std::stod(argv[3]) : .5;
        const double lambda = argc > 4 ? std::stod(argv[4]) : .5;
        const int repeats = argc > 5 ? std::stoi(argv[5]) : 5;
        if (count <= 0 || repeats <= 0) throw std::invalid_argument("Counts must be positive");
        Qu q;
        initQu(q, lambda, alpha);
        Devroye p;
        initDevroye(p, lambda, alpha);
        std::vector<double> numbers(count);
        std::vector<double> times[2];
        std::cout << std::setprecision(9);
        std::cout << "alpha=" << alpha << " lambda=" << lambda << " Qu envelope=" << q.envelope + 1 << '\n';
        // One unreported warmup round, then alternating algorithm order.
        for (int round = -1; round < repeats; ++round) {
            for (int slot = 0; slot < 2; ++slot) {
                const int algorithm = (slot + round + 1) % 2;
                RandomStream state(seed + static_cast<uint64_t>(round + 1), 0);
                const auto start = std::chrono::steady_clock::now();
                if (algorithm == 0) {
                    for (double& value : numbers) value = NTSDevroye(state, p, .5, 0, 1);
                } else {
                    for (double& value : numbers) value = NTSQu(state, q, .5, 0, 1);
                }
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                double mean = 0, m2 = 0;
                size_t n = 0;
                for (double value : numbers) {
                    if (!std::isfinite(value)) throw std::runtime_error("Non-finite output");
                    const double delta = value - mean;
                    mean += delta / ++n;
                    m2 += delta * (value - mean);
                }
                if (round >= 0) times[algorithm].push_back(ms);
                // Printing warmup statistics also ensures its output is consumed.
                std::cout << (algorithm == 0 ? "Devroye" : "Qu")
                    << " round=" << round << " samples=" << count << " ms=" << ms
                    << " mean=" << mean << " variance=" << m2 / count << '\n';
            }
        }
        for (int algorithm = 0; algorithm < 2; ++algorithm) {
            auto& t = times[algorithm];
            std::sort(t.begin(), t.end());
            const double median = (t[(t.size()-1)/2] + t[t.size()/2]) / 2;
            std::cout << (algorithm == 0 ? "Devroye" : "Qu") << " median_ms=" << median << '\n';
        }
        std::cout << "Expected NTS mean=0 variance="
            << 1 + .25 * (1-alpha) / (alpha * std::pow(lambda, alpha)) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
