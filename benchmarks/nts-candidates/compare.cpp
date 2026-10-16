#include MC_NTS_SOURCE
#include "batched_qu.hpp"

#include <complex>
#include <iomanip>
#include <iostream>
#include <string>

struct Moments {
    double sum = 0.0, square = 0.0;
    void add(double value) { sum += value; square += value * value; }
    double z_score(double expected, int count) const {
        const double mean = sum / count;
        const double variance = std::max(0.0, square / count - mean * mean);
        return std::abs(mean - expected) / std::max(1e-12, std::sqrt(variance / count));
    }
};

template <typename Draw>
void measure(Draw draw, RandomStream &random, const NTSSubordinatorParameters &p,
             int samples, int assets, bool validate) {
    std::vector<double> mixing(samples), output(static_cast<std::size_t>(samples) * assets);
    const double beta[4] = {0.5, -0.2, 0.1, 0.3};
    for (int i = 0; i < 4096; ++i) {
        draw();
        for (int j = 0; j < assets; ++j) random.normal();
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < samples; ++i) {
        const double t = draw();
        mixing[i] = t;
        const double scale = std::sqrt(t);
        for (int j = 0; j < assets; ++j) {
            output[static_cast<std::size_t>(i) * assets + j] = beta[j] * (t - 1.0) + scale * random.normal();
        }
    }
    const double ns = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - start).count() / samples;
    double checksum = 0.0;
    for (double value : output) {
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite NTS sample");
        checksum += value;
    }
    double max_z = 0.0;
    Moments mean;
    for (double t : mixing) {
        if (!std::isfinite(t) || t <= 0.0) throw std::runtime_error("invalid subordinator");
        mean.add(t);
    }
    if (validate) {
        const double alpha = p.qu.alpha, tilt = p.qu.lambda;
        for (double s : {0.1, 1.0, 10.0}) {
            Moments laplace;
            for (double t : mixing) laplace.add(std::exp(-s * t));
            const double expected = std::exp(std::pow(tilt, alpha) - std::pow(tilt + p.scale * s, alpha));
            max_z = std::max(max_z, laplace.z_score(expected, samples));
        }
        const double direction[4] = {0.6, -0.9, 0.3, 0.5};
        for (double multiplier : {0.2, 1.0, 2.0}) {
            double drift = 0.0, variance = 0.0;
            for (int j = 0; j < assets; ++j) {
                const double u = direction[j] * multiplier;
                drift += beta[j] * u;
                variance += u * u;
            }
            const std::complex<double> expected = std::exp(std::complex<double>(0.0, -drift)
                + std::pow(tilt, alpha)
                - std::pow(tilt + p.scale * std::complex<double>(0.5 * variance, -drift), alpha));
            Moments real, imaginary;
            for (int i = 0; i < samples; ++i) {
                double phase = 0.0;
                for (int j = 0; j < assets; ++j) {
                    phase += output[static_cast<std::size_t>(i) * assets + j] * direction[j] * multiplier;
                }
                real.add(std::cos(phase));
                imaginary.add(std::sin(phase));
            }
            max_z = std::max({max_z, real.z_score(expected.real(), samples),
                imaginary.z_score(expected.imag(), samples)});
        }
    }
    std::cout << std::setprecision(12) << "{\"ns_per_vector\":" << ns
        << ",\"checksum\":" << checksum << ",\"mean_t\":" << mean.sum / samples
        << ",\"max_z\":" << max_z << ",\"envelope\":" << p.qu.envelope << "}\n";
    if (max_z > 7.0) throw std::runtime_error("distribution check exceeds seven standard errors");
}

int main(int argc, char **argv) {
    if (argc != 8) return 1;
    const std::string mode = argv[1];
    const double tail = std::stod(argv[2]), tempering = std::stod(argv[3]);
    const int samples = std::stoi(argv[4]), assets = std::stoi(argv[6]);
    if (samples < 1 || (assets != 1 && assets != 4) || !(tail > 0.0 && tail < 2.0)
            || !(tempering > 0.0 && std::isfinite(tempering))) return 1;
    auto parameters = make_nts_subordinator_parameters(tail, tempering);
    RandomStream random(std::stoull(argv[5]), 0);
    const bool validate = std::stoi(argv[7]) != 0;
    if (mode == "info") {
        std::cout << "{\"envelope\":" << parameters.qu.envelope << ",\"log_constants\":[";
        for (int i = 0; i < 4; ++i) std::cout << (i ? "," : "") << parameters.qu.log_constants[i];
        std::cout << "]}\n";
    } else if (mode == "batch64" || mode == "batch256") {
        if (mode == "batch64") {
            BatchedQu<64, false> batch(random, parameters);
            measure([&] { return batch.draw(); }, random, parameters, samples, assets, validate);
        } else {
            BatchedQu<256, false> batch(random, parameters);
            measure([&] { return batch.draw(); }, random, parameters, samples, assets, validate);
        }
#ifdef __APPLE__
    } else if (mode == "vforce64" || mode == "vforce256") {
        if (mode == "vforce64") {
            BatchedQu<64, true> batch(random, parameters);
            measure([&] { return batch.draw(); }, random, parameters, samples, assets, validate);
        } else {
            BatchedQu<256, true> batch(random, parameters);
            measure([&] { return batch.draw(); }, random, parameters, samples, assets, validate);
        }
#endif
    } else if (mode == "ig") {
        if (tail != 1.0) return 1;
        // Michael, Schucany & Haas: IG(mean=1, shape=2*tempering).
        // Rationalize the smaller quadratic root to avoid cancellation.
        measure([&] {
            const double z = random.normal();
            const double a = z * z / (4.0 * tempering);
            const double small = 1.0 / (1.0 + a + std::sqrt(a * (a + 2.0)));
            return random.uniform() <= 1.0 / (1.0 + small) ? small : 1.0 / small;
        }, random, parameters, samples, assets, validate);
    } else {
        if (mode.size() == 4 && mode.substr(0, 3) == "env" && mode[3] >= '0' && mode[3] <= '3') {
            parameters.qu.envelope = mode[3] - '0';
        } else if (mode != "scalar") return 1;
        measure([&] { return nts_subordinator(random, parameters); },
            random, parameters, samples, assets, validate);
    }
}
