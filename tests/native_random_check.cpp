#include "native_random.hpp"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <thread>

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

int main() {
    // Exercise concurrent first use of the immutable Ziggurat tables.
    double threaded[8]{};
    std::thread workers[8];
    for (int index = 0; index < 8; ++index) {
        workers[index] = std::thread([&, index] {
            RandomStream random(42, index);
            threaded[index] = random.normal();
        });
    }
    for (auto &worker : workers) worker.join();
    for (int index = 0; index < 8; ++index) {
        RandomStream random(42, index);
        require(random.normal() == threaded[index], "thread-dependent stream");
    }

    constexpr int count = 2'000'000;
    const double boundaries[] = {-4.0, -NormalZigguratTables::tail, -3.0,
        -1.0, 0.0, 1.0, 3.0, NormalZigguratTables::tail, 4.0};
    for (std::uint64_t seed : {0ULL, 42ULL, 0xffffffffffffffffULL}) {
        RandomStream random(seed, 7), repeat(seed, 7), neighbor(seed, 8);
        RandomStream high_seed(seed ^ (1ULL << 32), 7);
        require(random.uniform() == repeat.uniform(), "uniform is not reproducible");
        require(neighbor.uniform() != high_seed.uniform(), "seed/stream collision");
        double sum = 0.0, square = 0.0, fourth = 0.0;
        double lag_products[4]{}, previous[4]{};
        int bins[9]{};
        for (int index = 0; index < count; ++index) {
            const double value = random.normal();
            require(std::isfinite(value), "nonfinite normal");
            if (index < 1024) {
                require(value == repeat.normal(), "normal is not reproducible");
                require(value != neighbor.normal(), "adjacent streams are identical");
                require(value != high_seed.normal(), "upper seed bits ignored");
            }
            sum += value;
            square += value * value;
            fourth += value * value * value * value;
            for (int lag = 3; lag >= 0; --lag) {
                lag_products[lag] += value * previous[lag];
                previous[lag] = lag ? previous[lag - 1] : value;
            }
            for (int bin = 0; bin < 9; ++bin) bins[bin] += value <= boundaries[bin];
        }
        require(std::abs(sum / count) < 6.0 / std::sqrt(count), "normal mean");
        require(std::abs(square / count - 1.0) < 6.0 * std::sqrt(2.0 / count), "normal variance");
        require(std::abs(fourth / count - 3.0) < 6.0 * std::sqrt(96.0 / count), "normal fourth moment");
        for (double product : lag_products) {
            require(std::abs(product / count) < 6.0 / std::sqrt(count), "correlated normals");
        }
        for (int bin = 0; bin < 9; ++bin) {
            const double probability = 0.5 * std::erfc(-boundaries[bin] / std::sqrt(2.0));
            const double tolerance = 6.0 * std::sqrt(probability * (1.0 - probability) / count);
            require(std::abs(double(bins[bin]) / count - probability) < tolerance, "normal CDF/tail");
        }
        double uniform_sum = 0.0, exponential_sum = 0.0;
        bool has_low_bits = false;
        for (int index = 0; index < count; ++index) {
            const double uniform = random.uniform();
            require(uniform >= 0.0 && uniform < 1.0, "uniform outside [0, 1)");
            has_low_bits |= uniform * 0x1.0p32 != std::floor(uniform * 0x1.0p32);
            uniform_sum += uniform;
            const double exponential = random.exponential();
            require(std::isfinite(exponential) && exponential > 0.0, "invalid exponential");
            exponential_sum += exponential;
        }
        require(has_low_bits, "uniform reduced to 32-bit precision");
        require(std::abs(uniform_sum / count - 0.5) < 6.0 / std::sqrt(12.0 * count), "uniform mean");
        require(std::abs(exponential_sum / count - 1.0) < 6.0 / std::sqrt(count), "exponential mean");
    }
}
