#include MC_NTS_SOURCE
#include "batched_qu.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

using Parameters = std::array<NTSSubordinatorParameters, 4>;

struct ScalarPaths {
    RandomStream &random;
    const Parameters &parameters;
    ScalarPaths(RandomStream &rng, const Parameters &p) : random(rng), parameters(p) {}
    double draw(int state) { return nts_subordinator(random, parameters[state]); }
};

template <int Size>
struct BufferedPaths {
    std::array<BatchedQu<Size, true>, 4> batches;
    BufferedPaths(RandomStream &rng, const Parameters &p)
        : batches{BatchedQu<Size, true>(rng, p[0]), BatchedQu<Size, true>(rng, p[1]),
                  BatchedQu<Size, true>(rng, p[2]), BatchedQu<Size, true>(rng, p[3])} {}
    double draw(int state) { return batches[state].draw(); }
};

template <typename Sampler>
void run(int paths, int periods, std::uint64_t seed, const Parameters &parameters) {
    std::vector<double> output(static_cast<std::size_t>(paths) * periods * 4);
    // Warm the code, but start each measured path with empty buffers. Include
    // path seeding, buffer initialization and all unused proposal work in timing.
    for (int path = 0; path < 100; ++path) {
        RandomStream random(seed, path);
        Sampler sampler(random, parameters);
        for (int period = 0; period < periods; ++period) {
            const double t = sampler.draw((period / 30) % 4);
            if (!std::isfinite(t)) std::abort();
            for (int asset = 0; asset < 4; ++asset) random.normal();
        }
    }
    const auto start = std::chrono::steady_clock::now();
    for (int path = 0; path < paths; ++path) {
        RandomStream random(seed, path);
        Sampler sampler(random, parameters);
        for (int period = 0; period < periods; ++period) {
            const int state = (period / 30) % 4;
            const double t = sampler.draw(state), scale = std::sqrt(t);
            for (int asset = 0; asset < 4; ++asset) {
                output[(static_cast<std::size_t>(path) * periods + period) * 4 + asset]
                    = 0.2 * (t - 1.0) + scale * random.normal();
            }
        }
    }
    const double ns = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - start).count() / (static_cast<double>(paths) * periods);
    double checksum = 0.0;
    for (double value : output) {
        if (!std::isfinite(value)) std::abort();
        checksum += value;
    }
    std::cout << std::setprecision(12) << "{\"ns_per_vector\":" << ns
              << ",\"checksum\":" << checksum << "}\n";
}

int main(int argc, char **argv) {
    if (argc != 5) return 1;
    const std::string mode = argv[1];
    const int paths = std::stoi(argv[2]), periods = std::stoi(argv[3]);
    if (paths < 1 || periods < 1) return 1;
    const std::uint64_t seed = std::stoull(argv[4]);
    const Parameters parameters = {make_nts_subordinator_parameters(1.7, 0.9),
        make_nts_subordinator_parameters(1.55, 0.6), make_nts_subordinator_parameters(1.3, 0.35),
        make_nts_subordinator_parameters(1.4, 0.45)};
    if (mode == "scalar") run<ScalarPaths>(paths, periods, seed, parameters);
#ifdef __APPLE__
    else if (mode == "vforce64") run<BufferedPaths<64>>(paths, periods, seed, parameters);
    else if (mode == "vforce256") run<BufferedPaths<256>>(paths, periods, seed, parameters);
#endif
    else return 1;
}
