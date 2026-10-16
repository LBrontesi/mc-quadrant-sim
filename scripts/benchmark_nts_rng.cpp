// Compile with the production flags, e.g.:
// c++ -O3 -std=c++17 -pthread scripts/benchmark_nts_rng.cpp -o /tmp/nts-rng
// /tmp/nts-rng [samples=1000000] [repeats=7] [tail_index=1.5] [tempering=0.5] [assets=4]
// MC_NTS_SOURCE can name a previous native_simulation.cpp snapshot for comparison.
#ifndef MC_NTS_SOURCE
#define MC_NTS_SOURCE "../src/mc_quadrants/native_simulation.cpp"
#endif
#include MC_NTS_SOURCE

#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    const int samples = argc > 1 ? std::atoi(argv[1]) : 1'000'000;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 7;
    const double tail_index = argc > 3 ? std::atof(argv[3]) : 1.5;
    const double tempering = argc > 4 ? std::atof(argv[4]) : 0.5;
    const int assets = argc > 5 ? std::atoi(argv[5]) : 4;
    if (samples < 1 || repeats < 1 || assets < 1
            || !(tail_index > 0.0 && tail_index < 2.0)
            || !(tempering > 0.0 && std::isfinite(tempering))) return 1;
    const auto parameters = make_nts_subordinator_parameters(tail_index, tempering);
    std::vector<double> output(static_cast<std::size_t>(samples) * assets);
    const char *names[] = {"normal", "subordinator", "nts"};
    std::cout << "mode,round,samples,assets,tail_index,tempering,ns_per_sample,mean,second_moment\n";
    std::cout << std::setprecision(12);
    // Warm all modes, rotate execution order, and consume every output after
    // timing. Setup/allocation/checksums are excluded from generation time.
    for (int round = -1; round < repeats; ++round) {
        for (int offset = 0; offset < 3; ++offset) {
            const int mode = (round + 1 + offset) % 3;
            const int width = mode == 2 ? assets : 1;
            RandomStream random(static_cast<std::uint64_t>(42 + round), 0);
            const auto started = std::chrono::steady_clock::now();
            for (int index = 0; index < samples; ++index) {
                if (mode == 0) output[index] = random.normal();
                else {
                    const double subordinator = nts_subordinator(random, parameters);
                    if (mode == 1) output[index] = subordinator;
                    else {
                        const double scale = std::sqrt(subordinator);
                        const double location = 0.5 * (subordinator - 1.0);
                        for (int asset = 0; asset < assets; ++asset) {
                            output[static_cast<std::size_t>(index) * assets + asset] =
                                location + scale * random.normal();
                        }
                    }
                }
            }
            const double nanos = std::chrono::duration<double, std::nano>(
                std::chrono::steady_clock::now() - started).count();
            double sum = 0.0, square = 0.0;
            const auto count = static_cast<std::size_t>(samples) * width;
            for (std::size_t index = 0; index < count; ++index) {
                const double value = output[index];
                if (!std::isfinite(value)) return 2;
                sum += value;
                square += value * value;
            }
            std::cout << names[mode] << ',' << round << ',' << samples << ',' << width
                << ',' << tail_index << ',' << tempering << ',' << nanos / samples
                << ',' << sum / count << ',' << square / count << '\n';
        }
    }
}
