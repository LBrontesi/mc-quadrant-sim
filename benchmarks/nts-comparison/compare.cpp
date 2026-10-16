#include <iostream>
#include <random>
#include <iomanip>
#include "../../src/mc_quadrants/native_simulation.cpp"
namespace reference {
#define main upstream_main
#include "upstream-NTS.cpp"
#undef main
#undef r
#undef v
#undef pi
}

volatile double checksum_sink = 0;

int main(int argc, char **argv) {
    const double alpha = argc > 1 ? std::stod(argv[1]) : 0.5;
    const double lambda = argc > 2 ? std::stod(argv[2]) : 0.5;
    constexpr int count = 1000000;
    std::vector<double> output(count);
    reference::compute_wki();
    reference::Qu ref_qu;
    reference::Devroye ref_dev;
    reference::initQu(ref_qu, lambda, alpha);
    reference::initDevroye(ref_dev, lambda, alpha);
    const auto local_qu = make_qu_tilted_stable_parameters(alpha, lambda);
    const auto local_dev = make_tilted_stable_parameters(alpha, lambda);
    const double inverse_mean = 1.0 / (alpha * std::pow(lambda, alpha - 1));
    std::cout << "round,implementation,kind,ms,mean,variance\n";
    // Round zero warms each implementation. Rotate order to reduce order bias.
    for (int round = 0; round < 8; ++round) {
        for (int slot = 0; slot < 6; ++slot) {
            const int variant = (slot + round) % 6;
            reference::XorwowState state;
            reference::init_xorwow(state, 7 + round);
            reference::BitPool pool{};
            reference::XorwowEngine engine{state};
            RandomStream random(7 + round, 0);
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < count; ++i) {
                switch (variant) {
                    case 0: output[i] = reference::NTSQu(state, pool, ref_qu, engine, .5, 0, 1); break;
                    case 1: {
                        const double t = exponentially_tilted_stable_qu(random, local_qu) * inverse_mean;
                        output[i] = .5 * (t - 1) + std::sqrt(t) * random.normal();
                        break;
                    }
                    case 2: output[i] = reference::tilted_tempered_stable_Qu(state, pool, ref_qu, engine) * inverse_mean; break;
                    case 3: output[i] = exponentially_tilted_stable_qu(random, local_qu) * inverse_mean; break;
                    case 4: output[i] = reference::NTSDevroye(state, pool, ref_dev, .5, 0, 1); break;
                    case 5: {
                        const double t = exponentially_tilted_stable(random, local_dev) * inverse_mean;
                        output[i] = .5 * (t - 1) + std::sqrt(t) * random.normal();
                        break;
                    }
                }
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            double sum = 0, squares = 0;
            for (double value : output) { sum += value; squares += value * value; }
            checksum_sink = sum;
            const double mean = sum / count;
            std::cout << round << ',' << (variant % 2 == 0 ? "upstream" : "local") << ','
                      << (variant < 2 ? "Qu_full" : variant < 4 ? "Qu_subordinator" : "Devroye_full")
                      << ',' << std::setprecision(10) << ms << ',' << mean << ',' << squares/count-mean*mean << '\n';
        }
    }
}
