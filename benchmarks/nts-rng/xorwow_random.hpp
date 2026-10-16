// Benchmark-only XORWOW alternative; production uses native_random.hpp.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Equal-area, 256-layer normal Ziggurat. Immutable after C++'s thread-safe
// local-static initialization; every path keeps its own RNG and layer pool.
struct NormalZigguratTables {
    static constexpr double tail = 3.6541528853610088;
    double width[256]{};
    double inner[256]{};
    double density[256]{};

    NormalZigguratTables() {
        constexpr double area = 0.004928673233974655;
        double edges[256]{};
        edges[255] = tail;
        for (int index = 254; index > 0; --index) {
            const double next = edges[index + 1];
            edges[index] = std::sqrt(-2.0 * std::log(
                area / next + std::exp(-0.5 * next * next)));
        }
        width[0] = area / std::exp(-0.5 * tail * tail);
        inner[0] = tail;
        density[0] = 1.0;
        for (int index = 1; index < 256; ++index) {
            width[index] = edges[index];
            inner[index] = edges[index - 1];
            density[index] = std::exp(-0.5 * edges[index] * edges[index]);
        }
    }

    static const NormalZigguratTables &get() {
        static const NormalZigguratTables tables;
        return tables;
    }
};

class RandomStream {
  public:
    RandomStream(std::uint64_t seed, std::uint64_t stream)
        : tables_(NormalZigguratTables::get()) {
        std::uint64_t value = seed ^ (0x9e3779b97f4a7c15ULL * (stream + 1));
        for (int index = 0; index < 6; index += 2) {
            const std::uint64_t bits = splitmix64(value);
            state_[index] = static_cast<std::uint32_t>(bits);
            state_[index + 1] = static_cast<std::uint32_t>(bits >> 32);
        }
        // XORWOW's five xorshift words must not all be zero.
        if ((state_[0] | state_[1] | state_[2] | state_[3] | state_[4]) == 0) {
            state_[0] = 1;
        }
    }

    double uniform() {
        const std::uint64_t high = next() >> 5;
        const std::uint64_t low = next() >> 6;
        return static_cast<double>((high << 26) | low) * 0x1.0p-53;
    }

    double normal() {
        // Doornik (2005), section 4: choose the layer independently of the
        // horizontal coordinate. Reusing coordinate bits for the layer biases
        // the output. Pool four independent layer indices per XORWOW word.
        // https://www.doornik.com/research/ziggurat.pdf
        const unsigned layer = next_layer();
        const double candidate = (2.0 * uniform() - 1.0) * tables_.width[layer];
        if (std::abs(candidate) < tables_.inner[layer]) return candidate;
        return normal_slow(candidate, layer);
    }

    double exponential() {
        const double draw = uniform();
        return -std::log(draw > 0.0 ? draw : std::numeric_limits<double>::min());
    }

  private:
    const NormalZigguratTables &tables_;
    std::uint32_t state_[6]{};
    std::uint32_t layer_bits_ = 0;
    unsigned layers_remaining_ = 0;

    unsigned next_layer() {
        if (layers_remaining_ == 0) {
            layer_bits_ = next();
            layers_remaining_ = 4;
        }
        const unsigned layer = layer_bits_ & 255u;
        layer_bits_ >>= 8;
        --layers_remaining_;
        return layer;
    }

    [[gnu::noinline]] double normal_slow(double candidate, unsigned layer) {
        for (;;) {
            if (layer == 0) {
                double tail, vertical;
                do {
                    tail = exponential() / NormalZigguratTables::tail;
                    vertical = exponential();
                } while (2.0 * vertical < tail * tail);
                return std::copysign(NormalZigguratTables::tail + tail, candidate);
            }
            if (uniform() * (tables_.density[layer - 1] - tables_.density[layer])
                    < std::exp(-0.5 * candidate * candidate) - tables_.density[layer]) {
                return candidate;
            }
            layer = next_layer();
            candidate = (2.0 * uniform() - 1.0) * tables_.width[layer];
            if (std::abs(candidate) < tables_.inner[layer]) return candidate;
        }
    }

    static std::uint64_t splitmix64(std::uint64_t &value) {
        std::uint64_t result = (value += 0x9e3779b97f4a7c15ULL);
        result = (result ^ (result >> 30)) * 0xbf58476d1ce4e5b9ULL;
        result = (result ^ (result >> 27)) * 0x94d049bb133111ebULL;
        return result ^ (result >> 31);
    }

    std::uint32_t next() {
        // Marsaglia's XORWOW: five-word xorshift plus a Weyl sequence.
        // https://www.jstatsoft.org/article/view/v008i14
        const std::uint32_t temporary = state_[0] ^ (state_[0] >> 2);
        const std::uint32_t result = state_[4] ^ (state_[4] << 4)
            ^ temporary ^ (temporary << 1);
        state_[0] = state_[1];
        state_[1] = state_[2];
        state_[2] = state_[3];
        state_[3] = state_[4];
        state_[4] = result;
        state_[5] += 362437u;
        return result + state_[5];
    }
};

}  // namespace
