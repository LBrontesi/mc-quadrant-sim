# Production NTS random-generator comparison

Measured on 2026-09-30, arm64 macOS, Apple clang 21.0.0. All executables use
the production flags `-O3 -std=c++17 -pthread`, one thread, and the same Qu
subordinator implementation. The baseline is `native_simulation.cpp` from
commit `9a549744b39816b113432b368fb73c4b9a19033c` (Xoshiro256** + Box–Muller).

Production now uses `Xoshiro256** + 256-layer Ziggurat + Qu`. An otherwise
equivalent XORWOW/Ziggurat prototype was slower with 53-bit uniforms, so the
existing 64-bit generator is retained. The alternate implementation is saved
in `xorwow_random.hpp` for benchmarking only.

The normal sampler uses independent horizontal and layer draws, as described
by [Doornik (2005), section 4](https://www.doornik.com/research/ziggurat.pdf),
with eight layer indices cached per 64-bit word (four for XORWOW). Tables are
immutable and initialized once. The rare rejection and tail work is kept out
of the inlined fast path, which also accelerates Qu's gamma proposals.

Each trial warms up all modes with seed 41, then measures one million draws
with seed 42. Five trials rotate the three executable versions; each executable
also rotates the normal/subordinator/full-NTS workloads. Timing excludes
allocation, parameter setup, and output checksums. Full NTS produces four
assets sharing one unit-mean subordinator:
`X[j] = 0.5 * (T - 1) + sqrt(T) * Z[j]`.
These are marginal NTS samples with shared mixing, without GARCH, correlation
matrix multiplication, portfolio accounting, or reporting overhead.

Median milliseconds per million **four-asset vectors**:

| Tail index | Tempering | Baseline | XORWOW + Ziggurat | Xoshiro + Ziggurat | Time reduction vs baseline |
|---|---|---|---|---|---|
| 1.50 | 0.50 | 127.188 | 114.803 | 105.610 | 17.0% |
| 1.00 | 0.3535533906 | 162.133 | 149.989 | 135.371 | 16.5% |
| 1.10 | 20.00 | 110.862 | 83.647 | 77.082 | 30.5% |
| 1.90 | 0.04 | 90.640 | 77.298 | 68.090 | 24.9% |

At tail index 1.5 and tempering 0.5, standard normals improved from 6.252 to
3.152 ms/million, and subordinators from 98.853 to 90.433 ms/million.
Raw measurements, including output moments, are in `results.csv`.
These results establish the fastest of the tested variants on this host;
they are not an end-to-end application speedup or a hardware-independent claim.

Reproduce from the repository root:

```sh
c++ -O3 -std=c++17 -pthread scripts/benchmark_nts_rng.cpp -o /tmp/nts-current
/tmp/nts-current 1000000 7 1.5 0.5 4

git show 9a549744b39816b113432b368fb73c4b9a19033c:src/mc_quadrants/native_simulation.cpp > /tmp/nts-baseline.cpp
c++ -O3 -std=c++17 -pthread -Isrc/mc_quadrants \
  '-DMC_NTS_SOURCE="/tmp/nts-baseline.cpp"' scripts/benchmark_nts_rng.cpp -o /tmp/nts-baseline
/tmp/nts-baseline 1000000 7 1.5 0.5 4

sed 's/"native_random.hpp"/"xorwow_random.hpp"/' src/mc_quadrants/native_simulation.cpp > /tmp/nts-xorwow.cpp
c++ -O3 -std=c++17 -pthread -Isrc/mc_quadrants -Ibenchmarks/nts-rng \
  '-DMC_NTS_SOURCE="/tmp/nts-xorwow.cpp"' scripts/benchmark_nts_rng.cpp -o /tmp/nts-xorwow
/tmp/nts-xorwow 1000000 7 1.5 0.5 4
```

Arguments are samples, measured rounds, tail index, tempering, and assets.
Discard round `-1` (warmup); `ns_per_sample` measures a full vector for `nts`
and a scalar for the other modes. For the saved protocol, run each executable
with one measured round five times and rotate their order each time.

Rebuild the shared library with `./scripts/build_native.sh`. Native draw
sequences involving normals change; determinism across worker counts remains.
The ABI and Python API are unchanged. Regression tests cover normal moments,
CDFs and tails, stream independence/repeatability, concurrent table setup,
the subordinator's Laplace transform, the joint NTS characteristic function,
and detailed/compact/multithreaded simulation parity.
