# NTS combination experiments

Measured 2026-09-30 on arm64 macOS with Apple clang 21.0.0, one application
thread, `-O3 -std=c++17 -pthread`, no fast-math. Accelerate is linked for the
vForce variants. Production sources and the shared library were not modified.
Exact baseline source hashes and the compiler version are in `metadata.json`.

## Findings

The largest general improvement came from batching **Qu's transcendental
arithmetic**, especially with the modified normal Ziggurat. Changing the
underlying RNG alone produced a small improvement. Buffer size matters:
large buffers win bulk throughput but can lose on short individual paths.

Speedup relative to the current Xoshiro256** + 256-layer Ziggurat + Qu sampler,
for full NTS vectors with four assets, over nine parameter points:

| Candidate | Median speedup | Range across points |
|---|---:|---:|
| Xoshiro256+ + existing Ziggurat + Qu | 1.026x | 1.024–1.038x |
| Xoshiro256++ + existing Ziggurat + Qu | 1.001x | 0.997–1.018x |
| Xoshiro256** + modified Ziggurat + Qu | 1.012x | 1.007–1.030x |
| Current RNG/normal + scalar batching, 64 proposals | 1.036x | 0.967–1.172x |
| Current RNG/normal + vForce Qu, 64 proposals | 1.632x | 1.366–1.688x |
| Current RNG/normal + vForce Qu, 256 proposals | 1.673x | 1.403–1.728x |
| Xoshiro256+ + existing Ziggurat + vForce Qu, 256 | 1.697x | 1.416–1.753x |
| Xoshiro256** + modified Ziggurat + vForce Qu, 256 | **1.921x** | **1.594–1.978x** |

At tail index 1.5 and tempering 0.5, the current sampler took **104.321 ns per
vector**, versus **52.728 ns** for modified Ziggurat plus vForce batches of 256.
The ratio is 1.978x. Nanoseconds per vector numerically equal milliseconds per
million vectors. Each vector contains four asset innovations sharing one
subordinator; it is not four independent subordinators.

### Shorter paths and unused buffers

The second benchmark includes fresh RNG seeding, buffer construction, and
discarded proposals for every path. It generates 3,000 paths with four assets,
switching regimes every 30 periods through the project's four representative
parameter sets. Each path has its own stream and one buffer per regime. Seven
trials rotate candidate execution order. These timings cover innovation
generation, not GARCH, HSMM inference, correlations, taxes, or reporting.

| Periods per path | Current scalar, ns/vector | Modified Ziggurat + vForce64 | Speedup | Modified Ziggurat + vForce256 |
|---|---:|---:|---:|---:|
| 120 | 106.402 | **79.790** | **1.334x** | 290.291 |
| 360 | 105.977 | **72.883** | **1.454x** | 102.728 |

The 64-proposal candidate reduces time by **25.0% and 31.2%**, respectively.
The 256-proposal candidate is much worse at 120 periods because most proposals
are discarded. Thus the bulk winner should not simply replace each scalar
call with a large per-path buffer. Grouping proposals across active paths or
choosing smaller/adaptive buffers deserves an integration benchmark.

### Envelope choice

An independent pilot measured all envelopes whose log bound was within
`log(8)` of the smallest bound. Three 50,000-vector trials selected the lowest
median time. Candidates outside this threshold were not timed. Selection
changed at only two of the 50 pilot parameter points. Confirmation uses fresh
seeds and one million vectors per trial:

| Tail index | Tempering | Current envelope | Pilot choice | Current ns/vector | Chosen ns/vector |
|---|---:|---:|---:|---:|---:|
| 1.3 | 0.5 | 3 | 1 | 122.494 | 111.898 |
| 1.0 | 0.3535533905932738 | 3 | 1 | 137.338 | 125.968 |

Envelope numbers are zero-based, matching C++. The reduction is about 8–9%
at those points, rather than a broad improvement. Pilot setup cost is excluded
from timing; this only makes sense when amortized over repeated fixed parameters.
These choices were measured for scalar Qu and are not automatically optimal
for the batched implementation.

### Exact inverse-Gaussian special case

For `tail_index == 1.0` (subordinator alpha 0.5), the unit-mean subordinator is
inverse Gaussian with shape `2 * tempering`. The Michael–Schucany–Haas
transformation bypasses Qu, using a rationalized quadratic root for stability.
At tempering 0.3535533905932738:

| Assets | Current Qu ns/vector | Direct IG ns/vector | Speedup |
|---|---:|---:|---:|
| 1 | 128.217 | 9.792 | 13.093x |
| 4 | 137.338 | 19.748 | 6.954x |

This is an exact special-case optimization, not an approximation for nearby
tail indices, and does not accelerate the usual calibrated tail-index range
of 1.1–1.9.

## Method and correctness

- Baseline and candidates are independently compiled using the same production
  source snapshot and flags. Variant generation is confined to the build directory.
- Xoshiro256+ discards the low 16 bits when filling its layer pool (six indices
  per word); its floating-point uniforms use the high 53 bits. The other
  generators retain their existing layer selection.
- The modified normal sampler follows McFarland's interior-rectangle method,
  using 253 rectangles and an alias table for residual regions. It uses ordinary
  rejection within the selected residual wedge; it does not implement the
  paper's additional triangle squeezes. Position, layer, and sign use disjoint
  bits. All acceptance uniforms retain 53-bit precision.
- Batched Qu keeps RNG, gamma proposals, and truncated normals scalar. It
  collects 64 or 256 independent proposals, then evaluates sine, logarithm,
  and exponential arrays through scalar libm or Apple's vForce. This is
  vectorization of the numerical work, not a SIMD RNG implementation.
- Bulk confirmation uses seven trials of one million vectors, both one and
  four assets, rotating candidate order and changing seeds. Allocation,
  parameter setup, 4,096 warmup draws, and correctness reductions are outside
  the timed region. Generation includes output stores and buffer refills.
  Every output is consumed after timing.
- The 45-point calibration grid spans tail indices 1.1–1.9 and tempering
  0.04–20. Extra points exercise all four selected envelopes, actual benchmark
  regime parameters, and the inverse-Gaussian special case.
- **671 distribution checks across 56 parameter points passed**, with 300,000
  vectors per check. These compare three analytical subordinator Laplace values
  and three joint NTS characteristic-function values (real and imaginary
  components). The maximum discrepancy was **3.535 standard errors**, below
  the preselected seven-standard-error threshold.
- All four normal-generator variants passed the existing direct RNG tests:
  six million normal draws per variant, Gaussian moments/CDF/tails, lag checks,
  uniform precision/range, exponential means, seed sensitivity, and concurrent
  first initialization. These are regression checks, not a complete PRNG test battery.
- AddressSanitizer and UndefinedBehaviorSanitizer passed representative
  vForce, scalar-batch, and direct-IG distribution runs. Python lint passed.

The vector backend is **macOS-specific**. Equivalent Linux or Windows speedups
require a separate vector-math implementation and measurements. These results
do not establish whole-application speedups; production integration must also
preserve per-path streams, worker-count reproducibility, and memory bounds.

## Reproduction and artifacts

```sh
.venv/bin/python benchmarks/nts-candidates/run.py --build-dir /tmp/nts-candidates --phase all
.venv/bin/python benchmarks/nts-candidates/summarize.py
```

Individual phases are `build`, `screen`, `validate`, `confirm`, and `paths`.
Run timing phases serially on an otherwise idle machine. `build` runs the
direct RNG tests; `validate` checks the subordinator and joint NTS law.
`screen` must precede `confirm`/`validate` to supply envelope selections.

- `screen.csv`, `selection.json`: pilot timings and frozen envelope choices.
- `confirmation.csv`: 1,526 independent bulk timing runs.
- `validation.csv`: 671 distribution-check results.
- `short_paths.csv`: 126 runs including path setup and buffer wastage.
- `summary.json`: medians and ratios computed from those raw files.
- `metadata.json`: platform, compiler, flags, and baseline source hashes.
- `compare.cpp`, `batched_qu.hpp`, `short_paths.cpp`, `run.py`: candidate
  implementations, generation, and benchmark orchestration.

References: [Xoshiro authors' guidance](https://prng.di.unimi.it/),
[McFarland's modified Ziggurat](https://pubmed.ncbi.nlm.nih.gov/27041780/),
[NTS and the NIG special case](https://link.springer.com/article/10.1007/s11222-022-10153-8),
[Michael, Schucany, and Haas](https://www.tandfonline.com/doi/abs/10.1080/00031305.1976.10479147).
