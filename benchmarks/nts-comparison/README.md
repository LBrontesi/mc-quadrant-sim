# NTS implementation comparison

> Correction: the combined harness changes optimization materially. Its 2.16x
> ratio must not be presented as the advantage over the original standalone
> executable. Running upstream standalone gives Qu 189.908, 188.431, 189.015 ms
> and Devroye 292.584, 275.955, 277.107 ms (seed 7). Local standalone full NTS
> gives 136.810, 137.719, 138.128 ms: median speed ratio approximately 1.37x.
> Both use -O3 -march=native -flto; local additionally requires -std=c++17.
> Upstream was compiled with -include chrono to supply its missing include.
> Hardcoding parameters in the combined harness restores upstream Devroye to
> 280.4 ms but leaves Qu at 305.9 ms, so runtime parameters alone do not explain
> the Qu distortion. Exact compiler/inlining causes have not been isolated.
> Fixed-harness data are in results-fixed.csv; local-standalone.cpp provides
> the standalone local benchmark. Earlier results below remain for audit only.

Measured 2026-09-23 on this session's arm64 macOS host with Apple clang 21.0.0.
Both implementations use the same compiler flags, one thread, and the same
tempered-stable parameters alpha=0.5 and lambda=0.5. Full NTS draws use
beta=0.5, mu=0, sigma=1, and T normalized by alpha*lambda^(alpha-1).
The local raw tilted-stable functions are used directly to avoid confusing
these parameters with the application's standardized subordinator parameters.
Each implementation retains its own RNG and normal/gamma generation routines.

Source snapshot: https://github.com/LBrontesi/NTS_generator/blob/main/NTS.cpp
SHA256: 4d29d025a709d904e7c4fed3e7fc679f5230a2825d88b87273bc4cd7c09568c0
The snapshot is included unchanged in a namespace with its main renamed;
the compiler warning about the renamed unused main is harmless here.

Reproduce from the repository root:

```sh
clang++ benchmarks/nts-comparison/compare.cpp -std=c++17 -O3 -march=native -flto -o /tmp/nts-compare
/tmp/nts-compare 0.5 0.5 > benchmarks/nts-comparison/results.csv
```

One warmup round, then seven measured rounds of one million outputs each.
Execution order rotates; seeds are 7+round. Timing excludes setup, allocation,
and the subsequent output checksum/moment calculation. Output consumption
prevents elimination of the generation work. This is an implementation
comparison, not a comparison with identical underlying random streams.

| Implementation | Output | Median milliseconds per million | Range |
|---|---|---:|---:|
| Upstream Qu | Full NTS | 304.708 | 302.222–323.585 |
| Local Qu | Full NTS | 141.051 | 140.383–142.226 |
| Upstream Qu | Normalized subordinator | 299.389 | 297.188–308.882 |
| Local Qu | Normalized subordinator | 134.166 | 132.857–136.135 |
| Upstream Devroye | Full NTS | 532.563 | 526.460–537.788 |
| Local Devroye | Full NTS | 595.063 | 591.893–665.958 |

Local Qu is 2.16x faster for full NTS and 2.23x for the subordinator.
Local Devroye is about 12% slower than upstream Devroye.
The full-NTS Qu sample means averaged approximately -0.00059 upstream and
-0.00004 locally; average sample variances were 1.3505 and 1.3544, respectively.
Theoretical mean is zero and variance is 1+sqrt(2)/4 = 1.353553.
These moments are a sanity check, not a distributional correctness proof.

The upstream timings on this host are slower than the user's pasted timings;
compare implementations within this run rather than mixing absolute times
from different machines or runtime conditions. No production code was changed.
