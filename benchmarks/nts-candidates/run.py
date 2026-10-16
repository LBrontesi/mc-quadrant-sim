"""Build isolated NTS candidates and run serial, order-rotated comparisons.

No production files or shared libraries are modified. Run from any directory:
python benchmarks/nts-candidates/run.py --build-dir /tmp/nts-candidates --phase all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINES = ("current", "plus", "plusplus", "mcfarland")
MODES = ("scalar", "batch64", "batch256") + (("vforce64", "vforce256") if platform.system() == "Darwin" else ())
GRID = [(2 * a, t) for a in (0.55, 0.65, 0.75, 0.85, 0.95)
        for t in (0.04, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)]
POINTS = [(1.5, 0.5), (1.1, 20.0), (1.9, 0.04), (1.3, 0.5), (1.7, 0.9),
          (1.55, 0.6), (1.3, 0.35), (1.4, 0.45), (1.0, 0.3535533905932738)]


def mcfarland_tables() -> str:
    """Equal-area interior rectangles plus an alias table for residual masses."""
    area = math.sqrt(math.pi / 2) / 256
    edges, heights = [], []
    previous, upper = 0.0, 12.0
    for _ in range(256):
        low, high = 0.0, 1.0
        for _ in range(80):
            mid = (low + high) / 2
            if math.exp(-mid * mid / 2) * (1 - mid * mid) > previous:
                low = mid
            else:
                high = mid
        peak = (low + high) / 2
        if peak * (math.exp(-peak * peak / 2) - previous) < area:
            break
        low, high = peak, upper
        for _ in range(90):
            mid = (low + high) / 2
            if mid * (math.exp(-mid * mid / 2) - previous) > area:
                low = mid
            else:
                high = mid
        edge = (low + high) / 2
        edges.append(edge)
        heights.append(math.exp(-edge * edge / 2))
        previous, upper = heights[-1], edge
    mass = math.sqrt(math.pi / 2)
    weights = [mass * math.erfc(edges[0] / math.sqrt(2))]
    weights += [mass * (math.erf(edges[i - 1] / math.sqrt(2)) - math.erf(edges[i] / math.sqrt(2)))
                - heights[i - 1] * (edges[i - 1] - edges[i]) for i in range(1, len(edges))]
    weights.append(mass * math.erf(edges[-1] / math.sqrt(2)) - heights[-1] * edges[-1])
    assert abs(sum(weights) - (256 - len(edges)) * area) < 1e-13
    scaled = [w * len(weights) / sum(weights) for w in weights]
    probabilities, aliases = [1.0] * len(weights), list(range(len(weights)))
    small = [i for i, value in enumerate(scaled) if value < 1]
    large = [i for i, value in enumerate(scaled) if value >= 1]
    while small and large:
        i, j = small.pop(), large.pop()
        probabilities[i], aliases[i] = scaled[i], j
        scaled[j] -= 1 - scaled[i]
        (small if scaled[j] < 1 else large).append(j)
    result = f"static constexpr unsigned mc_layers = {len(edges)}, mc_regions = {len(weights)};\n"
    for name, values, kind in (("mc_x", edges, "double"), ("mc_f", heights, "double"),
                               ("mc_prob", probabilities, "double"), ("mc_alias", aliases, "unsigned")):
        result += f"static constexpr {kind} {name}[] = {{" + ",".join(map(repr, values)) + "};\n"
    return result


def candidate_header(original: str, engine: str) -> str:
    if engine == "plus":
        return original.replace("rotate_left(state_[1] * 5, 7) * 9", "state_[0] + state_[3]").replace(
            "layer_bits_ = next();", "layer_bits_ = next() >> 16;"
        ).replace("layers_remaining_ = 8;", "layers_remaining_ = 6;")
    if engine == "plusplus":
        return original.replace("rotate_left(state_[1] * 5, 7) * 9", "rotate_left(state_[0] + state_[3], 23) + state_[0]")
    if engine != "mcfarland":
        return original
    result = original.replace("class RandomStream {", mcfarland_tables() + "\nclass RandomStream {")
    begin, end = result.index("    double normal()"), result.index("    double exponential()")
    result = result[:begin] + """    double normal() {
        const std::uint64_t bits = next();
        const unsigned layer = bits & 255u;
        const double sign = bits & 256u ? -1.0 : 1.0;
        const double horizontal = static_cast<double>(bits >> 11) * 0x1.0p-53;
        if (layer < mc_layers) return sign * horizontal * mc_x[layer];
        return modified_slow(sign);
    }

""" + result[end:]
    begin, end = result.index("    // Keep rejection/tail work"), result.index("    static std::uint64_t splitmix64")
    result = result[:begin] + """    [[gnu::noinline]] double modified_slow(double sign) {
        const double selection = uniform() * mc_regions;
        const unsigned slot = static_cast<unsigned>(selection);
        const unsigned region = selection - slot < mc_prob[slot] ? slot : mc_alias[slot];
        if (region == 0) {
            double tail, vertical;
            do {
                tail = exponential() / mc_x[0];
                vertical = exponential();
            } while (2.0 * vertical < tail * tail);
            return sign * (mc_x[0] + tail);
        }
        const double left = region == mc_layers ? 0.0 : mc_x[region];
        const double right = mc_x[region - 1];
        const double bottom = mc_f[region - 1];
        const double top = region == mc_layers ? 1.0 : mc_f[region];
        for (;;) {
            const double candidate = left + uniform() * (right - left);
            if (bottom + uniform() * (top - bottom) < std::exp(-0.5 * candidate * candidate))
                return sign * candidate;
        }
    }

""" + result[end:]
    return result


def build(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get("MC_CXX", "c++"))
    flags = ["-O3", "-std=c++17", "-pthread", f"-I{ROOT / 'src/mc_quadrants'}"]
    frameworks = ["-framework", "Accelerate"] if platform.system() == "Darwin" else []
    header = (ROOT / "src/mc_quadrants/native_random.hpp").read_text()
    source = (ROOT / "src/mc_quadrants/native_simulation.cpp").read_text()
    for engine in ENGINES:
        random_file = directory / f"random-{engine}.hpp"
        random_file.write_text(candidate_header(header, engine))
        native = directory / f"native-{engine}.cpp"
        native.write_text(source.replace('"native_random.hpp"', json.dumps(str(random_file))))
        subprocess.run([*compiler, *flags, f'-DMC_NTS_SOURCE="{native}"', str(HERE / "compare.cpp"),
                        *frameworks, "-o", str(directory / engine)], check=True, capture_output=True, text=True)
        check = directory / f"check-{engine}.cpp"
        check.write_text((ROOT / "tests/native_random_check.cpp").read_text().replace(
            '"native_random.hpp"', json.dumps(str(random_file))))
        subprocess.run([*compiler, *flags, str(check), "-o", str(directory / f"check-{engine}")],
                       check=True, capture_output=True, text=True)
        subprocess.run([str(directory / f"check-{engine}")], check=True, capture_output=True, text=True)
        print(f"Built {engine}; normal/uniform/exponential/stream checks passed", flush=True)
    metadata = {"platform": platform.platform(), "compiler": subprocess.check_output([*compiler, "--version"], text=True),
                "flags": flags, "frameworks": frameworks, "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "random_sha256": hashlib.sha256(header.encode()).hexdigest()}
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run(directory: Path, engine: str, mode: str, tail: float, tempering: float,
        samples: int, seed: int, assets: int = 4, validate: bool = False) -> dict:
    command = [str(directory / engine), mode, str(tail), str(tempering), str(samples), str(seed), str(assets), str(int(validate))]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=40)
    if completed.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout)


def write_rows(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def screen(directory: Path) -> None:
    rows, selection = [], []
    # Train envelope choices on separate seeds/samples from confirmation runs.
    for index, (tail, tempering) in enumerate(GRID + [p for p in POINTS if p not in GRID]):
        info = run(directory, "current", "info", tail, tempering, 1, 0)
        minimum = min(info["log_constants"])
        viable = [f"env{i}" for i, c in enumerate(info["log_constants"]) if c - minimum < math.log(8)]
        # A much larger envelope is unlikely to recover its acceptance penalty.
        candidates = [("current", m) for m in viable + list(MODES)]
        candidates += [(e, "scalar") for e in ENGINES if e != "current"]
        for trial in range(3):
            shift = (trial + index) % len(candidates)
            for engine, mode in candidates[shift:] + candidates[:shift]:
                result = run(directory, engine, mode, tail, tempering, 50_000, 100 + trial)
                rows.append(dict(tail=tail, tempering=tempering, trial=trial, engine=engine, mode=mode, **result))
        timings = {mode: statistics.median(r["ns_per_vector"] for r in rows
            if r["tail"] == tail and r["tempering"] == tempering and r["engine"] == "current" and r["mode"] == mode)
            for mode in viable}
        selection.append(dict(tail=tail, tempering=tempering, original=info["envelope"],
                              selected=int(min(timings, key=timings.get)[-1]), pilot_ns=timings))
        if index % 10 == 0:
            print(f"Screened {index + 1} parameter points", flush=True)
        write_rows("screen.csv", rows)
        (HERE / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")


def confirm(directory: Path) -> None:
    selection = json.loads((HERE / "selection.json").read_text())
    rows = []
    for index, (tail, tempering) in enumerate(POINTS):
        choice = next(p for p in selection if p["tail"] == tail and p["tempering"] == tempering)
        candidates = [(e, "scalar") for e in ENGINES]
        candidates += [("current", m) for m in MODES if m != "scalar"]
        if platform.system() == "Darwin":
            candidates += [(e, "vforce256") for e in ENGINES if e != "current"]
        candidates += [("current", f"env{choice['selected']}")]
        if tail == 1.0:
            candidates += [("current", "ig")]
        for assets in (1, 4):
            for trial in range(7):
                shift = (trial + index) % len(candidates)
                for engine, mode in candidates[shift:] + candidates[:shift]:
                    result = run(directory, engine, mode, tail, tempering, 1_000_000, 9000 + trial, assets)
                    rows.append(dict(tail=tail, tempering=tempering, assets=assets, trial=trial,
                                     engine=engine, mode=mode, **result))
            write_rows("confirmation.csv", rows)
        print(f"Confirmed {tail=}, {tempering=}", flush=True)


def validate(directory: Path) -> None:
    rows = []
    # Include all four Qu envelopes plus the calibration grid and IG boundaries.
    points = list(dict.fromkeys(GRID + POINTS + [
        (0.38, 0.19), (1.0, 0.3154786722400966), (0.4, 0.2), (1.0, 0.5), (1.0, 0.04), (1.0, 20.0),
    ]))
    selection = json.loads((HERE / "selection.json").read_text())
    for index, (tail, tempering) in enumerate(points):
        candidates = [(e, "scalar") for e in ENGINES]
        candidates += [("current", m) for m in MODES if m != "scalar"]
        if platform.system() == "Darwin":
            candidates += [(e, "vforce256") for e in ENGINES if e != "current"]
        chosen = next((p for p in selection if p["tail"] == tail and p["tempering"] == tempering), None)
        if chosen:
            candidates += [("current", f"env{chosen['selected']}")]
        if tail == 1.0:
            candidates += [("current", "ig")]
        for engine, mode in candidates:
            result = run(directory, engine, mode, tail, tempering, 300_000, 41000 + index, validate=True)
            rows.append(dict(tail=tail, tempering=tempering, engine=engine, mode=mode, **result))
        if index % 10 == 0:
            print(f"Validated {index + 1} parameter points", flush=True)
        write_rows("validation.csv", rows)


def short_paths(directory: Path) -> None:
    if platform.system() != "Darwin":
        return
    compiler = shlex.split(os.environ.get("MC_CXX", "c++"))
    for engine in ("current", "plus", "mcfarland"):
        native = directory / f"native-{engine}.cpp"
        subprocess.run([*compiler, "-O3", "-std=c++17", "-pthread", f"-I{ROOT / 'src/mc_quadrants'}",
                        f'-DMC_NTS_SOURCE="{native}"', str(HERE / "short_paths.cpp"), "-framework", "Accelerate",
                        "-o", str(directory / f"paths-{engine}")], check=True, capture_output=True, text=True)
    candidates = [(e, m) for e in ("current", "plus", "mcfarland") for m in ("scalar", "vforce64", "vforce256")]
    rows = []
    for periods in (120, 360):
        for trial in range(7):
            shift = trial % len(candidates)
            for engine, mode in candidates[shift:] + candidates[:shift]:
                result = json.loads(subprocess.check_output(
                    [str(directory / f"paths-{engine}"), mode, "3000", str(periods), str(10000 + trial)], text=True))
                rows.append(dict(periods=periods, paths=3000, trial=trial, engine=engine, mode=mode, **result))
        write_rows("short_paths.csv", rows)
        print(f"Short-path benchmark completed: {periods=}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("build", "screen", "confirm", "validate", "paths", "all"), default="all")
    args = parser.parse_args()
    directory = args.build_dir.resolve()
    for name, function in (("build", build), ("screen", screen), ("validate", validate),
                           ("confirm", confirm), ("paths", short_paths)):
        if args.phase in (name, "all"):
            function(directory)
