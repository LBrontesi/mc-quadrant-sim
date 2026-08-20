#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source_file="$project_root/src/mc_quadrants/native_simulation.cpp"
mc_compiler=${MC_CXX:-c++}

case "$(uname -s)" in
  Darwin)
    output_file="$project_root/src/mc_quadrants/_native_sim.dylib"
    library_flag=-dynamiclib
    ;;
  Linux)
    output_file="$project_root/src/mc_quadrants/_native_sim.so"
    library_flag=-shared
    ;;
  *)
    echo "Unsupported platform: $(uname -s)" >&2
    exit 1
    ;;
esac

"$mc_compiler" -O3 -std=c++17 -pthread -fPIC "$library_flag" "$source_file" -o "$output_file"
echo "Built $output_file"
