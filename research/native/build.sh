#!/usr/bin/env bash
# Build the native clique core. No dependencies beyond g++ and libstdc++.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/libclique.so"
# -march=native is safe here: the solver is built on the box it runs on. Drop to
# -mavx2 -mbmi2 -mpopcnt if a binary ever has to move between machines.
g++ -O3 -march=native -funroll-loops -fno-plt -std=c++17 -pthread \
    -shared -fPIC "$here/clique.cpp" -o "$out"
echo "built $out"
