#!/bin/bash
set -e
echo "🏗️  Building GrainVDB Native Core..."
xcrun -sdk macosx metal -c src/grain_kernel.metal -o gv_kernel.air
mkdir -p grainvdb
xcrun -sdk macosx metallib gv_kernel.air -o grainvdb/gv_kernel.metallib
rm gv_kernel.air
clang++ -dynamiclib -std=c++17 -O3 -fobjc-arc -Iinclude -framework Metal -framework Foundation src/grainvdb.mm -o libgrainvdb.dylib
echo "✅ Build Complete."
