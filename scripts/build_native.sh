#!/bin/bash
set -e

echo "🏗️ Building GrainVDB Native Core..."

# 1. Setup build directory
mkdir -p dist

# 2. Compile Metal Kernel
echo "🌀 Compiling Metal Kernel..."
xcrun -sdk macosx metal -c src/grain_kernel.metal -o dist/gv_kernel.air
xcrun -sdk macosx metallib dist/gv_kernel.air -o dist/gv_kernel.metallib
rm dist/gv_kernel.air

# 3. Compile Objective-C++ Core to Dynamic Library
echo "💎 Compiling C++/Metal Bridge..."
clang++ -dynamiclib -std=c++17 -O3 \
    -Iinclude \
    -framework Metal -framework Foundation -framework CoreGraphics \
    src/grainvdb.mm -o dist/libgrainvdb.dylib

echo "✅ Build Complete. Artifacts in 'dist/'"
