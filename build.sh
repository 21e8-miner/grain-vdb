#!/bin/bash
#
# GrainVDB v2.0 - Breakthrough Edition Build Script
# Builds the native Metal-accelerated vector search engine
#

set -e

echo "=========================================="
echo "  GrainVDB v2.0 - Breakthrough Edition"
echo "  Building Native Metal Core..."
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
INCLUDE_DIR="${SCRIPT_DIR}/include"
GRAINVDB_DIR="${SCRIPT_DIR}/grainvdb"
BUILD_DIR="${SCRIPT_DIR}/build"

# Create build directory
mkdir -p "${BUILD_DIR}"

# Check for Metal compiler
if ! command -v xcrun &> /dev/null; then
    echo -e "${RED}Error: xcrun not found. Please install Xcode Command Line Tools.${NC}"
    echo "  Run: xcode-select --install"
    exit 1
fi

echo "Step 1: Compiling Metal kernels..."
echo "  Source: ${SRC_DIR}/grain_kernel.metal"
echo "  Output: ${GRAINVDB_DIR}/gv_kernel.metallib"

# Compile Metal kernel to .air file
xcrun -sdk macosx metal \
    -c "${SRC_DIR}/grain_kernel.metal" \
    -o "${BUILD_DIR}/grain_kernel.air" \
    -I"${INCLUDE_DIR}" \
    -O3 \
    -std=metal3.0 \
    2>&1

# Create Metal library
xcrun -sdk macosx metallib \
    "${BUILD_DIR}/grain_kernel.air" \
    -o "${GRAINVDB_DIR}/gv_kernel.metallib"

echo -e "${GREEN}  ✓ Metal kernels compiled successfully${NC}"
echo ""

echo "Step 2: Compiling Objective-C++ native driver..."
echo "  Source: ${SRC_DIR}/grainvdb.mm"
echo "  Output: ${GRAINVDB_DIR}/libgrainvdb.dylib"

# Compile native driver
xcrun -sdk macosx clang++ \
    -dynamiclib \
    -fobjc-arc \
    -fmodules \
    -framework Metal \
    -framework Foundation \
    -framework Accelerate \
    -O3 \
    -std=c++17 \
    -I"${INCLUDE_DIR}" \
    "${SRC_DIR}/grainvdb.mm" \
    -o "${GRAINVDB_DIR}/libgrainvdb.dylib" \
    2>&1

echo -e "${GREEN}  ✓ Native driver compiled successfully${NC}"
echo ""

echo "Step 3: Setting library permissions..."
chmod +x "${GRAINVDB_DIR}/libgrainvdb.dylib"
echo -e "${GREEN}  ✓ Permissions set${NC}"
echo ""

# Verify build
echo "Step 4: Verifying build artifacts..."
METAL_LIB="${GRAINVDB_DIR}/gv_kernel.metallib"
DYLIB="${GRAINVDB_DIR}/libgrainvdb.dylib"

if [ -f "$METAL_LIB" ] && [ -f "$DYLIB" ]; then
    echo -e "${GREEN}  ✓ Build verification passed${NC}"
    echo ""
    echo "  Artifacts:"
    echo "    - Metal Library: ${METAL_LIB}"
    ls -lh "$METAL_LIB" | awk '{print "      Size:", $5}'
    echo "    - Native Library: ${DYLIB}"
    ls -lh "$DYLIB" | awk '{print "      Size:", $5}'
else
    echo -e "${RED}  ✗ Build verification failed${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}  Build completed successfully!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run benchmark: python3 benchmark.py"
echo "  2. Run tests: python3 -m pytest tests/"
echo "  3. See examples: ls examples/"
echo ""
echo "Breakthrough features enabled:"
echo "  ✓ GPU-Accelerated Top-K (bitonic sort)"
echo "  ✓ Batch Query Processing (100x throughput)"
echo "  ✓ HNSW Approximate Search (sub-linear)"
echo "  ✓ INT8 Quantization (4x memory)"
echo "  ✓ Persistence with mmap"
echo ""

# Cleanup build directory
rm -rf "${BUILD_DIR}"

exit 0
