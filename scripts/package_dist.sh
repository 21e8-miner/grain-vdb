#!/usr/bin/env bash
set -e

VERSION="2.1.0"

echo "=========================================="
echo "  Packaging GrainVDB v${VERSION} Distribution"
echo "  Target: macOS Darwin arm64 (Apple Silicon)"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$ROOT_DIR/dist"

mkdir -p "$DIST_DIR"

# 1. Build native Metal core binaries
cd "$ROOT_DIR"
./build.sh

# 2. Build Python wheel and source distribution
python3 -m pip install --upgrade build setuptools wheel
python3 -m build --outdir "$DIST_DIR"

# 3. Package C/Swift release tarball
PACKAGE_NAME="grainvdb-v${VERSION}-darwin-arm64"
STAGE_DIR="$DIST_DIR/$PACKAGE_NAME"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/include" "$STAGE_DIR/lib" "$STAGE_DIR/swift"

cp include/gv_core.h "$STAGE_DIR/include/"
cp grainvdb/libgrainvdb.dylib "$STAGE_DIR/lib/"
cp grainvdb/gv_kernel.metallib "$STAGE_DIR/lib/"
cp Package.swift "$STAGE_DIR/swift/"
cp -R Sources "$STAGE_DIR/swift/"
cp README.md "$STAGE_DIR/"
cp LICENSE "$STAGE_DIR/"
cp COMMERCIAL_LICENSE.md "$STAGE_DIR/"

cd "$DIST_DIR"
tar -czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME"
rm -rf "$STAGE_DIR"

# 4. Build Native macOS Menu Bar App (grain-memory-mac-app.zip)
cd "$ROOT_DIR"
./scripts/build_mac_app.sh

echo "=========================================="
echo "✓ Distribution build complete:"
ls -lh "$DIST_DIR"
echo "=========================================="
