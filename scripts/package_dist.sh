#!/usr/bin/env bash
set -e

echo "=========================================="
echo "  Packaging GrainVDB v2.0.0 Distribution"
echo "  Target: macOS Darwin arm64 (Apple Silicon)"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$ROOT_DIR/dist"

mkdir -p "$DIST_DIR"

# 1. Build latest binaries
cd "$ROOT_DIR"
./build.sh

# 2. Package release tarball
PACKAGE_NAME="grainvdb-v2.0.0-darwin-arm64"
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

echo "✓ Created distribution archive: $DIST_DIR/${PACKAGE_NAME}.tar.gz"
ls -lh "$DIST_DIR/${PACKAGE_NAME}.tar.gz"
