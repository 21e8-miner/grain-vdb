#!/usr/bin/env bash
set -e

VERSION="2.1.0"

echo "=========================================="
echo "  Building Native macOS App: GrainMemory.app"
echo "  Version: v${VERSION} (Apple Silicon)"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$ROOT_DIR/dist"
APP_BUNDLE="$DIST_DIR/GrainMemory.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES_DIR="$CONTENTS/Resources"

mkdir -p "$DIST_DIR"
rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

# 1. Build Swift executable
cd "$ROOT_DIR"
./build.sh
cp grainvdb/gv_kernel.metallib Sources/GrainVDB/gv_kernel.metallib 2>/dev/null || true

swift build -c release --product GrainMemoryApp

BIN_PATH="$(swift build -c release --show-bin-path)/GrainMemoryApp"

# 2. Copy binary and resources into .app bundle
cp "$BIN_PATH" "$MACOS_DIR/GrainMemoryApp"
chmod +x "$MACOS_DIR/GrainMemoryApp"

# Copy Agent DVR and HTML assets into resources
cp docs/agent_dvr.html "$RESOURCES_DIR/agent_dvr.html"
cp docs/index.html "$RESOURCES_DIR/index.html"
cp grainvdb/gv_kernel.metallib "$RESOURCES_DIR/gv_kernel.metallib"
if [ -d "docs/assets" ]; then
    cp -R docs/assets "$RESOURCES_DIR/assets"
fi

# 3. Create Info.plist
cat << 'EOF' > "$CONTENTS/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>GrainMemoryApp</string>
    <key>CFBundleIdentifier</key>
    <string>dev.grainvdb.GrainMemory</string>
    <key>CFBundleName</key>
    <string>GrainMemory</string>
    <key>CFBundleDisplayName</key>
    <string>GrainMemory</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>2.1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# 4. Ad-hoc codesign
codesign -s - --force --deep "$APP_BUNDLE" 2>/dev/null || true

# 5. Create ZIP archive
cd "$DIST_DIR"
rm -f "grain-memory-mac-app.zip"
zip -r -y "grain-memory-mac-app.zip" "GrainMemory.app"

# 6. Copy to Downloads folder if exists
DOWNLOADS_DIR="$HOME/Downloads"
if [ -d "$DOWNLOADS_DIR" ]; then
    cp "grain-memory-mac-app.zip" "$DOWNLOADS_DIR/grain-memory-mac-app.zip"
    echo "✓ Copied to: $DOWNLOADS_DIR/grain-memory-mac-app.zip"
fi

echo "=========================================="
echo "✓ Native macOS App built successfully:"
echo "  • App Bundle: $APP_BUNDLE"
echo "  • Distribution Zip: $DIST_DIR/grain-memory-mac-app.zip"
ls -lh "$DIST_DIR/grain-memory-mac-app.zip"
echo "=========================================="
