// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GrainVDB",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "GrainVDB",
            targets: ["GrainVDB"]
        ),
    ],
    targets: [
        .target(
            name: "CGrainVDB",
            path: "Sources/CGrainVDB",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include"),
                .unsafeFlags(["-std=c++17"])
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
                .linkedFramework("Accelerate")
            ]
        ),
        .target(
            name: "GrainVDB",
            dependencies: ["CGrainVDB"],
            path: "Sources/GrainVDB",
            resources: [
                .copy("gv_kernel.metallib")
            ]
        ),
        .testTarget(
            name: "GrainVDBTests",
            dependencies: ["GrainVDB"],
            path: "Tests/GrainVDBTests"
        ),
    ],
    cxxLanguageStandard: .cxx17
)
