// swift-tools-version: 6.2
// mlx-yue2-swift — M-A-P's YuE2-3B (lyrics + style prompt → a full song with vocals through an editable ABC score)
// ported to Swift-MLX. Port plan, licence posture and gates: mlxengine-audio/WIP/yue2-port/PORTING-SPEC.md;
// decision AB-D-0073 (reversed 2026-10-24), ask AB-A-0200, task AB-T-0350.
//
// Licence: this package's code is MIT. The YuE2-3B, YuE2-Vae and YuE2-Vae-legacy WEIGHTS are CC BY-NC 4.0 — a
// non-commercial, named-only engine model; outputs are non-commercial. The bundles ship both licence texts.
//
// Pipeline (YuE2Core, 1:1 with multimodal-art-projection/YuE@1647252 `src/yue2`):
//   • Tokenizer — tiktoken byte-level BPE over the checkpoint's qwen.tiktoken (151 643 ranks), encode_ordinary + NFC.
//   • Backbone  — Qwen3-style 28 × 2048 Mixture-of-Transformers: an AR path and a NAR path per layer, one shared SDPA.
//   • Sampler   — abc phase (the ABC score) then the semantic phase (codec tokens), windowed repetition penalty,
//                 top-k / top-p, optional two-branch CFG.
//   • NAR       — flow matching: the AR prefix prefilled once per chunk, 32-step midpoint over the NAR path.
//   • VAE       — Oobleck decoder (fp32): 64-dim latents at 25 Hz → 48 kHz stereo, exact tiled decode.
//   • MLXYuE2   — the engine-facing package on `Capability.music` (mlx-engine-swift contract 1.56.0, AB-A-0201): style +
//                 lyrics, the planning mode (full / off) with the ABC score in and out, seed, cfg, a length cap; CC-BY-NC-4.0
//                 weights (never allowlisted), named-only; tiers bf16 / ar8 / int8 with `phys_footprint` footprints.
//   • yue2-gates — parity gates against the fp32 torch oracle (WIP/yue2-port/oracle), a render lane, and the V3 validate /
//                 envelope lanes (MLXServeEngine register → prepare → run → refusals → evict).
import PackageDescription

let package = Package(
    name: "mlx-yue2-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "YuE2Core", targets: ["YuE2Core"]),
        .library(name: "MLXYuE2", targets: ["MLXYuE2"]),
        .executable(name: "yue2-gates", targets: ["yue2-gates"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.3"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.71.0"),
    ],
    targets: [
        .target(
            name: "YuE2Core",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/YuE2Core",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXYuE2",
            dependencies: [
                "YuE2Core",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/MLXYuE2",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "yue2-gates",
            dependencies: [
                "YuE2Core", "MLXYuE2",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/yue2-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "YuE2CoreTests",
            dependencies: ["YuE2Core"],
            path: "Tests/YuE2CoreTests"
        ),
        .testTarget(
            name: "MLXYuE2Tests",
            dependencies: [
                "YuE2Core", "MLXYuE2",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXYuE2Tests"
        ),
    ]
)
