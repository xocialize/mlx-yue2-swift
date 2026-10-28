// YuE2Configuration.swift — init-time configuration for `YuE2Package` (C9). Per-request style / lyrics / planning / score /
// seed / cfg / length ride the canonical `MusicRequest`.
//
// TWO weight sets back the model, each one of the V1 converter's bundles (`Tools/convert/convert.py`, bit-equal to upstream's
// tensors): the backbone (`config.json`, `model.safetensors` bf16 as m-a-p ships it, `qwen.tiktoken`) and the listening VAE
// decoder (`config.json`, `decoder.safetensors` fp32, weight_norm folded, encoder dropped). Both under CC BY-NC 4.0, both
// carrying LICENSE + MODEL_LICENSE + NOTICE. Published at V4 (operator's decision AB-D-0186, 2026-10-28) as
// `mlx-community/YuE2-3B-bf16` and `mlx-community/YuE2-Vae-fp32`, sha256-equal to the gated bundles; the revisions below pin
// those commits. Explicit directories (`modelDirectory` / `vaeDirectory`) still load a local bundle without the network. Tiers quantise the bf16 backbone at load (affine, group 64):
// `.ar8` the AR path's Linears to 8 bits, `.int8` the AR and NAR paths both — V2b's two passing 8-bit rows.

import Foundation
import MLX
import MLXToolKit
import YuE2Core

/// The precision tier — the load-time quantisation of the bf16 backbone (the VAE decoder stays fp32 on every tier, as upstream
/// requires).
public enum YuE2Tier: String, Sendable, Codable, CaseIterable {
    /// The release's precision (upstream runs bf16): 1.2–1.3e-2 velocity drift from the fp32 oracle (V2b).
    case bf16
    /// The AR path's Linears 8-bit, the NAR path bf16: the fast tier (143 vs 108 semantic tok/s on an M5 Max), ≤ 1.4e-2 drift.
    case ar8
    /// The AR and NAR paths both 8-bit: the smallest tier V2b passed (≤ 1.6e-2 drift). 4-bit AR is NOT a tier: the NAR attends to
    /// the K/V the AR prefix computes, and a 4-bit AR drifts it to 0.2 after 32 steps (V2b).
    case int8

    /// The `Quant` the governor keys the declared `QuantFootprint` on. `.int8` owns the manifest's `.int8` footprint; `.ar8`
    /// shares the case (the contract has one 8-bit quant) and hints its own, larger bytes (`FootprintConfigured`).
    public var quant: Quant {
        switch self {
        case .bf16: .bf16
        case .ar8, .int8: .int8
        }
    }

    /// The Core's load-time quantisation.
    public var quantization: YuE2Quantization {
        switch self {
        case .bf16: .none
        case .ar8: YuE2Quantization(arBits: 8, narBits: nil)
        case .int8: YuE2Quantization(arBits: 8, narBits: 8)
        }
    }
}

public struct YuE2Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var tier: YuE2Tier
    /// Revision of the backbone repo / the VAE repo; nil = the V4-published, hash-verified commit (`modelRevisionPin` /
    /// `vaeRevisionPin`).
    public var revision: String?
    public var vaeRevision: String?
    /// Explicit backbone bundle directory (the converter's layout) — the dev escape hatch, never touches the network.
    public var modelDirectory: URL?
    /// Explicit VAE decoder bundle directory (the converter's `YuE2-Vae-fp32` layout).
    public var vaeDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// MLX buffer-cache limit while the model is resident (the fleet's bounded-cache rule; every footprint was measured at 1 GiB).
    public var cacheLimitBytes: UInt64

    public init(tier: YuE2Tier = .bf16, revision: String? = nil, vaeRevision: String? = nil, modelDirectory: URL? = nil,
                vaeDirectory: URL? = nil, modelsRootDirectory: URL? = nil, cacheLimitBytes: UInt64 = 1 << 30) {
        self.tier = tier; self.revision = revision; self.vaeRevision = vaeRevision
        self.modelDirectory = modelDirectory; self.vaeDirectory = vaeDirectory
        self.modelsRootDirectory = modelsRootDirectory; self.cacheLimitBytes = cacheLimitBytes
    }

    public var quant: Quant { tier.quant }

    // MARK: repos (published at V4, 2026-10-28; the V1-gated sources are m-a-p's)

    public static let modelRepo = "mlx-community/YuE2-3B-bf16"
    public static let vaeRepo = "mlx-community/YuE2-Vae-fp32"
    /// The V4 commits: `model.safetensors` / `decoder.safetensors` sha256-equal to the gated local bundles.
    public static let modelRevisionPin = "7024750ff947a1c57191a34908e14a8f55548516"
    public static let vaeRevisionPin = "486147eeb1e8397a87a7545ab51d9df95db60a4c"
    public var effectiveRevision: String { revision ?? Self.modelRevisionPin }
    public var effectiveVAERevision: String { vaeRevision ?? Self.vaeRevisionPin }
    /// The upstream checkpoints the converter read (V1 gated against them).
    public static let upstreamModelRepo = "m-a-p/YuE2-3B"
    public static let upstreamModelRevision = "8f24312f187763b9854adeec87e437e2b03bbff1"

    // Environment-specific URLs are excluded from Codable.
    private enum CodingKeys: String, CodingKey { case tier, revision, vaeRevision, cacheLimitBytes }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        tier = try c.decodeIfPresent(YuE2Tier.self, forKey: .tier) ?? .bf16
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        vaeRevision = try c.decodeIfPresent(String.self, forKey: .vaeRevision)
        cacheLimitBytes = try c.decodeIfPresent(UInt64.self, forKey: .cacheLimitBytes) ?? (1 << 30)
    }
}

// MARK: - Footprint hint (two tiers on `.int8`)

extension YuE2Configuration: FootprintConfigured {
    /// `.ar8` shares `.int8` with the 8-bit-both tier but keeps the NAR path bf16, so it hints its own (larger) resident bytes;
    /// the activation is the same stage's either way (the NAR solve and the VAE), so no activation hint.
    public var residentBytesHint: UInt64? { tier == .ar8 ? YuE2Package.ar8ResidentBytes : nil }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension YuE2Configuration: WeightSourcing {
    /// The files `YuE2Pipeline.load` opens per bundle — the materialization contract.
    public static let modelFiles = ["config.json", "model.safetensors", "qwen.tiktoken"]
    public static let vaeFiles = ["config.json", "decoder.safetensors"]
    /// Representative file per bundle for the missing-probe.
    static let modelProbe = "model.safetensors"
    static let vaeProbe = "decoder.safetensors"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "model", repo: Self.modelRepo, revision: effectiveRevision, matching: Self.modelFiles),
         WeightSource(role: "vae", repo: Self.vaeRepo, revision: effectiveVAERevision, matching: Self.vaeFiles)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if modelDirectory != nil || vaeDirectory != nil {
            let fm = FileManager.default
            let present = [modelDirectory.map { fm.fileExists(atPath: $0.appendingPathComponent(Self.modelProbe).path) } ?? false,
                           vaeDirectory.map { fm.fileExists(atPath: $0.appendingPathComponent(Self.vaeProbe).path) } ?? false]
            return zip(weightSources, present).filter { !$0.1 }.map(\.0)
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    /// The two bundle directories `load()` opens: the explicit ones, else each repo's snapshot (hub-client layout) or flat directory
    /// (engine-executed layout) under the store. Nil when either does not resolve.
    public func resolvedBundles(storeRoot: URL?) -> (model: URL, vae: URL)? {
        let store = ModelStore(root: storeRoot)
        func dir(_ explicit: URL?, _ repo: String, _ revision: String?, _ probe: String) -> URL? {
            if let explicit { return explicit }
            if let snapshot = store.snapshotDirectory(for: repo, revision: revision),
               FileManager.default.fileExists(atPath: snapshot.appendingPathComponent(probe).path) { return snapshot }
            if let flat = store.directory(for: repo),
               FileManager.default.fileExists(atPath: flat.appendingPathComponent(probe).path) { return flat }
            return nil
        }
        guard let m = dir(modelDirectory, Self.modelRepo, effectiveRevision, Self.modelProbe),
              let v = dir(vaeDirectory, Self.vaeRepo, effectiveVAERevision, Self.vaeProbe) else { return nil }
        return (m, v)
    }
}

// MARK: - Cold-start prewarm

extension YuE2Configuration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let b = resolvedBundles(storeRoot: modelsRootDirectory) else { return [] }
        return [b.model.appendingPathComponent(Self.modelProbe), b.vae.appendingPathComponent(Self.vaeProbe)]
    }
}
