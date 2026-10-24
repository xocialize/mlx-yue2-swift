// YuE2Pipeline.swift — src/yue2/pipeline.py `YuE2Pipeline`: plan (the ABC score) → semantic tokens → acoustic flow
// matching → VAE decode. Loads the converter's bundles: <model>/{config.json, model.safetensors, qwen.tiktoken} and
// <vae>/{config.json, decoder.safetensors}. The backbone runs at `dtype` (bf16 as shipped; fp32 for the gates); the
// decoder stays fp32 as upstream requires.
import Foundation
import MLX
import MLXNN

public enum YuE2Progress {
    case plan(tokens: Int)
    case semantic(tokens: Int)
    case synthesize(step: Int, total: Int)
    case decode(tile: Int, total: Int)
}

/// Runtime quantization tiers (PORTING-SPEC §3 V2b): affine per-group, applied after the weights load; AR and NAR
/// paths can take different bit widths (the NAR path is the quant-sensitive half).
public struct YuE2Quantization: Sendable {
    public var arBits: Int?
    public var narBits: Int?
    public var groupSize: Int
    public static let none = YuE2Quantization(arBits: nil, narBits: nil)
    public init(arBits: Int?, narBits: Int?, groupSize: Int = 64) { self.arBits = arBits; self.narBits = narBits; self.groupSize = groupSize }

    public func apply(_ model: YuE2Model) {
        if let b = arBits { model.quantizeLayers(groupSize: groupSize, bits: b, path: YuE2Model.isARPath) }
        if let b = narBits { model.quantizeLayers(groupSize: groupSize, bits: b, path: YuE2Model.isNARPath) }
        if arBits != nil || narBits != nil { eval(model.parameters()) }
    }
}

public struct SymbolicPlan: Sendable {
    public var request: SongRequest
    public var abc: String?
    public var abcIDs: [Int]
    public var prefix: [Int]
    public var truncated: Bool
    public var seconds: Double
}

public struct SongResult {
    /// Interleaved stereo at 48 kHz, clipped to ±1.
    public var audio: [Float]
    public var channels: Int
    public var sampleRate: Int
    public var plan: SymbolicPlan
    public var codec: [Int]
    public var latents: MLXArray
    public var semanticTruncated: Bool
    public var timing: [String: Double]
    public var truncated: Bool { plan.truncated || semanticTruncated }
}

public final class YuE2Pipeline {
    public let config: YuE2Config
    public let model: YuE2Model
    public let vae: YuE2VAE
    public let tokenizer: YuE2Tokenizer
    public let dtype: DType
    public var generation: GenerationConfig

    public init(config: YuE2Config, model: YuE2Model, vae: YuE2VAE, tokenizer: YuE2Tokenizer, dtype: DType) {
        self.config = config; self.model = model; self.vae = vae; self.tokenizer = tokenizer; self.dtype = dtype
        generation = config.generation
    }

    public static func loadModel(_ dir: URL, dtype: DType) throws -> (YuE2Config, YuE2Model, YuE2Tokenizer) {
        let config = try ConfigIO.load(YuE2Config.self, from: dir.appendingPathComponent("config.json"))
        let model = YuE2Model(config)
        if let q = config.mlx_yue2?.quantization { model.quantizeLayers(groupSize: q.group_size, bits: q.bits) }
        try WeightIO.apply(try WeightIO.load(dir.appendingPathComponent("model.safetensors"), dtype: dtype), to: model, component: "backbone", rename: WeightIO.backboneKey)
        let tok = try YuE2Tokenizer(vocabularyURL: dir.appendingPathComponent("qwen.tiktoken"))
        return (config, model, tok)
    }

    public static func loadVAE(_ dir: URL) throws -> YuE2VAE {
        let config = try ConfigIO.load(YuE2VAEConfig.self, from: dir.appendingPathComponent("config.json"))
        let vae = YuE2VAE(config)
        try WeightIO.apply(try WeightIO.load(dir.appendingPathComponent("decoder.safetensors"), dtype: .float32), to: vae, component: "vae", rename: WeightIO.underscoreIndices)
        return vae
    }

    public static func load(modelDir: URL, vaeDir: URL, dtype: DType, quantization: YuE2Quantization = .none) throws -> YuE2Pipeline {
        let (config, model, tok) = try loadModel(modelDir, dtype: dtype)
        if config.mlx_yue2?.quantization == nil { quantization.apply(model) }
        let vae = try loadVAE(vaeDir)
        return YuE2Pipeline(config: config, model: model, vae: vae, tokenizer: tok, dtype: dtype)
    }

    // MARK: stages

    public func plan(_ request: SongRequest, sampling: SamplingConfig? = nil, progress: ((YuE2Progress) -> Void)? = nil,
                     isCancelled: (() -> Bool)? = nil) throws -> SymbolicPlan {
        if request.mode == .off {
            return SymbolicPlan(request: request, abc: nil, abcIDs: [], prefix: try request.prefix(tokenizer, abcIDs: nil), truncated: false, seconds: 0)
        }
        if let abc = request.abc {
            let ids = tokenizer.encode(abc)
            return SymbolicPlan(request: request, abc: abc, abcIDs: ids, prefix: try request.prefix(tokenizer, abcIDs: ids), truncated: false, seconds: 0)
        }
        let run = try ARSampler.generate(model: model, prefix: try request.prefix(tokenizer, abcIDs: nil), sampling: sampling ?? generation.abc,
                                         seed: request.seed, phase: .abc, onToken: { s, _ in progress?(.plan(tokens: s + 1)) }, isCancelled: isCancelled)
        let abc = tokenizer.decode(run.tokens)
        return SymbolicPlan(request: request, abc: abc, abcIDs: run.tokens, prefix: try request.prefix(tokenizer, abcIDs: run.tokens),
                            truncated: run.truncated, seconds: run.seconds)
    }

    /// Codec ids (0…32767) for the plan.
    public func generateSemantic(_ plan: SymbolicPlan, sampling: SamplingConfig? = nil, progress: ((YuE2Progress) -> Void)? = nil,
                                 isCancelled: (() -> Bool)? = nil) throws -> (codec: [Int], run: TokenRun) {
        let req = plan.request
        let negative = req.guidance != 1 ? try req.negativePrefix(tokenizer, abcIDs: plan.abcIDs) : nil
        let run = try ARSampler.generate(model: model, prefix: plan.prefix, sampling: sampling ?? generation.semantic, seed: req.seed,
                                         phase: .semantic, negative: negative, cfgScale: req.guidance, legacyOff: req.mode == .off,
                                         onToken: { s, _ in progress?(.semantic(tokens: s + 1)) }, isCancelled: isCancelled)
        return (run.tokens.map { $0 - YuE2Protocol.codecOffset }, run)
    }

    /// Acoustic flow matching → fp32 latents (frames, 64). `noise` injects the song noise (gates); nil draws from the seed.
    public func synthesize(_ plan: SymbolicPlan, codec: [Int], noise: MLXArray? = nil, progress: ((YuE2Progress) -> Void)? = nil,
                           isCancelled: (() -> Bool)? = nil) throws -> MLXArray {
        try NARSolver.synthesize(model: model, prefix: plan.prefix, codec: codec, noise: noise, seed: plan.request.seed,
                                 steps: generation.ode_steps, onStep: { s, t in progress?(.synthesize(step: s, total: t)) },
                                 isCancelled: isCancelled)
    }

    /// Tiled VAE decode → (S, 2) fp32 clipped to ±1.
    public func decode(_ latents: MLXArray, progress: ((YuE2Progress) -> Void)? = nil) -> MLXArray {
        let y = vae.decodeTiled(latents, onTile: { t, n in progress?(.decode(tile: t, total: n)) })
        return clip(y, min: -1, max: 1)
    }

    public func generate(_ request: SongRequest, progress: ((YuE2Progress) -> Void)? = nil, isCancelled: (() -> Bool)? = nil) throws -> SongResult {
        var timing = [String: Double]()
        let t0 = Date()
        let plan = try plan(request, progress: progress, isCancelled: isCancelled)
        timing["abc_seconds"] = plan.seconds
        let (codec, run) = try generateSemantic(plan, progress: progress, isCancelled: isCancelled)
        timing["semantic_seconds"] = run.seconds; timing["semantic_tps"] = run.tokensPerSecond
        let t1 = Date()
        let latents = try synthesize(plan, codec: codec, progress: progress, isCancelled: isCancelled)
        eval(latents)
        timing["nar_seconds"] = Date().timeIntervalSince(t1)
        let t2 = Date()
        let audio = decode(latents, progress: progress)
        let samples = audio.asArray(Float.self)
        timing["vae_seconds"] = Date().timeIntervalSince(t2)
        timing["e2e_seconds"] = Date().timeIntervalSince(t0)
        return SongResult(audio: samples, channels: audio.dim(1), sampleRate: vae.config.sample_rate, plan: plan, codec: codec,
                          latents: latents, semanticTruncated: run.truncated, timing: timing)
    }
}
