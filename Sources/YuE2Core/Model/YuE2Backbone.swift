// YuE2Backbone.swift — src/yue2/modeling_yue2.py: the AR–NAR Mixture-of-Transformers with checkpoint-compatible names.
//
// Every DecoderLayer carries two full paths — `input_layernorm` / `self_attn` / `post_attention_layernorm` / `mlp`
// (AR, the token path) and `nar_input_layernorm` / `nar_self_attn` / `nar_pre_mlp_layernorm` / `nar_mlp` (NAR, the
// latent path) — and ONE attention over the concatenated K/V. At inference the two never mix inside a forward:
//   • AR  (`forwardAR`): causal over tokens, KV-cached; positions 0…L−1.
//   • NAR (`forwardNAR`): the NAR path over N+2 latent positions whose keys are [cached AR K/V ; NAR K/V], UNMASKED
//     (NAR→AR full, NAR↔NAR bidirectional); RoPE positions continue from the AR prefix length (nar.py CachedNAR).
// Facts kept from upstream: q/k RMSNorm per head BEFORE RoPE; RoPE rotate-half (MLXFast traditional: false), θ 1e6;
// RMSNorm `x · rsqrt(mean(x²) + eps) · w` with the mean in fp32; untied lm_head; `latent_pos_embed.pe` is the
// checkpoint's own (bf16-rounded) sinusoid table; TimestepEmbedder = [cos ; sin] of 128 frequencies → Linear → SiLU
// → Linear, in `mlp.0` / `mlp.2`.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Append-only K/V store for one layer, grown in 512-slot steps (the mlx-swift-lm idiom): no per-step reallocation.
public final class KVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public private(set) var offset = 0
    let step = 512

    public init() {}

    /// Appends k, v (1, Hkv, T, D) and returns the used prefix views (1, Hkv, offset+T, D).
    public func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let t = k.dim(2)
        let prev = offset
        if keys == nil || prev + t > keys!.dim(2) {
            let needed = ((prev + t + step - 1) / step) * step
            let shape = [k.dim(0), k.dim(1), needed, k.dim(3)]
            var nk = MLXArray.zeros(shape, dtype: k.dtype), nv = MLXArray.zeros(shape, dtype: v.dtype)
            if let ok = keys, let ov = values, prev > 0 {
                nk[0..., 0..., ..<prev, 0...] = ok[0..., 0..., ..<prev, 0...]
                nv[0..., 0..., ..<prev, 0...] = ov[0..., 0..., ..<prev, 0...]
            }
            keys = nk; values = nv
        }
        keys![0..., 0..., prev ..< (prev + t), 0...] = k
        values![0..., 0..., prev ..< (prev + t), 0...] = v
        offset = prev + t
        return (keys![0..., 0..., ..<offset, 0...], values![0..., 0..., ..<offset, 0...])
    }

    /// The used K/V (for the NAR pass over an AR prefix).
    public var used: (MLXArray, MLXArray)? {
        guard let k = keys, let v = values, offset > 0 else { return nil }
        return (k[0..., 0..., ..<offset, 0...], v[0..., 0..., ..<offset, 0...])
    }
}

public final class YuE2Attention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, theta: Float, scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: YuE2Config) {
        heads = c.num_attention_heads; kvHeads = c.num_key_value_heads; headDim = c.head_dim
        theta = Float(c.rope_theta); scale = 1 / Float(c.head_dim).squareRoot()
        _qProj.wrappedValue = Linear(c.hidden_size, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, c.hidden_size, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(c.rms_norm_eps))
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(c.rms_norm_eps))
    }

    /// `project_qkv`: project, per-head norm, RoPE from `offset`. Returns (1, H, T, D) / (1, Hkv, T, D).
    func projectQKV(_ x: MLXArray, offset: Int) -> (MLXArray, MLXArray, MLXArray) {
        let (b, t) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, t, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, t, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: offset)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: offset)
        return (q, k, v)
    }

    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, causal: Bool) -> MLXArray {
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: causal ? .causal : .none)
        return oProj(o.transposed(0, 2, 1, 3).reshaped(q.dim(0), q.dim(2), -1))
    }
}

public final class YuE2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: YuE2Config) {
        _gateProj.wrappedValue = Linear(c.hidden_size, c.intermediate_size, bias: false)
        _upProj.wrappedValue = Linear(c.hidden_size, c.intermediate_size, bias: false)
        _downProj.wrappedValue = Linear(c.intermediate_size, c.hidden_size, bias: false)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

public final class YuE2DecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: YuE2Attention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: YuE2MLP
    @ModuleInfo(key: "nar_input_layernorm") var narInputLayernorm: RMSNorm
    @ModuleInfo(key: "nar_self_attn") var narSelfAttn: YuE2Attention
    @ModuleInfo(key: "nar_pre_mlp_layernorm") var narPreMLPLayernorm: RMSNorm
    @ModuleInfo(key: "nar_mlp") var narMLP: YuE2MLP

    init(_ c: YuE2Config) {
        let eps = Float(c.rms_norm_eps)
        _inputLayernorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: eps)
        _selfAttn.wrappedValue = YuE2Attention(c)
        _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: eps)
        _mlp.wrappedValue = YuE2MLP(c)
        _narInputLayernorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: eps)
        _narSelfAttn.wrappedValue = YuE2Attention(c)
        _narPreMLPLayernorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: eps)
        _narMLP.wrappedValue = YuE2MLP(c)
    }

    /// AR path. With a cache: append and attend over the whole cache (causal when prefilling T > 1 into an empty cache,
    /// no mask for single-token decode). Without: causal over x.
    func forwardAR(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let offset = cache?.offset ?? 0
        let (q, k0, v0) = selfAttn.projectQKV(inputLayernorm(x), offset: offset)
        var k = k0, v = v0
        if let cache { (k, v) = cache.update(k0, v0) }
        let causal = x.dim(1) > 1
        precondition(!(causal && offset > 0), "chunked prefill is not supported: prefill the whole prefix at once")
        let h = x + selfAttn.attend(q, k, v, causal: causal)
        return h + mlp(postAttentionLayernorm(h))
    }

    /// NAR path over latent positions: keys = [AR prefix K/V ; NAR K/V], unmasked; RoPE from `offset` (= AR length).
    func forwardNAR(_ x: MLXArray, arK: MLXArray, arV: MLXArray, offset: Int) -> MLXArray {
        let (q, k, v) = narSelfAttn.projectQKV(narInputLayernorm(x), offset: offset)
        let kk = concatenated([arK, k], axis: 2), vv = concatenated([arV, v], axis: 2)
        let h = x + narSelfAttn.attend(q, kk, vv, causal: false)
        return h + narMLP(narPreMLPLayernorm(h))
    }
}

public final class YuE2Backbone: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [YuE2DecoderLayer]
    @ModuleInfo(key: "norm") public var norm: RMSNorm

    init(_ c: YuE2Config) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocab_size, dimensions: c.hidden_size)
        _layers.wrappedValue = (0 ..< c.num_hidden_layers).map { _ in YuE2DecoderLayer(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
    }
}

final class TimestepMLP: Module {
    @ModuleInfo(key: "i0") var l0: Linear
    @ModuleInfo(key: "i2") var l2: Linear
    init(_ hidden: Int, frequency: Int) {
        _l0.wrappedValue = Linear(frequency, hidden); _l2.wrappedValue = Linear(hidden, hidden)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { l2(silu(l0(x))) }
}

public final class TimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") var mlp: TimestepMLP
    let frequency: Int
    init(_ hidden: Int, frequency: Int = 256) {
        self.frequency = frequency; _mlp.wrappedValue = TimestepMLP(hidden, frequency: frequency)
    }
    /// t (N,) → (N, hidden): freqs = exp(−ln 10000 · i / half), emb = [cos ; sin] (cos first), cast to the weight dtype.
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let half = frequency / 2
        let freqs = exp(MLXArray(0 ..< half).asType(.float32) * Float(-log(10000.0) / Double(half)))
        let args = t.asType(.float32).expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)
        let emb = concatenated([cos(args), sin(args)], axis: -1)
        return mlp(emb.asType(mlp.l0.weight.dtype))
    }
}

public final class AudioPositionEmbedding: Module {
    @ParameterInfo(key: "pe") var pe: MLXArray
    init(maxFrames: Int, hidden: Int) { _pe.wrappedValue = MLXArray.zeros([maxFrames, hidden]) }
    func callAsFunction(_ positions: MLXArray) -> MLXArray { pe[positions] }
}

/// `YuE2ForCausalLM`: backbone + lm_head + the NAR head and embedders.
public final class YuE2Model: Module {
    public let config: YuE2Config
    @ModuleInfo(key: "model") public var model: YuE2Backbone
    @ModuleInfo(key: "lm_head") public var lmHead: Linear
    @ModuleInfo(key: "llm2vae") public var llm2vae: Linear
    @ModuleInfo(key: "vae2llm") public var vae2llm: Linear
    @ModuleInfo(key: "time_embedder") public var timeEmbedder: TimestepEmbedder
    @ModuleInfo(key: "latent_pos_embed") public var latentPosEmbed: AudioPositionEmbedding

    public init(_ c: YuE2Config) {
        config = c
        _model.wrappedValue = YuE2Backbone(c)
        _lmHead.wrappedValue = Linear(c.hidden_size, c.vocab_size, bias: false)
        _llm2vae.wrappedValue = Linear(c.hidden_size, c.latent_dim)
        _vae2llm.wrappedValue = Linear(c.latent_dim, c.hidden_size)
        _timeEmbedder.wrappedValue = TimestepEmbedder(c.hidden_size)
        _latentPosEmbed.wrappedValue = AudioPositionEmbedding(maxFrames: c.max_latent_frames, hidden: c.hidden_size)
    }

    public var dtype: DType { lmHead.weight.dtype }

    /// The Linears a quantised tier replaces: every Linear inside the 28 layers, AR and NAR paths alike. `which`
    /// narrows to one path ("self_attn"/"mlp" = AR, "nar_" = NAR) so the tiers can quantise AR and NAR differently.
    public func quantizeLayers(groupSize: Int, bits: Int, path which: ((String) -> Bool)? = nil) {
        quantize(model: self, groupSize: groupSize, bits: bits) { path, m in
            guard m is Linear, path.hasPrefix("model.layers.") else { return false }
            return which?(path) ?? true
        }
    }

    public static func isARPath(_ path: String) -> Bool { !path.contains(".nar_") }
    public static func isNARPath(_ path: String) -> Bool { path.contains(".nar_") }

    // MARK: AR

    /// Runs the AR path over `ids` (1, L) with an optional cache; returns the final hidden states after `norm` (1, L, D).
    public func forwardAR(ids: MLXArray, caches: [KVCache]?) -> MLXArray {
        var h = model.embedTokens(ids)
        for (i, layer) in model.layers.enumerated() { h = layer.forwardAR(h, cache: caches?[i]) }
        return model.norm(h)
    }

    /// Logits (vocab,) of the LAST position only (lm_head on one row).
    public func lastLogits(ids: MLXArray, caches: [KVCache]?) -> MLXArray {
        var h = model.embedTokens(ids)
        for (i, layer) in model.layers.enumerated() { h = layer.forwardAR(h, cache: caches?[i]) }
        return lmHead(model.norm(h[0..., (h.dim(1) - 1)..., 0...]))[0, 0]
    }

    public func newCaches() -> [KVCache] { model.layers.map { _ in KVCache() } }

    // MARK: NAR

    /// `_shift_t_value`: sigmoid(raw) in the model dtype, then the (identity at 1.0) timestep shift.
    func shiftedT(_ raw: Double) -> MLXArray {
        let t = sigmoid(MLXArray(Float(raw)).asType(dtype))
        let s = Float(config.timestep_shift)
        return s * t / (1 + (s - 1) * t)
    }

    /// `CachedNAR.velocity`: state (N, D) at raw t → velocity (N, D). `arKV[i]` is layer i's prefilled AR K/V; `arLength`
    /// its length (the NAR RoPE offset). Positions: local 0…N+1 for the sinusoid table; START/END rows are zero latents.
    public func narVelocity(state: MLXArray, rawT: Double, arKV: [(MLXArray, MLXArray)], arLength: Int) -> MLXArray {
        let n = state.dim(0)
        let xNar = MLX.padded(state.asType(dtype), widths: [IntOrPair((1, 1)), IntOrPair((0, 0))]).expandedDimensions(axis: 0)
        var x = vae2llm(xNar)
        let t = broadcast(shiftedT(rawT).reshaped([1]), to: [n + 2])
        x = x + timeEmbedder(t).expandedDimensions(axis: 0)
        let pos = minimum(MLXArray(0 ..< (n + 2)), MLXArray(Int32(config.max_latent_frames - 1)))
        x = x + latentPosEmbed(pos).asType(dtype).expandedDimensions(axis: 0)
        for (i, layer) in model.layers.enumerated() {
            x = layer.forwardNAR(x, arK: arKV[i].0, arV: arKV[i].1, offset: arLength)
        }
        return llm2vae(model.norm(x))[0, 1 ..< (n + 1), 0...]
    }
}
