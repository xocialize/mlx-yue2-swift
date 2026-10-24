// ARSampler.swift — src/yue2/sampling.py `distribution` + `generate_tokens`: request-local sampling for the abc phase
// (text vocabulary + </abc>) and the semantic phase (codec range + MUSIC_END). Order kept: vocabulary mask → end token
// held back below min_tokens → windowed frequency penalty (alpha = penalty^freq; logits<0 ? ·alpha : /alpha) →
// temperature → top-k threshold → top-p on the sorted softmax (keeps the first 1, or 3 in legacy-off) → categorical.
// CFG when scale ≠ 1: `uncond + s·(cond − uncond)` in the model dtype, two KV caches. Sampling goes through
// MLXRandom.categorical, so a seeded rollout is never a parity gate (AB-L-0090); the gates are teacher-forced.

import Foundation
import MLX
import MLXRandom

public enum SamplingPhase { case abc, semantic }

public struct TokenRun {
    public var tokens: [Int]
    public var truncated: Bool
    public var prefillSeconds: Double
    public var seconds: Double
    public var tokensPerSecond: Double { seconds > 0 ? Double(tokens.count) / seconds : 0 }
}

public enum ARSampler {
    /// `distribution`: logits (vocab,) → scores (vocab,) ready for categorical sampling.
    static func distribution(_ logits: MLXArray, sampling s: SamplingConfig, history: [Int], step: Int, phase: SamplingPhase,
                             legacyOff: Bool, allowed: MLXArray, endToken: Int) -> MLXArray {
        var scores = legacyOff ? logits : logits.asType(.float32)
        scores = scores + allowed.asType(scores.dtype)
        let vocab = scores.dim(0)
        let negInf = MLXArray(-Float.infinity).asType(scores.dtype)
        if step < s.min_tokens {
            scores = putAlong(scores, MLXArray([Int32(endToken)]), values: negInf.reshaped([1]), axis: 0)
        }
        if s.repetition_penalty != 1, !history.isEmpty {
            let recent = history.suffix(s.penalty_window)
            var counts = [Int: Int]()
            for t in recent { counts[t, default: 0] += 1 }
            let ids = MLXArray(counts.keys.map { Int32($0) })
            let alpha = MLXArray(counts.keys.map { Float(pow(Double(s.repetition_penalty), Double(counts[$0]!))) }).asType(scores.dtype)
            let sel = scores[ids]
            scores = putAlong(scores, ids, values: which(sel .< 0, sel * alpha, sel / alpha), axis: 0)
        }
        if s.temperature == 0 { return scores }
        if s.temperature != 1 { scores = scores / MLXArray(s.temperature).asType(scores.dtype) }
        let k = min(s.top_k, vocab)
        let threshold = sorted(scores)[vocab - k]
        scores = which(scores .< threshold, negInf, scores)
        if s.top_p < 1 {
            let order = argSort(-scores)                  // descending
            let values = scores[order]
            let probs = softmax(values, axis: 0)
            var removed = (cumsum(probs, axis: 0) - probs) .> MLXArray(s.top_p).asType(probs.dtype)
            let keep = legacyOff ? 3 : 1
            removed = putAlong(removed, MLXArray((0 ..< keep).map { Int32($0) }), values: MLXArray([Bool](repeating: false, count: keep)), axis: 0)
            let kept = which(removed, negInf, values)
            scores = putAlong(scores, order, values: kept, axis: 0)
        }
        return scores
    }

    /// The additive vocabulary mask for a phase: 0 where allowed, −inf elsewhere.
    static func allowedMask(phase: SamplingPhase, vocab: Int) -> MLXArray {
        var m = [Float](repeating: -.infinity, count: vocab)
        switch phase {
        case .abc:
            for i in 0 ..< YuE2Protocol.eod { m[i] = 0 }
            m[YuE2Protocol.abcEnd] = 0
        case .semantic:
            for i in YuE2Protocol.codecOffset ..< (YuE2Protocol.codecOffset + YuE2Protocol.codecSize) { m[i] = 0 }
            m[YuE2Protocol.musicEnd] = 0
        }
        return MLXArray(m)
    }

    /// `generate_tokens`: returns the generated ids WITHOUT the end token; `truncated` when max_tokens hit first.
    public static func generate(model: YuE2Model, prefix: [Int], sampling s: SamplingConfig, seed: UInt64, phase: SamplingPhase,
                                negative: [Int]? = nil, cfgScale: Float = 1, legacyOff: Bool = false,
                                onToken: ((Int, Int) -> Void)? = nil, isCancelled: (() -> Bool)? = nil) throws -> TokenRun {
        guard prefix.count + s.max_tokens <= YuE2Protocol.context else {
            throw YuE2Error.invalidInput("prefix + generation budget exceeds \(YuE2Protocol.context)")
        }
        if cfgScale != 1 { guard negative != nil else { throw YuE2Error.invalidInput("CFG needs a negative prefix") } }
        let endToken = phase == .abc ? YuE2Protocol.abcEnd : YuE2Protocol.musicEnd
        let allowed = allowedMask(phase: phase, vocab: model.config.vocab_size)
        var key = MLXRandom.key(seed)
        let t0 = Date()

        let posCaches = model.newCaches()
        var conditional = model.lastLogits(ids: MLXArray(prefix.map { Int32($0) }).expandedDimensions(axis: 0), caches: posCaches)
        var negCaches: [KVCache]? = nil
        var unconditional: MLXArray? = nil
        if cfgScale != 1, let negative {
            negCaches = model.newCaches()
            unconditional = model.lastLogits(ids: MLXArray(negative.map { Int32($0) }).expandedDimensions(axis: 0), caches: negCaches)
            eval(unconditional!)
        }
        eval(conditional)
        let prefillSeconds = Date().timeIntervalSince(t0)

        var history = [Int]()
        var eos = false
        for step in 0 ..< s.max_tokens {
            if let isCancelled, isCancelled() { throw YuE2Error.cancelled("during \(phase)") }
            let logits = cfgScale == 1 ? conditional : unconditional! + MLXArray(cfgScale).asType(conditional.dtype) * (conditional - unconditional!)
            let scores = distribution(logits, sampling: s, history: history, step: step, phase: phase, legacyOff: legacyOff,
                                      allowed: allowed, endToken: endToken)
            let next: MLXArray
            if s.temperature == 0 { next = argMax(scores, axis: 0) }
            else {
                let (k1, k2) = MLXRandom.split(key: key); key = k1
                next = MLXRandom.categorical(scores, key: k2)
            }
            let token = Int(next.item(Int32.self))
            onToken?(step, token)
            if token == endToken { eos = true; break }
            history.append(token)
            if step + 1 < s.max_tokens {
                let ids = MLXArray([Int32(token)]).reshaped(1, 1)
                conditional = model.lastLogits(ids: ids, caches: posCaches)
                if let negCaches { unconditional = model.lastLogits(ids: ids, caches: negCaches) }
                eval(conditional)
                if let u = unconditional { eval(u) }
            }
        }
        return TokenRun(tokens: history, truncated: !eos, prefillSeconds: prefillSeconds, seconds: Date().timeIntervalSince(t0))
    }
}
