// NARSolver.swift — src/yue2/nar.py: memory-bounded acoustic flow matching with one AR prefill per chunk.
//   • `songChunks`: the chunk's AR tokens are prefix + (codec + CODEC_OFFSET) + [MUSIC_END]; the song's noise is ONE
//     draw (frames, 64) sliced per chunk — the reference draws it with torch.randn on the CPU, so a gate injects it.
//   • `CachedNAR`: prefill the AR tokens once through the AR path (per-layer K/V kept), then each velocity runs only
//     the NAR path over N+2 positions against [AR ; NAR] keys.
//   • `solve`: midpoint, `steps` steps, t from 1 to 0, raw t = clamp(logit(t), −20, 20) in float64 on the host;
//     `mid = x − v(x,t)·dt/2`, `x = x − v(mid, t − dt/2)·dt`; the state stays in the model dtype.

import Foundation
import MLX
import MLXRandom

public struct NARChunk {
    public var arTokens: [Int]
    public var noise: MLXArray   // (frames, 64) fp32
    public init(arTokens: [Int], noise: MLXArray) { self.arTokens = arTokens; self.noise = noise }
}

public enum NARSolver {
    public static func rawT(_ t: Double) -> Double { min(20, max(-20, log(t / (1 - t)))) }

    /// `song_chunks` with an injected (or freshly drawn) whole-song noise.
    public static func songChunks(prefix: [Int], codec: [Int], noise: MLXArray?, seed: UInt64) throws -> [NARChunk] {
        guard !codec.isEmpty, codec.allSatisfy({ $0 >= 0 && $0 < YuE2Protocol.codecSize }) else {
            throw YuE2Error.invalidInput("codec ids outside [0, 32768)")
        }
        let ranges = try YuE2Protocol.chunkRanges(frames: codec.count, prefixTokens: prefix.count)
        let n = noise ?? MLXRandom.normal([codec.count, 64], key: MLXRandom.key(seed))
        guard n.shape == [codec.count, 64] else { throw YuE2Error.invalidInput("noise must be (frames, 64)") }
        return ranges.map { r in
            NARChunk(arTokens: prefix + codec[r].map { $0 + YuE2Protocol.codecOffset } + [YuE2Protocol.musicEnd],
                     noise: n[r.lowerBound ..< r.upperBound])
        }
    }

    public final class CachedNAR {
        let model: YuE2Model
        public let chunk: NARChunk
        let arKV: [(MLXArray, MLXArray)]
        public let arLength: Int

        public init(model: YuE2Model, chunk: NARChunk) throws {
            self.model = model; self.chunk = chunk
            arLength = chunk.arTokens.count
            guard arLength + chunk.noise.dim(0) + 2 <= model.config.max_position_embeddings else {
                throw YuE2Error.invalidInput("acoustic chunk exceeds the model context")
            }
            let caches = model.newCaches()
            _ = model.forwardAR(ids: MLXArray(chunk.arTokens.map { Int32($0) }).expandedDimensions(axis: 0), caches: caches)
            arKV = caches.map { $0.used! }
            eval(arKV.flatMap { [$0.0, $0.1] })
        }

        public func velocity(_ state: MLXArray, rawT: Double) -> MLXArray {
            model.narVelocity(state: state, rawT: rawT, arKV: arKV, arLength: arLength)
        }

        /// Midpoint solve from the chunk's noise; returns fp32 (frames, 64).
        public func solve(steps: Int, onStep: ((Int, Int) -> Void)? = nil, isCancelled: (() -> Bool)? = nil) throws -> MLXArray {
            var x = chunk.noise.asType(model.dtype)
            let dt = 1.0 / Double(steps)
            for step in 0 ..< steps {
                if let isCancelled, isCancelled() { throw YuE2Error.cancelled("during acoustic flow matching") }
                let t = 1.0 - Double(step) * dt
                let v1 = velocity(x, rawT: NARSolver.rawT(t))
                let mid = x - v1 * MLXArray(Float(dt / 2)).asType(x.dtype)
                let v2 = velocity(mid, rawT: NARSolver.rawT(t - dt / 2))
                x = x - v2 * MLXArray(Float(dt)).asType(x.dtype)
                eval(x)
                onStep?(step + 1, steps)
            }
            return x.asType(.float32)
        }
    }

    /// `synthesize`: all chunks serially → fp32 (frames, 64).
    public static func synthesize(model: YuE2Model, prefix: [Int], codec: [Int], noise: MLXArray?, seed: UInt64, steps: Int,
                                  onStep: ((Int, Int) -> Void)? = nil, isCancelled: (() -> Bool)? = nil) throws -> MLXArray {
        let chunks = try songChunks(prefix: prefix, codec: codec, noise: noise, seed: seed)
        var out = [MLXArray]()
        for (ci, chunk) in chunks.enumerated() {
            let engine = try CachedNAR(model: model, chunk: chunk)
            let lat = try engine.solve(steps: steps, onStep: { s, total in onStep?(ci * total + s, total * chunks.count) },
                                       isCancelled: isCancelled)
            out.append(lat)
        }
        return concatenated(out, axis: 0)
    }
}
