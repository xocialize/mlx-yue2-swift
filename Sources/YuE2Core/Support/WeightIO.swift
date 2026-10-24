// WeightIO.swift — safetensors in, key contract enforced (the mlx-auk-swift / mlx-step-audio-editx-swift idiom): load
// on the CPU stream, cast floating tensors to the requested dtype, compare the module's flattened parameter paths with
// the on-disk keys — 0 missing, 0 unused, or throw — then `update(verify:)`. A partial load is the silent-failure class.
//
// Bundle layout = the V1 converter's (Tools/convert/convert.py): backbone keys verbatim; VAE decoder with weight-norm
// folded and conv weights in MLX (O, K, I).

import Foundation
import MLX
import MLXNN

public enum YuE2Error: Error, CustomStringConvertible {
    case missingFile(String)
    case keyContract(component: String, missing: [String], unused: [String])
    case badConfig(String)
    case invalidInput(String)
    case cancelled(String)

    public var description: String {
        switch self {
        case .missingFile(let p): return "missing file: \(p)"
        case .keyContract(let c, let m, let u):
            return "\(c): key contract violated — missing \(m.count) \(m.prefix(6)), unused \(u.count) \(u.prefix(6))"
        case .badConfig(let s): return "bad config: \(s)"
        case .invalidInput(let s): return "invalid input: \(s)"
        case .cancelled(let s): return "cancelled: \(s)"
        }
    }
}

public enum WeightIO {
    public static func load(_ url: URL, dtype: DType?, keepFP32: (String) -> Bool = { _ in false }) throws -> [String: MLXArray] {
        guard FileManager.default.fileExists(atPath: url.path) else { throw YuE2Error.missingFile(url.path) }
        return try Device.withDefaultDevice(.cpu) { () throws -> [String: MLXArray] in
            var out = [String: MLXArray]()
            for (k, v) in try loadArrays(url: url) {
                let floating = [DType.float16, .bfloat16, .float32].contains(v.dtype)
                if floating, let d = dtype, !keepFP32(k), v.dtype != d { out[k] = v.asType(d) } else { out[k] = v }
            }
            eval(Array(out.values))
            return out
        }
    }

    /// `rename` maps an on-disk key to the module's parameter path. Numeric path components are array indices to
    /// `ModuleParameters.unflattened`, so torch Sequential indices that are NOT arrays here (time_embedder.mlp.{0,2},
    /// the whole VAE decoder) are spelled `iN` in the Swift modules (a leading underscore marks a key private to MLXNN); `rename` applies that spelling.
    public static func apply(_ arrays: [String: MLXArray], to module: Module, component: String,
                             rename: (String) -> String = { $0 }) throws {
        var arrays = Dictionary(uniqueKeysWithValues: arrays.map { (rename($0.key), $0.value) })
        let expected = Set(module.parameters().flattened().map(\.0))
        let onDisk = Set(arrays.keys)
        let missing = expected.subtracting(onDisk).sorted()
        let unused = onDisk.subtracting(expected).sorted()
        guard missing.isEmpty && unused.isEmpty else {
            throw YuE2Error.keyContract(component: component, missing: missing, unused: unused)
        }
        try module.update(parameters: ModuleParameters.unflattened(arrays), verify: [.noUnusedKeys, .shapeMismatch])
        module.train(false)
        eval(module.parameters())
        arrays.removeAll()
    }

    /// Every numeric component → `iN` (the VAE decoder's Sequential tree).
    public static func underscoreIndices(_ key: String) -> String {
        key.split(separator: ".", omittingEmptySubsequences: false).map { Int($0) != nil ? "i\($0)" : String($0) }.joined(separator: ".")
    }

    /// Only `time_embedder.mlp.N` (the backbone's one Sequential); `model.layers.N` stays an array.
    public static func backboneKey(_ key: String) -> String {
        key.hasPrefix("time_embedder.mlp.") ? underscoreIndices(key) : key
    }
}

public enum ConfigIO {
    public static func load<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        guard FileManager.default.fileExists(atPath: url.path) else { throw YuE2Error.missingFile(url.path) }
        return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
    }
}
