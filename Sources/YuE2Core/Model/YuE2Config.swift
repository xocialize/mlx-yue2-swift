// YuE2Config.swift — config.json of m-a-p/YuE2-3B plus the converter's `mlx_yue2` section.
import Foundation

public struct QuantizationConfig: Codable, Sendable, Equatable {
    public var group_size: Int
    public var bits: Int
    public init(group_size: Int, bits: Int) { self.group_size = group_size; self.bits = bits }
}

public struct YuE2MLXSection: Codable, Sendable {
    public var dtype: String?
    public var source_repo: String?
    public var source_revision: String?
    public var generation: GenerationConfig?
    /// Present on a pre-quantised bundle: the layer Linears are stored as weight / scales / biases.
    public var quantization: QuantizationConfig?
}

public struct YuE2Config: Codable, Sendable {
    public var hidden_size: Int
    public var num_hidden_layers: Int
    public var num_attention_heads: Int
    public var num_key_value_heads: Int
    public var head_dim: Int
    public var intermediate_size: Int
    public var vocab_size: Int
    public var rms_norm_eps: Double
    public var rope_theta: Double
    public var max_position_embeddings: Int
    public var latent_dim: Int
    public var max_latent_frames: Int
    public var timestep_shift: Double
    public var mlx_yue2: YuE2MLXSection?

    public var generation: GenerationConfig { mlx_yue2?.generation ?? GenerationConfig() }
}
