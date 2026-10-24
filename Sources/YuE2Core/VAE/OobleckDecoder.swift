// OobleckDecoder.swift — src/yue2/modeling_vae.py (decoder half): Oobleck / stable-audio-tools lineage, fp32, with
// exact tiled decoding. Tensors are channels-last (B, T, C); weight-norms arrive folded; conv weights are MLX (O, K, I).
// Keys mirror torch's Sequential indices: decoder.layers.{0 conv_pre, 1…6 DecoderBlock, 7 SnakeBeta, 8 conv_post};
// DecoderBlock.layers.{0 Snake, 1 ConvTranspose, 2…4 ResidualUnit}; ResidualUnit.layers.{0 Snake, 1 conv k7, 2 Snake,
// 3 conv k1}. SnakeBeta is log-scale: x + sin²(x·eᵅ) / (eᵝ + 1e−9). `final_tanh: false`.

import Foundation
import MLX
import MLXNN

public struct OobleckDecoderConfig: Codable, Sendable {
    public var out_channels: Int
    public var channels: Int
    public var latent_dim: Int
    public var c_mults: [Int]
    public var strides: [Int]
    public var use_snake: Bool
    public var final_tanh: Bool?
}

public struct YuE2VAEConfig: Codable, Sendable {
    public var decoder_config: OobleckDecoderConfig
    public var sample_rate: Int
    public var latent_dim: Int
    public var downsampling_ratio: Int
    public var audio_channels: Int
    public var release_variant: String?
    public var decode_core_frames: Int
    public var decode_halo_frames: Int
}

public final class WNConv1d: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?
    let stride: Int, dilation: Int, padding: Int

    init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int = 1, dilation: Int = 1, padding: Int, bias: Bool = true) {
        _weight.wrappedValue = MLXArray.zeros([outC, kernel, inC])
        _bias.wrappedValue = bias ? MLXArray.zeros([outC]) : nil
        self.stride = stride; self.dilation = dilation; self.padding = padding
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)
        if let bias { y = y + bias }
        return y
    }
}

public final class WNConvTranspose1d: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let stride: Int, padding: Int

    init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int, padding: Int) {
        _weight.wrappedValue = MLXArray.zeros([outC, kernel, inC])
        _bias.wrappedValue = MLXArray.zeros([outC])
        self.stride = stride; self.padding = padding
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        convTransposed1d(x, weight, stride: stride, padding: padding) + bias
    }
}

public final class SnakeBeta: Module, UnaryLayer {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray
    init(_ channels: Int) { _alpha.wrappedValue = MLXArray.zeros([channels]); _beta.wrappedValue = MLXArray.zeros([channels]) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = sin(x * exp(alpha))
        return x + (1.0 / (exp(beta) + 1e-9)) * (s * s)
    }
}

final class ResidualUnitLayers: Module {
    @ModuleInfo(key: "i0") var act0: SnakeBeta
    @ModuleInfo(key: "i1") var conv1: WNConv1d
    @ModuleInfo(key: "i2") var act2: SnakeBeta
    @ModuleInfo(key: "i3") var conv3: WNConv1d
    init(_ ch: Int, dilation: Int) {
        _act0.wrappedValue = SnakeBeta(ch)
        _conv1.wrappedValue = WNConv1d(ch, ch, kernel: 7, dilation: dilation, padding: (dilation * 6) / 2)
        _act2.wrappedValue = SnakeBeta(ch)
        _conv3.wrappedValue = WNConv1d(ch, ch, kernel: 1, padding: 0)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv3(act2(conv1(act0(x)))) }
}

public final class ResidualUnit: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: ResidualUnitLayers
    init(_ ch: Int, dilation: Int) { _layers.wrappedValue = ResidualUnitLayers(ch, dilation: dilation) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { layers(x) + x }
}

final class DecoderBlockLayers: Module {
    @ModuleInfo(key: "i0") var act: SnakeBeta
    @ModuleInfo(key: "i1") var up: WNConvTranspose1d
    @ModuleInfo(key: "i2") var res1: ResidualUnit
    @ModuleInfo(key: "i3") var res3: ResidualUnit
    @ModuleInfo(key: "i4") var res9: ResidualUnit
    init(_ inC: Int, _ outC: Int, stride: Int) {
        _act.wrappedValue = SnakeBeta(inC)
        _up.wrappedValue = WNConvTranspose1d(inC, outC, kernel: 2 * stride, stride: stride, padding: (stride + 1) / 2)
        _res1.wrappedValue = ResidualUnit(outC, dilation: 1)
        _res3.wrappedValue = ResidualUnit(outC, dilation: 3)
        _res9.wrappedValue = ResidualUnit(outC, dilation: 9)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { res9(res3(res1(up(act(x))))) }
}

public final class DecoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: DecoderBlockLayers
    init(_ inC: Int, _ outC: Int, stride: Int) { _layers.wrappedValue = DecoderBlockLayers(inC, outC, stride: stride) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { layers(x) }
}

final class OobleckDecoderLayers: Module {
    @ModuleInfo(key: "i0") var convPre: WNConv1d
    @ModuleInfo(key: "i1") var b1: DecoderBlock
    @ModuleInfo(key: "i2") var b2: DecoderBlock
    @ModuleInfo(key: "i3") var b3: DecoderBlock
    @ModuleInfo(key: "i4") var b4: DecoderBlock
    @ModuleInfo(key: "i5") var b5: DecoderBlock
    @ModuleInfo(key: "i6") var b6: DecoderBlock
    @ModuleInfo(key: "i7") var actPost: SnakeBeta
    @ModuleInfo(key: "i8") var convPost: WNConv1d

    init(_ c: OobleckDecoderConfig) {
        precondition(c.c_mults.count == 6 && c.strides.count == 6 && c.use_snake, "the released decoder: 6 blocks, SnakeBeta")
        let m = [1] + c.c_mults                       // depth 7
        let ch = m.map { $0 * c.channels }
        _convPre.wrappedValue = WNConv1d(c.latent_dim, ch[6], kernel: 7, padding: 3)
        // DecoderBlock i (6…1): in m[i]·C → out m[i−1]·C, stride strides[i−1]
        _b1.wrappedValue = DecoderBlock(ch[6], ch[5], stride: c.strides[5])
        _b2.wrappedValue = DecoderBlock(ch[5], ch[4], stride: c.strides[4])
        _b3.wrappedValue = DecoderBlock(ch[4], ch[3], stride: c.strides[3])
        _b4.wrappedValue = DecoderBlock(ch[3], ch[2], stride: c.strides[2])
        _b5.wrappedValue = DecoderBlock(ch[2], ch[1], stride: c.strides[1])
        _b6.wrappedValue = DecoderBlock(ch[1], ch[0], stride: c.strides[0])
        _actPost.wrappedValue = SnakeBeta(ch[0])
        _convPost.wrappedValue = WNConv1d(ch[0], c.out_channels, kernel: 7, padding: 3, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convPost(actPost(b6(b5(b4(b3(b2(b1(convPre(x)))))))))
    }
}

public final class OobleckDecoder: Module {
    @ModuleInfo(key: "layers") var layers: OobleckDecoderLayers
    let finalTanh: Bool
    init(_ c: OobleckDecoderConfig) { _layers.wrappedValue = OobleckDecoderLayers(c); finalTanh = c.final_tanh ?? true }
    func callAsFunction(_ x: MLXArray) -> MLXArray { let y = layers(x); return finalTanh ? tanh(y) : y }
}

/// `YuE2VAE` (decoder only). Latents (T, 64) fp32 → audio (S, 2) fp32, S = 1920·T − 64 for the released strides.
public final class YuE2VAE: Module {
    public let config: YuE2VAEConfig
    @ModuleInfo(key: "decoder") var decoder: OobleckDecoder

    public init(_ c: YuE2VAEConfig) { config = c; _decoder.wrappedValue = OobleckDecoder(c.decoder_config) }

    /// Full decode, no clipping. latents (T, 64) → (S, 2).
    public func decode(_ latents: MLXArray) -> MLXArray {
        decoder(latents.asType(.float32).expandedDimensions(axis: 0))[0]
    }

    /// `decode_tiled`: cores of `coreFrames` with `haloFrames` of context each side, cropped exactly, no blending.
    public func decodeTiled(_ latents: MLXArray, coreFrames: Int? = nil, haloFrames: Int? = nil,
                            onTile: ((Int, Int) -> Void)? = nil) -> MLXArray {
        let core = coreFrames ?? config.decode_core_frames, halo = haloFrames ?? config.decode_halo_frames
        let frames = latents.dim(0), ratio = config.downsampling_ratio
        let tiles = (frames + core - 1) / core
        if tiles <= 1 { let y = decode(latents); eval(y); onTile?(1, 1); return y }
        var total = -1
        var parts = [MLXArray]()
        var tileIndex = 0
        for start in stride(from: 0, to: frames, by: core) {
            let end = min(frames, start + core)
            let left = max(0, start - halo), right = min(frames, end + halo)
            let tile = decode(latents[left ..< right])
            if total < 0 {
                // natural output length of the whole song, from one tile's own length rule (linear in frames)
                let perFrame = ratio, bias = tile.dim(0) - (right - left) * perFrame
                total = frames * perFrame + bias
            }
            let outStart = start * ratio, outEnd = min(end * ratio, total)
            let cropStart = (start - left) * ratio
            let crop = tile[cropStart ..< (cropStart + outEnd - outStart)]
            eval(crop)
            parts.append(crop)
            tileIndex += 1
            onTile?(tileIndex, tiles)
        }
        return concatenated(parts, axis: 0)
    }
}
