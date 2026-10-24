// SongProtocol.swift — src/yue2/protocol.py: the checkpoint-native token protocol, sampling defaults, request shape,
// prefixes and chunking. Numbers are the release's (`yue2-native-v1`).
import Foundation

public enum YuE2Protocol {
    public static let eod = 151643
    public static let abcStart = 151847, abcEnd = 151848
    public static let musicStart = 151851, musicEnd = 151852
    public static let codecOffset = 151853, codecSize = 32768
    public static let latentStart = 184621, latentEnd = 184622, latentPad = 184623
    public static let vocabSize = 184704, context = 24576
    public static let version = "yue2-native-v1"

    public static func instruction(_ mode: PlanningMode) -> String {
        switch mode {
        case .off: return "Generate music with codec tokens from the given conditions."
        case .melody: return "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."
        case .full: return "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
        }
    }

    /// `chunk_ranges`: frames per NAR chunk = (context − prefix − 3) // 2.
    public static func chunkRanges(frames: Int, prefixTokens: Int, context: Int = context) throws -> [Range<Int>] {
        let size = min((context - prefixTokens - 3) / 2, Self.context)
        guard frames >= 1, size >= 1 else { throw YuE2Error.invalidInput("empty codec or prefix leaves no acoustic context") }
        return stride(from: 0, to: frames, by: size).map { $0 ..< min($0 + size, frames) }
    }
}

public enum PlanningMode: String, Codable, Sendable { case off, melody, full }

/// One sampling phase's knobs (`protocol.Sampling`).
public struct SamplingConfig: Codable, Sendable, Equatable {
    public var temperature: Float = 1.0
    public var top_p: Float = 0.95
    public var top_k: Int = 100
    public var repetition_penalty: Float = 1.2
    public var penalty_window: Int = 50
    public var min_tokens: Int = 200
    public var max_tokens: Int = 9000

    public init(temperature: Float = 1.0, top_p: Float = 0.95, top_k: Int = 100, repetition_penalty: Float = 1.2,
                penalty_window: Int = 50, min_tokens: Int = 200, max_tokens: Int = 9000) {
        self.temperature = temperature; self.top_p = top_p; self.top_k = top_k; self.repetition_penalty = repetition_penalty
        self.penalty_window = penalty_window; self.min_tokens = min_tokens; self.max_tokens = max_tokens
    }
    public static let abcDefault = SamplingConfig(temperature: 0.7, top_p: 0.9, top_k: 30, repetition_penalty: 1.005,
                                                  penalty_window: 100, min_tokens: 32, max_tokens: 4096)
    public static let semanticDefault = SamplingConfig()
}

/// `protocol.GenerationConfig` (the bundle's `mlx_yue2.generation`).
public struct GenerationConfig: Codable, Sendable, Equatable {
    public var abc: SamplingConfig = .abcDefault
    public var semantic: SamplingConfig = .semanticDefault
    public var ode_steps: Int = 32
    public var ode_method: String = "midpoint"
    public var context: Int = YuE2Protocol.context
    public var version: String = YuE2Protocol.version
    public init() {}
}

/// `protocol.SongRequest`.
public struct SongRequest: Sendable {
    public var style: String
    public var lyrics: String
    public var mode: PlanningMode
    public var seed: UInt64
    /// A supplied ABC score (full or melody only; never with `.off`).
    public var abc: String?
    /// CFG scale; nil → the mode's default (1.0 full/melody, 1.01 off).
    public var cfgScale: Float?

    public init(style: String, lyrics: String, mode: PlanningMode = .full, seed: UInt64 = 831001, abc: String? = nil,
                cfgScale: Float? = nil) throws {
        guard abc == nil || mode != .off else { throw YuE2Error.invalidInput("a supplied score needs full or melody mode") }
        if let c = cfgScale, !(c.isFinite && c >= 0 && c <= 20) { throw YuE2Error.invalidInput("cfg_scale must be in [0, 20]") }
        self.style = style; self.lyrics = lyrics; self.mode = mode; self.seed = seed; self.abc = abc; self.cfgScale = cfgScale
    }

    public var guidance: Float { cfgScale ?? (mode == .off ? 1.01 : 1.0) }
    public var text: String { "\(YuE2Protocol.instruction(mode))\n[Tags]\n\(style)\n[Lyrics]\n\(lyrics)\n" }

    /// `token_prefixes`.
    public func prefix(_ tok: YuE2Tokenizer, abcIDs: [Int]?) throws -> [Int] {
        let base = [YuE2Protocol.eod] + tok.encode(text)
        if mode == .off { return base + [YuE2Protocol.abcStart, YuE2Protocol.abcEnd, YuE2Protocol.musicStart] }
        var ids = abcIDs
        if ids == nil {
            guard let abc else { return base + [YuE2Protocol.abcStart] }
            ids = tok.encode(abc)
        }
        guard ids!.allSatisfy({ $0 >= 0 && $0 < YuE2Protocol.eod }) else { throw YuE2Error.invalidInput("ABC ids must stay in the text vocabulary") }
        return base + [YuE2Protocol.abcStart] + ids! + [YuE2Protocol.abcEnd, YuE2Protocol.musicStart]
    }

    /// `negative_prefix`: the CFG branch — instruction only, plus the SAME abc ids.
    public func negativePrefix(_ tok: YuE2Tokenizer, abcIDs: [Int]) throws -> [Int] {
        let base = [YuE2Protocol.eod] + tok.encode(YuE2Protocol.instruction(mode))
        if mode == .off { return base + [YuE2Protocol.musicStart] }
        return base + [YuE2Protocol.abcStart] + abcIDs + [YuE2Protocol.abcEnd, YuE2Protocol.musicStart]
    }
}
