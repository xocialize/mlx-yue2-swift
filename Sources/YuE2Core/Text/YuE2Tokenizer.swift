// YuE2Tokenizer.swift — src/yue2/tokenization_yue2.py: a tiktoken byte-level BPE over the checkpoint's `qwen.tiktoken`
// (151 643 mergeable ranks), the Qwen pre-tokenizer pattern, NFC normalisation, `encode_ordinary` only (special tokens
// are never produced from text; the protocol inserts them by id). The 208 specials ids follow the ranks; `<abc>` /
// `</abc>` sit at 151847 / 151848 (specials[204:206]).
//
// Algorithm = tiktoken's (as in the IndexTTS-2.5 Swift port, gated id-exact there): pre-tokenize with the ICU form of
// the pattern, then byte-pair-merge each piece by lowest rank first. Gated id-exact here on the goldens' strings and
// on the oracle's ABC round trip (lane T).

import Foundation

public final class YuE2Tokenizer: @unchecked Sendable {
    public static let ordinaryCount = 151643
    private let ranks: [[UInt8]: Int]
    private let bytesByRank: [Int: [UInt8]]
    private let pattern: NSRegularExpression
    public let specialTokens: [String: Int]
    private let specialByID: [Int: String]
    public let vocabularySize: Int

    static let patternString =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    static let specialStrings: [String] = {
        var s = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
        s += (0 ..< 200).map { "<extra_\($0)>" }
        s[204] = "<abc>"; s[205] = "</abc>"
        return s
    }()

    public init(vocabularyURL: URL) throws {
        guard let text = try? String(contentsOf: vocabularyURL, encoding: .utf8) else {
            throw YuE2Error.missingFile(vocabularyURL.path)
        }
        var ranks: [[UInt8]: Int] = [:]
        var byRank: [Int: [UInt8]] = [:]
        for (n, line) in text.split(separator: "\n", omittingEmptySubsequences: true).enumerated() {
            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            guard parts.count == 2, let rank = Int(parts[1]) else { throw YuE2Error.badConfig("qwen.tiktoken line \(n + 1)") }
            let b64 = String(parts[0])
            let bytes: [UInt8]
            if b64.allSatisfy({ $0 == "=" }) { bytes = [] }
            else { guard let d = Data(base64Encoded: b64) else { throw YuE2Error.badConfig("qwen.tiktoken line \(n + 1)") }; bytes = [UInt8](d) }
            ranks[bytes] = rank; byRank[rank] = bytes
        }
        guard ranks.count == Self.ordinaryCount else {
            throw YuE2Error.badConfig("expected 151643 ordinary ranks, got \(ranks.count)")
        }
        self.ranks = ranks; self.bytesByRank = byRank
        var specials: [String: Int] = [:], byID: [Int: String] = [:]
        for (i, s) in Self.specialStrings.enumerated() { specials[s] = Self.ordinaryCount + i; byID[Self.ordinaryCount + i] = s }
        specialTokens = specials; specialByID = byID
        vocabularySize = Self.ordinaryCount + Self.specialStrings.count
        pattern = try NSRegularExpression(pattern: Self.patternString)
    }

    /// `encode_ordinary(NFC(text))`.
    public func encode(_ text: String) -> [Int] {
        let nfc = text.precomposedStringWithCanonicalMapping
        var ids: [Int] = []
        let ns = nfc as NSString
        for match in pattern.matches(in: nfc, range: NSRange(location: 0, length: ns.length)) {
            let piece = [UInt8](ns.substring(with: match.range).utf8)
            if let whole = ranks[piece] { ids.append(whole) } else { ids += bytePairMerge(piece) }
        }
        return ids
    }

    /// `decode`: ordinary ids → bytes → UTF-8 (lossy replacement); specials by string; ids outside the vocabulary dropped.
    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        var out = ""
        func flush() { if !bytes.isEmpty { out += String(decoding: bytes, as: UTF8.self); bytes.removeAll() } }
        for id in ids {
            if let b = bytesByRank[id] { bytes += b }
            else if let s = specialByID[id] { flush(); out += s }
        }
        flush()
        return out
    }

    private func bytePairMerge(_ piece: [UInt8]) -> [Int] {
        var parts: [(start: Int, rank: Int)] = []
        parts.reserveCapacity(piece.count + 1)
        for i in 0 ..< piece.count {
            let rank = i + 1 < piece.count ? (ranks[Array(piece[i ... i + 1])] ?? Int.max) : Int.max
            parts.append((i, rank))
        }
        parts.append((piece.count, Int.max))
        func pairRank(_ index: Int) -> Int {
            guard index + 2 < parts.count else { return Int.max }
            return ranks[Array(piece[parts[index].start ..< parts[index + 2].start])] ?? Int.max
        }
        while parts.count > 2 {
            var best = Int.max, bestIndex = -1
            for i in 0 ..< parts.count - 1 where parts[i].rank < best { best = parts[i].rank; bestIndex = i }
            if bestIndex < 0 { break }
            parts.remove(at: bestIndex + 1)
            parts[bestIndex].rank = pairRank(bestIndex)
            if bestIndex > 0 { parts[bestIndex - 1].rank = pairRank(bestIndex - 1) }
        }
        var out: [Int] = []
        for i in 0 ..< parts.count - 1 { out.append(ranks[Array(piece[parts[i].start ..< parts[i + 1].start])]!) }
        return out
    }
}
