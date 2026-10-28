// yue2-gates — parity gates against the fp32 torch oracle (WIP/yue2-port/oracle/dump_goldens.py), PORTING-SPEC §3.
//
//   yue2-gates tokenizer <golden> <model>                 lane T: ids on the goldens' strings, abc round trip
//   yue2-gates ar        <golden> <model>                 lane A: teacher-forced logits (argmax agreement, tail relL2)
//   yue2-gates nar       <golden> <model>                 lanes N1 (velocity at t=1, t=0.5) + N2 (32-step chunk)
//   yue2-gates vae       <golden> <vae>                   lane D: decode of the oracle's chunk latents, full + tiled
//   yue2-gates e2e       <golden> <model> <vae> [out.wav] lane E: oracle tokens + noise → latents → audio
//   yue2-gates render    <model> <vae> <request.json> <out.wav>   a real generation (seeded; not a gate)
//   yue2-gates validate  <model> <vae> <request.json> <outdir>    V3: MLXServeEngine register → prepare → run (the song, then its
//                                                      own planned score re-rendered) → refusals → evict, phys_footprint at each step
//   yue2-gates envelope  <model> <vae> <score.abc>     V3: the input envelope — a 4 096-id score prefix + 9 000 codec frames (360 s)
//                                                      through the NAR prefill, the 32-step solve and the VAE; phys_footprint peak
//
// Options (before the subcommand): --cpu, --dtype float32|bfloat16, --ar-bits N, --nar-bits N; for validate / envelope:
// --tier bf16|ar8|int8 (the package tiers; sets the bits), --cache-gb G (MLX cache limit, default 1), --store <models root>
// (validate: resolve the pinned published repos under that root instead of the explicit <model> <vae> folders).
import Darwin
import Foundation
import MLX
import MLXNN
import MLXServeCore
import MLXToolKit
import MLXYuE2
import YuE2Core

var args = Array(CommandLine.arguments.dropFirst())
var useCPU = false
var dtype: DType = .float32
var quant = YuE2Quantization.none
var tier = YuE2Tier.bf16
var cacheGB = 1.0
var storeRoot: URL? = nil
while let a = args.first, a.hasPrefix("--") {
    args.removeFirst()
    switch a {
    case "--cpu": useCPU = true
    case "--dtype": dtype = ["bfloat16", "bf16"].contains(args.removeFirst()) ? .bfloat16 : .float32
    case "--ar-bits": quant.arBits = Int(args.removeFirst())
    case "--nar-bits": quant.narBits = Int(args.removeFirst())
    case "--tier": tier = YuE2Tier(rawValue: args.removeFirst())!; quant = tier.quantization
    case "--cache-gb": cacheGB = Double(args.removeFirst())!
    case "--store": storeRoot = URL(fileURLWithPath: args.removeFirst())
    default: fatalError("unknown option \(a)")
    }
}
guard let cmd = args.first else { print("usage: yue2-gates [--cpu] [--dtype d] [--tier t] tokenizer|ar|nar|vae|e2e|render|validate|envelope …"); exit(2) }
let rest = Array(args.dropFirst())
if useCPU { Device.setDefault(device: Device(.cpu)) }

struct Row { let name: String; let maxAbs: Float; let relL2: Float; let cos: Float }
var rows = [Row]()
func compare(_ name: String, _ got: MLXArray, _ ref: MLXArray) {
    let g = got.asType(.float32), r = ref.asType(.float32)
    guard g.shape == r.shape else {
        print(String(format: "  %-24@ SHAPE MISMATCH got %@ ref %@", name as NSString, "\(g.shape)" as NSString, "\(r.shape)" as NSString))
        rows.append(Row(name: name, maxAbs: .infinity, relL2: .infinity, cos: 0)); return
    }
    let d = g - r
    let maxAbs = abs(d).max().item(Float.self)
    let relL2 = (sqrt((d * d).sum()) / maximum(sqrt((r * r).sum()), MLXArray(Float(1e-30)))).item(Float.self)
    let cos = ((g * r).sum() / maximum(sqrt((g * g).sum()) * sqrt((r * r).sum()), MLXArray(Float(1e-30)))).item(Float.self)
    rows.append(Row(name: name, maxAbs: maxAbs, relL2: relL2, cos: cos))
    print(String(format: "  %-24@ max|Δ| %.3e   relL2 %.3e   cos %.7f   %@", name as NSString, maxAbs, relL2, cos, "\(g.shape)" as NSString))
}
func npy(_ dir: URL, _ name: String) throws -> MLXArray { try NPY.load(dir.appendingPathComponent("\(name).npy")) }
func ints(_ dir: URL, _ name: String) throws -> [Int] { try npy(dir, name).asArray(Int32.self).map { Int($0) } }
func url(_ s: String) -> URL { URL(fileURLWithPath: s) }
func elapsed(_ t: Date) -> String { String(format: "%.1fs", Date().timeIntervalSince(t)) }

struct TokGolden: Decodable { let strings: [String]; let ids: [[Int]] }
struct NarMeta: Decodable { let raw_t1: Double; let raw_t05: Double; let frames: Int; let ar_len: Int }
struct RequestJSON: Decodable { let style: String; let lyrics: String; let cot: String; let seed: Int; let cfg_scale: Float? }

func tokenizerGate(_ golden: URL, _ modelDir: URL) throws {
    let tok = try YuE2Tokenizer(vocabularyURL: modelDir.appendingPathComponent("qwen.tiktoken"))
    let g = try JSONDecoder().decode(TokGolden.self, from: Data(contentsOf: golden.appendingPathComponent("tokenizer.json")))
    var ok = 0
    for (s, ids) in zip(g.strings, g.ids) {
        let mine = tok.encode(s)
        let same = mine == ids
        ok += same ? 1 : 0
        print("  string \(ids.count) ids: \(same ? "EXACT" : "MISMATCH (mine \(mine.count))")")
        if !same { for (i, (a, b)) in zip(mine, ids).enumerated() where a != b { print("    first diff at \(i): mine \(a) ref \(b)"); break } }
    }
    if FileManager.default.fileExists(atPath: golden.appendingPathComponent("abc.txt").path) {
        let abc = try String(contentsOf: golden.appendingPathComponent("abc.txt"), encoding: .utf8)
        let ids = try ints(golden, "abc_ids")
        let enc = tok.encode(abc), dec = tok.decode(ids)
        print("  abc encode(abc) == abc_ids: \(enc == ids);  decode(abc_ids) == abc: \(dec == abc)")
        ok += (enc == ids && dec == abc) ? 1 : 0
    }
    let prefix = try ints(golden, "prefix")
    let req = try JSONDecoder().decode(RequestJSON.self, from: Data(contentsOf: golden.appendingPathComponent("request.json")))
    let sr = try SongRequest(style: req.style, lyrics: req.lyrics, mode: PlanningMode(rawValue: req.cot)!, seed: UInt64(req.seed))
    let abcIDs = FileManager.default.fileExists(atPath: golden.appendingPathComponent("abc.txt").path) ? try ints(golden, "abc_ids") : nil
    let mine = try sr.prefix(tok, abcIDs: sr.mode == .off ? nil : abcIDs)
    print("  prefix (\(prefix.count) ids): \(mine == prefix ? "EXACT" : "MISMATCH (mine \(mine.count))")")
    print("LANE T: \(ok == g.strings.count + (abcIDs == nil ? 0 : 1) && mine == prefix ? "PASS" : "FAIL")")
}

func arGate(_ golden: URL, _ modelDir: URL) throws {
    let t = Date()
    let (_, model, _) = try YuE2Pipeline.loadModel(modelDir, dtype: dtype); quant.apply(model)
    print("  loaded in \(elapsed(t))")
    var pass = true
    for tag in ["pos", "neg"] where FileManager.default.fileExists(atPath: golden.appendingPathComponent("ar_tf_\(tag)_ids.npy").path) {
        let ids = try ints(golden, "ar_tf_\(tag)_ids")
        let t1 = Date()
        let h = model.forwardAR(ids: MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0), caches: nil)
        let logits = model.lmHead(h)[0]                                 // (L, vocab)
        let argmax = argMax(logits, axis: -1)
        let refArg = try npy(golden, "ar_tf_\(tag)_argmax")
        let agree = (argmax .== refArg).sum().item(Int.self)
        print("  [\(tag)] L=\(ids.count) forward \(elapsed(t1)); argmax agreement \(agree)/\(ids.count)")
        let tail = try npy(golden, "ar_tf_\(tag)_logits_tail")
        compare("\(tag)_logits_tail", logits[(ids.count - tail.dim(0))...], tail)
        pass = pass && agree == ids.count && rows.last!.relL2 <= 1e-3
    }
    print("LANE A: \(pass ? "PASS" : "FAIL")")
}

func narGate(_ golden: URL, _ modelDir: URL) throws {
    let t = Date()
    let (_, model, _) = try YuE2Pipeline.loadModel(modelDir, dtype: dtype); quant.apply(model)
    print("  loaded in \(elapsed(t))")
    let meta = try JSONDecoder().decode(NarMeta.self, from: Data(contentsOf: golden.appendingPathComponent("nar_meta.json")))
    let ids = try ints(golden, "nar_chunk_ids")
    let noise = try npy(golden, "nar_state_t1")
    let t1 = Date()
    let engine = try NARSolver.CachedNAR(model: model, chunk: NARChunk(arTokens: ids, noise: noise))
    print("  prefill A=\(ids.count) in \(elapsed(t1))")
    compare("v_t1", engine.velocity(noise, rawT: meta.raw_t1), try npy(golden, "nar_v_t1"))
    compare("v_t05", engine.velocity(try npy(golden, "nar_state_t05"), rawT: meta.raw_t05), try npy(golden, "nar_v_t05"))
    let t2 = Date()
    let lat = try engine.solve(steps: 32)
    print("  32-step solve (\(meta.frames) frames) in \(elapsed(t2))")
    compare("latents_32", lat, try npy(golden, "nar_latents"))
    let pass = rows[0].relL2 <= 1e-3 && rows[1].relL2 <= 1e-3 && rows[2].relL2 <= 5e-3
    print("LANE N: \(pass ? "PASS" : "FAIL")")
}

func vaeGate(_ golden: URL, _ vaeDir: URL) throws {
    let vae = try YuE2Pipeline.loadVAE(vaeDir)
    let lat = try npy(golden, "nar_latents")
    let full = vae.decode(lat)
    compare("audio_full", full.transposed(1, 0), try npy(golden, "vae_audio_full"))
    let tiled = try vae.decodeTiled(lat, coreFrames: 64, haloFrames: 16)
    compare("audio_tiled64", tiled.transposed(1, 0), try npy(golden, "vae_audio_tiled64"))
    compare("tiled_vs_full", tiled, full)
    let pass = rows[0].relL2 <= 1e-3 && rows[1].relL2 <= 1e-3 && rows[2].relL2 <= 1e-4   // tiled == full up to GPU conv-kernel rounding (length-dependent kernels; ~1e-5 abs)
    print("LANE D: \(pass ? "PASS" : "FAIL")")
}

func e2eGate(_ golden: URL, _ modelDir: URL, _ vaeDir: URL, _ out: String?) throws {
    let t = Date()
    let pipe = try YuE2Pipeline.load(modelDir: url(modelDir.path), vaeDir: vaeDir, dtype: dtype, quantization: quant)
    print("  loaded in \(elapsed(t))")
    let prefix = try ints(golden, "prefix"), codec = try ints(golden, "semantic")
    let noise = try npy(golden, "song_noise")
    let t1 = Date()
    let lat = try NARSolver.synthesize(model: pipe.model, prefix: prefix, codec: codec, noise: noise, seed: 0, steps: pipe.generation.ode_steps)
    eval(lat)
    print("  NAR (\(codec.count) frames, prefix \(prefix.count)) in \(elapsed(t1))")
    compare("latents_full", lat, try npy(golden, "latents_full"))
    let t2 = Date()
    let audio = try pipe.decode(lat)
    eval(audio)
    print("  VAE in \(elapsed(t2))")
    compare("audio_ref", audio.transposed(1, 0), try npy(golden, "audio_ref"))
    if let out { try WAV.write16(audio.asArray(Float.self), channels: audio.dim(1), sampleRate: 48000, to: url(out)); print("  wrote \(out)") }
    let pass = rows[0].relL2 <= 5e-3 && rows[1].relL2 <= 1e-2
    print("LANE E: \(pass ? "PASS" : "FAIL")")
}

func render(_ modelDir: URL, _ vaeDir: URL, _ reqPath: URL, _ out: URL) throws {
    let t = Date()
    let pipe = try YuE2Pipeline.load(modelDir: modelDir, vaeDir: vaeDir, dtype: dtype, quantization: quant)
    print("  loaded in \(elapsed(t)) (\(dtype), ar \(quant.arBits.map(String.init) ?? "-") / nar \(quant.narBits.map(String.init) ?? "-") bits)")
    let r = try JSONDecoder().decode(RequestJSON.self, from: Data(contentsOf: reqPath))
    let req = try SongRequest(style: r.style, lyrics: r.lyrics, mode: PlanningMode(rawValue: r.cot)!, seed: UInt64(r.seed), cfgScale: r.cfg_scale)
    var last = ""
    let song = try pipe.generate(req) { p in
        let s: String
        switch p {
        case .plan(let n): s = n % 100 == 0 ? "plan \(n)" : ""
        case .semantic(let n): s = n % 100 == 0 ? "semantic \(n)" : ""
        case .synthesize(let a, let b): s = a % 8 == 0 ? "synth \(a)/\(b)" : ""
        case .decode(let a, let b): s = "decode \(a)/\(b)"
        }
        if !s.isEmpty, s != last { print("  \(s)"); last = s }
    }
    let peakGB = Double(Memory.peakMemory) / 1e9, activeGB = Double(Memory.activeMemory) / 1e9
    print(String(format: "  memory: peak %.2f GB  active %.2f GB (MLX-accounted; phys_footprint is the engine's number)", peakGB, activeGB))
    try WAV.write16(song.audio, channels: song.channels, sampleRate: song.sampleRate, to: out)
    if let abc = song.plan.abc { try abc.write(to: out.deletingPathExtension().appendingPathExtension("abc"), atomically: true, encoding: .utf8) }
    let secs = Double(song.audio.count / song.channels) / Double(song.sampleRate)
    print(String(format: "  song %.1fs  abc %d ids  codec %d  truncated %@  timing %@", secs, song.plan.abcIDs.count, song.codec.count,
                 "\(song.truncated)" as NSString, "\(song.timing.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.1f", $0.value))" })" as NSString))
    print("  wrote \(out.path)")
}

// MARK: - validate (V3): the consumer path — MLXServeEngine (default policy, advisory enforcement) register → prepare → run →
// refusals → evict, phys_footprint at each step (the manifest's C-memory numbers). Bundles are given explicitly (the engine's MAT
// path is not exercised: nothing is published before V4).

var failures = 0
func check(_ name: String, _ ok: Bool, _ detail: String = "") { if !ok { failures += 1 }; print("  \(name): \(ok ? "PASS" : "FAIL") \(detail)") }
func gb(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / 1e9) }
func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1e9) }

func pcm16(_ audio: Audio) -> [Float] {
    audio.data.dropFirst(44).withUnsafeBytes { raw -> [Float] in raw.bindMemory(to: Int16.self).map { Float($0) / 32768 } }
}

func runValidate(_ modelDir: URL, _ vaeDir: URL, _ reqPath: URL, _ outDir: URL) async throws {
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let r = try JSONDecoder().decode(RequestJSON.self, from: Data(contentsOf: reqPath))
    print("validate: MLXServeEngine → register → prepare → run (tier \(tier.rawValue), quant \(tier.quant)) → refusals → evict")
    let sampler = Footprint.Sampler()
    let before = sampler.mark()
    let engine = MLXServeEngine()   // the default policy (.permissiveOnly, .advisory): CC-BY-NC-4.0 registers with ONE advisory
    if let storeRoot { await engine.useModelStore(ModelStore(root: storeRoot)) }   // the host's models root (the engine stamps it on the config)
    let t0 = Date()
    // --store: no explicit folders — the package resolves its pinned repos' snapshots under the models root (the published path)
    let configuration = storeRoot.map { YuE2Configuration(tier: tier, modelsRootDirectory: $0, cacheLimitBytes: UInt64(cacheGB * Double(1 << 30))) }
        ?? YuE2Configuration(tier: tier, modelDirectory: modelDir, vaeDirectory: vaeDir, cacheLimitBytes: UInt64(cacheGB * Double(1 << 30)))
    if let storeRoot { print("  store \(storeRoot.path): resolves \(configuration.resolvedBundles(storeRoot: storeRoot).map { "\($0.model.path) + \($0.vae.path)" } ?? "NOTHING")") }
    let id = try await engine.register(YuE2Package.registration, configuration: configuration, id: PackageID("yue2-3b"))
    let advisories = await engine.licenseAdvisories
    check("registered \(id.rawValue); licence advisories: \(advisories.count)",
          advisories.count == 1 && advisories[0].license == "CC-BY-NC-4.0" && advisories[0].layer == .weight,
          advisories.map(\.summary).joined(separator: "; "))
    let needs = await engine.needsDownload(.music, package: id)
    check("needsDownload == \(needs) with the \(storeRoot == nil ? "explicit bundles" : "downloaded store")", !needs)
    try await engine.prepare(.music, package: id)
    Thread.sleep(forTimeInterval: 0.3)
    let afterLoad = sampler.mark()
    let charged = await engine.memory.residentBytes
    print(String(format: "  [VAL] prepared in %.1fs; phys_footprint %@ → %@ (resident +%@); MLX active %@; engine charges %@",
                 Date().timeIntervalSince(t0), gb(before) as NSString, gb(afterLoad) as NSString, gb(Int(afterLoad) - Int(before)) as NSString,
                 gb(Memory.activeMemory) as NSString, gb(charged) as NSString))
    var peak = afterLoad
    var afterSong = afterLoad
    var planned: String? = nil
    // run 1: the request as written (cot full → a planned score); run 2: that score supplied back (the edit-and-re-render loop)
    for pass in 0 ..< 2 {
        if pass == 1, planned == nil { break }
        sampler.resetPeak()
        let request = MusicRequest(caption: r.style, lyrics: r.lyrics, seed: UInt64(r.seed),
                                   guidanceScale: r.cfg_scale.map(Double.init), planning: r.cot == "off" ? .off : .full,
                                   score: pass == 1 ? planned : nil)
        var phases: [String: Int] = [:]
        let t = Date()
        let response = try await RunProgress.$sink.withValue({ report in
            phases[report.phase.rawValue, default: 0] += 1
            if report.phase == .generate, let n = report.step, n % 1000 == 0 { FileHandle.standardError.write("  generate \(n)\n".data(using: .utf8)!) }
        }) {
            try await engine.run(request, package: id)
        }
        let wall = Date().timeIntervalSince(t)
        Thread.sleep(forTimeInterval: 0.3)
        peak = max(peak, sampler.peak)
        afterSong = sampler.mark()
        guard let m = response as? MusicResponse else { check("MusicResponse", false, "got \(type(of: response))"); return }
        let samples = pcm16(m.audio)
        let secs = Double(samples.count) / Double((m.audio.sampleRate ?? 48_000) * (m.audio.channels ?? 2))
        let rms = 20 * log10(max(sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(max(samples.count, 1))), 1e-9))
        let peakDB = 20 * log10(max(samples.map(abs).max() ?? 0, 1e-9))
        let label = pass == 0 ? "song" : "re-render of its score"
        let wantsScore = r.cot != "off"
        check(String(format: "  [RUN] %@: %.2fs audio (%d Hz × %d ch), peak %.1f / rms %.1f dBFS, %.1fs wall (RTF %.2f), truncated %@, score %@ chars, phases %@, phys peak %@ after %@",
                     label as NSString, secs, m.audio.sampleRate ?? 0, m.audio.channels ?? 0, peakDB, rms, wall, wall / max(secs, 1e-9),
                     "\(m.truncated)" as NSString, "\(m.score?.count ?? 0)" as NSString, "\(phases.sorted { $0.key < $1.key })" as NSString,
                     gb(sampler.peak) as NSString, gb(afterSong) as NSString),
              secs > 10 && secs <= YuE2Package.maxDurationSeconds && rms > -40 && rms < -3 && m.audio.sampleRate == 48_000 && m.audio.channels == 2
                  && (phases["generate"] ?? 0) >= 200 && (phases["denoise"] ?? 0) >= 32 && (phases["decode"] ?? 0) >= 1
                  && (phases["postprocess"] ?? 0) == 1 && (pass == 1 ? (phases["plan"] ?? 0) == 0 : (phases["plan"] ?? 0) > (wantsScore ? 31 : -1))
                  && (wantsScore ? m.score != nil : m.score == nil) && (pass == 1 ? m.score == planned : true))
        try m.audio.data.write(to: outDir.appendingPathComponent("engine_\(tier.rawValue)_\(pass == 0 ? "song" : "rerender").wav"))
        if let score = m.score { try score.write(to: outDir.appendingPathComponent("engine_\(tier.rawValue)_\(pass == 0 ? "song" : "rerender").abc"), atomically: true, encoding: .utf8) }
        if pass == 0 { planned = m.score }
    }
    // the engine refuses an undeclared feature / out-of-vocabulary value BEFORE admission (MusicControls.refusal); the package
    // refuses its own bounds (empty lyrics, score + off, cfg range) inside run()
    for (label, bad) in [
        ("planning melody (undeclared mode)", MusicRequest(caption: r.style, lyrics: r.lyrics, planning: .melody)),
        ("edit/cover (undeclared)", MusicRequest(caption: r.style, lyrics: r.lyrics, edit: MusicEdit(source: Audio(format: .wav, data: Data([1, 2, 3])), operation: .cover(strength: 0.5)))),
        ("referenceAudio (undeclared)", MusicRequest(caption: r.style, lyrics: r.lyrics, referenceAudio: Audio(format: .wav, data: Data([1, 2, 3])))),
        ("vocalLanguage en (undeclared)", MusicRequest(caption: r.style, lyrics: r.lyrics, vocalLanguage: "en")),
        ("anchors bpm 88 (undeclared)", MusicRequest(caption: r.style, lyrics: r.lyrics, anchors: MusicAnchors(bpm: 88))),
        ("durationSeconds 400 (> 360)", MusicRequest(caption: r.style, lyrics: r.lyrics, durationSeconds: 400)),
        ("empty lyrics", MusicRequest(caption: r.style, lyrics: " ")),
        ("score with planning off", MusicRequest(caption: r.style, lyrics: r.lyrics, planning: .off, score: "X:1\nK:C\nCDEF|")),
        ("guidanceScale 25", MusicRequest(caption: r.style, lyrics: r.lyrics, guidanceScale: 25)),
    ] {
        do { _ = try await engine.run(bad, package: id); check("  refused: \(label)", false, "ADMITTED") }
        catch { check("  refused: \(label)", error is PackageError, String(describing: error).prefix(110).description) }
    }
    await engine.evict(package: id)
    Thread.sleep(forTimeInterval: 0.3)
    let afterEvict = sampler.mark()
    _ = sampler.stop()
    print(String(format: "  [VAL] %@: resident %@ (phys after prepare − before), %@ after the songs; peak phys %@ (activation %@ over the post-prepare resident); after evict %@; MLX peak %@",
                 tier.rawValue as NSString, gb(Int(afterLoad) - Int(before)) as NSString, gb(Int(afterSong) - Int(before)) as NSString, gb(peak) as NSString,
                 gb(Int(peak) - Int(afterLoad)) as NSString, gb(afterEvict) as NSString, gb(Memory.peakMemory) as NSString))
    print(failures == 0 ? "RESULT: PASS" : "RESULT: FAIL (\(failures) checks)")
}

// MARK: - envelope (V3): the worst case the declaration admits — a score prefix at the 4 096-token budget and 9 000 codec frames
// (the 360 s cap), i.e. the longest NAR prefill (prefix + 9 001 AR tokens) and the longest acoustic chunk, then the VAE. Loaded
// exactly as `YuE2Package.load` loads (bf16 bundle, the tier's quantisation, the same cache limit). phys_footprint is sampled.

func runEnvelope(_ modelDir: URL, _ vaeDir: URL, _ scorePath: URL) throws {
    let sampler = Footprint.Sampler()
    let before = sampler.mark()
    Memory.cacheLimit = Int(cacheGB * Double(1 << 30))
    let t0 = Date()
    let pipe = try YuE2Pipeline.load(modelDir: modelDir, vaeDir: vaeDir, dtype: .bfloat16, quantization: tier.quantization)
    Memory.clearCache()
    Thread.sleep(forTimeInterval: 0.3)
    let afterLoad = sampler.mark()
    print(String(format: "envelope (tier %@): loaded in %.1fs; phys %@ → %@ (resident +%@)", tier.rawValue as NSString,
                 Date().timeIntervalSince(t0), gb(before) as NSString, gb(afterLoad) as NSString, gb(Int(afterLoad) - Int(before)) as NSString))
    let abc = try String(contentsOf: scorePath, encoding: .utf8)
    var ids = [Int]()
    let unit = pipe.tokenizer.encode(abc + "\n")
    while ids.count < 4096 { ids += unit }
    ids = Array(ids.prefix(4096))
    let req = try SongRequest(style: "English, warm piano pop, expressive female voice", lyrics: "[Verse]\nla la la", mode: .full, seed: 1)
    let prefix = try req.prefix(pipe.tokenizer, abcIDs: ids)
    var rng = SystemRandomNumberGenerator()
    let codec = (0 ..< 9000).map { _ in Int.random(in: 0 ..< YuE2Protocol.codecSize, using: &rng) }
    sampler.resetPeak()
    let t1 = Date()
    let lat = try NARSolver.synthesize(model: pipe.model, prefix: prefix, codec: codec, noise: nil, seed: 1, steps: pipe.generation.ode_steps)
    eval(lat)
    let narPeak = sampler.peak
    let t2 = Date()
    let audio = try pipe.decode(lat)
    eval(audio)
    Thread.sleep(forTimeInterval: 0.3)
    let peak = max(narPeak, sampler.peak)
    _ = sampler.stop()
    print(String(format: "  prefix %d ids + %d frames: NAR %.1fs (phys peak %@), VAE %.1fs → %.1fs of audio; phys peak %@ (activation %@ over the resident); MLX peak %@",
                 prefix.count, codec.count, t2.timeIntervalSince(t1), gb(narPeak) as NSString, Date().timeIntervalSince(t2), Double(audio.dim(0)) / 48_000,
                 gb(peak) as NSString, gb(Int(peak) - Int(afterLoad)) as NSString, gb(Memory.peakMemory) as NSString))
}

do {
    switch cmd {
    case "validate":
        guard rest.count == 4 else { print("validate <model> <vae> <request.json> <outdir>"); exit(2) }
        try await runValidate(url(rest[0]), url(rest[1]), url(rest[2]), url(rest[3]))
        exit(failures == 0 ? 0 : 1)
    case "envelope":
        guard rest.count == 3 else { print("envelope <model> <vae> <score.abc>"); exit(2) }
        try runEnvelope(url(rest[0]), url(rest[1]), url(rest[2]))
    case "tokenizer": try tokenizerGate(url(rest[0]), url(rest[1]))
    case "ar": try arGate(url(rest[0]), url(rest[1]))
    case "nar": try narGate(url(rest[0]), url(rest[1]))
    case "vae": try vaeGate(url(rest[0]), url(rest[1]))
    case "e2e": try e2eGate(url(rest[0]), url(rest[1]), url(rest[2]), rest.count > 3 ? rest[3] : nil)
    case "render": try render(url(rest[0]), url(rest[1]), url(rest[2]), url(rest[3]))
    default: print("unknown lane \(cmd)"); exit(2)
    }
} catch {
    print("ERROR: \(error)"); exit(1)
}
