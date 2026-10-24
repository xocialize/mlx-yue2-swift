// yue2-gates — parity gates against the fp32 torch oracle (WIP/yue2-port/oracle/dump_goldens.py), PORTING-SPEC §3.
//
//   yue2-gates tokenizer <golden> <model>                 lane T: ids on the goldens' strings, abc round trip
//   yue2-gates ar        <golden> <model>                 lane A: teacher-forced logits (argmax agreement, tail relL2)
//   yue2-gates nar       <golden> <model>                 lanes N1 (velocity at t=1, t=0.5) + N2 (32-step chunk)
//   yue2-gates vae       <golden> <vae>                   lane D: decode of the oracle's chunk latents, full + tiled
//   yue2-gates e2e       <golden> <model> <vae> [out.wav] lane E: oracle tokens + noise → latents → audio
//   yue2-gates render    <model> <vae> <request.json> <out.wav>   a real generation (seeded; not a gate)
//
// Options (before the subcommand): --cpu, --dtype float32|bfloat16, --ar-bits N, --nar-bits N.
import Darwin
import Foundation
import MLX
import MLXNN
import YuE2Core

var args = Array(CommandLine.arguments.dropFirst())
var useCPU = false
var dtype: DType = .float32
var quant = YuE2Quantization.none
while let a = args.first, a.hasPrefix("--") {
    args.removeFirst()
    switch a {
    case "--cpu": useCPU = true
    case "--dtype": dtype = ["bfloat16", "bf16"].contains(args.removeFirst()) ? .bfloat16 : .float32
    case "--ar-bits": quant.arBits = Int(args.removeFirst())
    case "--nar-bits": quant.narBits = Int(args.removeFirst())
    default: fatalError("unknown option \(a)")
    }
}
guard let cmd = args.first else { print("usage: yue2-gates [--cpu] [--dtype d] tokenizer|ar|nar|vae|e2e|render …"); exit(2) }
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
    let tiled = vae.decodeTiled(lat, coreFrames: 64, haloFrames: 16)
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
    let audio = pipe.decode(lat)
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

do {
    switch cmd {
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
