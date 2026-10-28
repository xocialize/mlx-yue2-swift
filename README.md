# mlx-yue2-swift

[M-A-P's YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) — lyrics and a style prompt → a complete song with vocals,
48 kHz stereo, through an editable ABC score — ported to Swift-MLX for Apple Silicon. A full port from the released
PyTorch code (`multimodal-art-projection/YuE@1647252`, `src/yue2`), gated against an fp32 oracle; no third-party MLX
conversion is used.

**Licence.** This code is MIT. The weights (YuE2-3B, YuE2-Vae, YuE2-Vae-legacy) are **CC BY-NC 4.0**, and the GitHub
`MODEL_LICENSE` adds an individual-creator permission (personal users, creators and musicians acting in an individual
capacity may publish, sell and license their songs; companies may not use the weights commercially) that the Hugging
Face `LICENSE` files lack — the converter ships both texts with every bundle. As an engine offer it is named-only,
terms-accepted-before-first-use, `commercial_use: "conditional"` (`YuE2Licence.swift` holds the exact strings). Attribution per the weights' licence: YuE2, the model name, and
https://huggingface.co/m-a-p/YuE2-3B.

## What it does

- **Plan** (`cot = full | melody | off`): a chord-annotated or melody-only ABC score, or none. Supply your own score
  (an edited plan, or a cover transcribed by SheetSage2) with `melody` / `full`.
- **Generate**: semantic codec tokens (25 per second, up to 9000 = 6 min), optional CFG.
- **Synthesize**: acoustic flow matching — the AR prefix prefilled once per chunk, 32-step midpoint over the NAR path.
- **Decode**: the Oobleck VAE (fp32), exact tiled decode, 48 kHz stereo.

```swift
import YuE2Core
let pipe = try YuE2Pipeline.load(modelDir: model, vaeDir: vae, dtype: .bfloat16)          // or quantization: .init(arBits: 8, narBits: nil)
let req = try SongRequest(style: "City Pop, upbeat, groovy bass, synth", lyrics: "[Verse]\n…\n[Chorus]\n…", mode: .full, seed: 12300)
let song = try pipe.generate(req) { progress in /* .plan / .semantic / .synthesize / .decode */ }
try WAV.write16(song.audio, channels: song.channels, sampleRate: song.sampleRate, to: out)   // song.plan.abc is the score
```

## Bundles

`Tools/convert/convert.py` writes `YuE2-3B-{bf16,fp32}/` (`model.safetensors`, `config.json`, `qwen.tiktoken`,
licences) and `YuE2-Vae-fp32/`, `YuE2-Vae-legacy-fp32/` (`decoder.safetensors`, `config.json`, licences). Quantized
tiers are applied at load (`YuE2Quantization`): the AR and NAR paths can take different bit widths, because the NAR
(flow-matching) path is the quantization-sensitive half.

## Parity

`yue2-gates` runs the lanes of `PORTING-SPEC.md` §3 against the fp32 torch oracle (`Tools/oracle-capture`): tokenizer
(id-exact), AR teacher-forced logits, NAR velocity and 32-step solve, VAE decode (full and tiled), end to end with the
oracle's tokens and noise. Numbers: `MEASUREMENTS.md`.

Build with the Xcode build system so MLX's metallib ships: `swift build -c release --build-system swiftbuild`.
The exact fp32 lane needs `MLX_ENABLE_TF32=0`.

## Engine package (`MLXYuE2`)

`YuE2Package` serves `Capability.music` (mlx-engine-swift contract 1.56.0, `MusicRequest` → `MusicResponse`) as the surface
`yue2-3b`. It declares exactly what the V2 gates covered — `MusicControls(features: [.planning, .score, .guidanceScale],
planningModes: [.full, .off], maxDurationSeconds: 360)` — and the engine refuses everything else before admission
(`melody`, reference or source audio, anchors, a vocal language, a length past 360 s). `durationSeconds` caps the codec tokens
(25 per second); the song ends on its own when the lyrics do, and `truncated` says when a cap cut it. A `full` run returns
its ABC score on `MusicResponse.score`; supply an edited one back through `MusicRequest.score`.

```swift
import MLXServeCore, MLXToolKit, MLXYuE2
let engine = MLXServeEngine()   // CC-BY-NC-4.0 is outside .permissiveOnly: registered with one licence advisory
let id = try await engine.register(YuE2Package.registration,
                                   configuration: YuE2Configuration(tier: .bf16, modelsRootDirectory: modelsRoot),
                                   id: "yue2-3b")
try await engine.prepare(.music, package: id)
let song = try await engine.run(MusicRequest(caption: "warm piano pop, female vocal, 88 BPM",
                                             lyrics: "[Verse]\n…\n[Chorus]\n…", seed: 831001), package: id) as! MusicResponse
// song.audio: 48 kHz stereo .wav; song.score: the ABC plan; song.truncated
```

Tiers (`YuE2Tier`, quantised at load from the bf16 bundle): `bf16`, `ar8` (8-bit AR path), `int8` (8-bit AR and NAR);
4-bit AR is not a tier (V2b). The licence fields: `weightLicense` CC-BY-NC-4.0 (never allowlisted — ML[X] Audio Studio
never loads it), MIT port code, the operator's notice as a `.required` `LicenseNotice` and the rights note as
`.informational`; `YuE2Package.namedOnly` / `.acceptedRequired` / `.commercialUse` carry the offer fields a media-engine
adapter reads. The host never makes it the capability's default. Footprints (phys_footprint): `MEASUREMENTS.md` §V3.

`yue2-gates validate <model> <vae> <request.json> <outdir>` runs the consumer path (register → prepare → a song → its own
score re-rendered → refusals → evict); `yue2-gates envelope <model> <vae> <score.abc>` measures the 360 s worst case.

## Install

```swift
.package(url: "https://github.com/xocialize/mlx-yue2-swift", from: "0.1.0")   // products: MLXYuE2 (engine), YuE2Core
```

Weights (CC BY-NC 4.0, both licence texts and a NOTICE in each repo):
[mlx-community/YuE2-3B-bf16](https://huggingface.co/mlx-community/YuE2-3B-bf16) (7.26 GB) and
[mlx-community/YuE2-Vae-fp32](https://huggingface.co/mlx-community/YuE2-Vae-fp32) (265 MB). `YuE2Configuration` pins the
published commits, so the engine materializes exactly the hash-verified bytes; with `modelsRootDirectory` set and no explicit
folders, `prepare` resolves them from the store.

## Status

See `PORTING-SPEC.md` §0 and `MEASUREMENTS.md`. V0–V4 done: v0.1.0 and the two weight repos were published on the operator's
decision of 2026-10-28 (AB-D-0186).
