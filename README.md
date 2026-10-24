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

## Status

See `PORTING-SPEC.md` §0 and `MEASUREMENTS.md`. The engine package (`MLXYuE2`) lands on a music capability once
MLXEngine defines one (AB-A-0201).
