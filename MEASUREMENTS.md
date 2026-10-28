# YuE2-3B port — measurements

Hardware: Mac Studio Mac17,14, Apple M5 Max, 128 GB; macOS 27.2, Xcode 27.1, mlx-swift 0.32.3 (Python mlx 0.32.4),
torch 2.10.0 (CPU oracle). Release builds (`--build-system swiftbuild`). GPU lanes with `MLX_ENABLE_TF32=0` unless
noted. Goldens: `oracle/goldens/{tonight-awake-short (zh, cot=full, seed 12300), stay-awhile-off (en, cot=off, CFG
1.01 two-branch, seed 831001)}`, semantic capped at 1500 tokens = 60 s. Raw logs: `receipts/`, `oracle/dump_goldens.log`.

## V0 — fp32 CPU oracle (torch, 14 threads)

| case | abc | semantic | full NAR (1500 fr) | VAE (60 s) | N1/N2/D chunk |
|---|---|---|---|---|---|
| tonight-awake-short | 983 tok / 43 s (23 tok/s) | 1500 tok / 71 s (21 tok/s) | 215 s | 5 s | 46 s |
| stay-awhile-off | — | 1500 tok / 126 s (two CFG branches) | 198 s | 5 s | 50 s |

Upstream's tiled decode (core 1024 / halo 16) equals its full decode bit for bit on the CPU (max|Δ| 0).

## V1 — converter gate

628 backbone tensors bit-equal (bf16 kept; fp32 bundle is the exact cast). 173 decoder tensors: max|Δ| 0 against
upstream's `remove_weight_norm`. MLX conv / conv_transpose on the converted (O, K, I) layouts vs torch: conv_pre
7.4e-7, ConvTranspose(k12, s6, p3) 1.2e-6, conv_post 5.5e-7 relmax — with TF32 on, the transposed conv reads 8.4e-4
(AB-D-0114). Bundle sizes: YuE2-3B-bf16 6.8 GB, -fp32 14 GB, each VAE 253 MB.

## V2 — Swift core, fp32 GPU, TF32 off (the exact lane)

| lane | tonight-awake-short | stay-awhile-off | rule |
|---|---|---|---|
| T tokenizer | 5/5 strings exact; abc round trip exact; prefix (1170) exact | 4/4 exact; prefix exact | id-exact |
| A AR teacher-forced | argmax 1234/1234; tail logits relL2 1.8e-6 | pos 173/173 relL2 6.7e-7; neg 76/76 relL2 7.1e-7 | 100 %, ≤ 1e-3 |
| N1 velocity t=1 / t=0.5 | 1.6e-5 / 1.1e-5 | 2.0e-6 / 2.3e-6 | ≤ 1e-3 |
| N2 32-step chunk (256 fr) | 4.8e-5 | 3.0e-6 | ≤ 5e-3 |
| D VAE full / tiled(64) | 4.6e-6 / 3.0e-6; tiled-vs-full 2.7e-6 | 8.9e-7 / 8.6e-7; 4.4e-7 | ≤ 1e-3; tiled≈full |
| E latents / audio | 7.3e-5 / 1.7e-4 | 9.2e-5 / 1.5e-4 | ≤ 5e-3 / ≤ 1e-2 |

Wall (fp32 GPU): load 0.5 s (lazy), AR forward of 1234 tokens 0.4 s, NAR prefill of 1427 tokens 0.3 s, 32-step
solve of 256 frames 5.3 s, full NAR of 1500 frames (prefix 1170) 29.9 s, VAE of 60 s audio 0.9 s.

## V2b — tiers (bf16 bundle; runtime affine quantization, group 64), against the fp32 oracle

Pre-registered rule (PORTING-SPEC §3): N1 velocity relL2 ≤ 2e-2 (bf16) / 5e-2 (8-bit) / 1e-1 (4-bit AR only), plus
a listening pass and a lyric-ASR check within 10 points of bf16. The gate CLI prints the fp32 lane's PASS/FAIL labels
on every run; for tiers the rule above is what counts.

| tier | case | N1 t=1 / t=0.5 | N2 chunk | E latents | E audio | NAR 1500 fr |
|---|---|---|---|---|---|---|
| bf16 | tonight | 1.29e-2 / 1.29e-2 | 1.50e-2 | 2.01e-2 | 4.35e-2 | 7.6 s |
| bf16 | stay | 1.22e-2 / 1.18e-2 | 1.11e-2 | 1.90e-2 | 3.09e-2 | 6.7 s |
| 8-bit AR + bf16 NAR | tonight | 1.42e-2 / 1.29e-2 | 1.57e-2 | 3.39e-2 | 6.37e-2 | 7.3 s |
| 8-bit AR + bf16 NAR | stay | 1.23e-2 / 1.20e-2 | 1.22e-2 | 2.39e-2 | 3.01e-2 | 6.7 s |
| 8-bit AR + 8-bit NAR | tonight | 1.58e-2 / 1.58e-2 | 2.25e-2 | 4.01e-2 | 7.07e-2 | 8.0 s |
| 8-bit AR + 8-bit NAR | stay | 1.41e-2 / 1.47e-2 | 2.03e-2 | 1.78e-2 | 2.26e-2 | 7.4 s |
| 4-bit AR + 8-bit NAR | tonight | 7.95e-2 / 4.41e-2 | **2.00e-1** (cos 0.980) | 1.07e-1 | 2.21e-1 | 8.0 s |
| 4-bit AR + 8-bit NAR | stay | 3.68e-2 / 3.54e-2 | 8.48e-2 | 1.53e-1 | 3.07e-1 | 7.4 s |

Reading: bf16 sits at ~1.2e-2 velocity drift from fp32 (upstream itself runs bf16 on CUDA, so this is where the
release lives); 8-bit on either path adds little (≤ 1.6e-2). **4-bit AR breaks the NAR through the cached prefix
K/V** (the NAR attends to K/V computed by the 4-bit AR weights): 8e-2 on the first velocity and 0.2 after 32 steps —
the quant-sensitive half is the AR→NAR context, not only the NAR weights (E17's third-party note saw 12 % at fully
4-bit). 4-bit AR is not a shippable tier for the NAR stage; a 4-bit AR for the TOKEN stages with a bf16/8-bit AR
prefill for the NAR prefix would need two AR copies (not done).

### Lyric ASR (mlx-whisper large-v3-turbo, temperature 0; `receipts/asr-{en,zh}.txt`)

| file | en WER (stay-awhile-off, 60 s) | zh CER (tonight-awake-short, 60 s) |
|---|---|---|
| oracle `audio_ref.flac` (torch fp32) | 3.9 % | 93.3 % |
| Swift fp32 e2e | 3.9 % | 93.3 % |
| bf16 | 3.9 % | 93.3 % |
| 8-bit AR + bf16 NAR | 3.9 % | 93.3 % |
| 8-bit AR + 8-bit NAR | 3.9 % | 91.7 % |
| 4-bit AR + 8-bit NAR | 3.9 % | 91.7 % |

English: the transcript is identical on every tier and on the oracle (one word off in 52), so tier drift is 0 points
and the within-10-points rule is met. Mandarin: Whisper hallucinates a "作词李宗盛作曲李宗盛…" credits loop over the
sung mix on EVERY file, the oracle included (it then picks up real lines: 路灯眨着眼睛 偷看谁的身影 …), so the zh
number is an ASR failure, not a model signal — drift between tiers is still ≤ 1.6 points. A lyric check for sung
Mandarin over a full mix needs a music-aware ASR; the listening pass is the operator's (renders sent 2026-10-24).

## Real renders (bf16 bundle; `render` lane; shortened 今晚不眠, cot=full, seed 12300 — a seeded rollout, so the two
tiers make different songs)

| tier | abc | semantic | NAR | VAE | e2e | song | MLX peak / active |
|---|---|---|---|---|---|---|---|
| bf16 | 1114 ids / 11.2 s | 2098 tok / 19.4 s = **108 tok/s** | 11.1 s | 1.3 s | **43.1 s** | 83.9 s | 13.25 / 7.53 GB |
| 8-bit AR + bf16 NAR | 1082 ids / 7.4 s | 2094 tok / 14.6 s = **143 tok/s** | 11.0 s | 1.2 s | **34.3 s** | 83.8 s | 11.93 / 6.21 GB |

So an 84 s song renders in 43 s at bf16 — about 0.5× real time — against E17's estimate of 2–3 min per 3-min song
(≈ 0.8×). The AR decode is memory-bound (7.3 GB of bf16 weights per token → 108 tok/s; 8-bit AR → 143 tok/s); the NAR
stage is 11 s for 2100 frames (64 velocity evals). Memory is MLX-accounted (`Memory.peakMemory`); the declared
footprint comes from `phys_footprint` through MLXServeEngine at V3 (AB-L-0113).

## V3 — engine package `MLXYuE2` through MLXServeEngine (2026-10-28)

mlx-engine-swift 0.72.0 (contract 1.57.0; the music capability is 1.56.0), default engine policy (`.permissiveOnly`,
`.advisory`), MLX cache 1 GiB, bundles `weights/YuE2-3B-bf16` + `weights/YuE2-Vae-fp32` given explicitly (nothing is
published; the MAT path is covered offline). Logs: `receipts/v3-validate-{bf16,ar8,int8,bf16-off}.log`,
`receipts/v3-envelope-{bf16,ar8,int8}.log`, `receipts/v3-lane-d-recheck.txt`; audio + scores in `v3/out*/`. Driver
`v3/run_v3.sh` (one model process at a time).

**Offline conformance (17 tests, `MLXYuE2Tests`).** C0 contract ≥ 1.56.0; C1 `music` only; C7 `CC-BY-NC-4.0` weights —
`.rejectedWeight` under `.permissiveOnly` AND `.permissiveOrAcknowledged` (on neither allowlist), refused under `.blocking`,
one weight advisory under the default; the operator's notice (`YuE2Licence.notice`, AB-D-0161) as the one `.required`
`LicenseNotice` with the HF terms URL, the rights note and the named-only / accepted-before-first-use / conditional
commercial-use line `.informational`; C8 MIT; C9 Codable without environment URLs; C10 split footprints; C11 the surface
advertises exactly caption / lyrics / durationSeconds / seed / guidanceScale / planning / score, `declarationProblems`
empty; C12 `@unknown default` on the planning switch; C13 + CAN-1..3 (pre-cancelled run → `CancellationError`;
checkpoints per score token, codec token, ODE step and VAE tile — the VAE tile checkpoint is new in the Core at V3); MAT
per tier (two roles: `model`, `vae`); the request plane (14 refusals: melody, edit, reference, anchors, language, > 360 s,
0 s, empty caption / lyrics / score, score + off, cfg 21 / −1 / NaN).

**Declared** (`MusicControls`): `features [.planning, .score, .guidanceScale]`, `planningModes [.full, .off]`,
`maxDurationSeconds 360`. `.melody` is NOT declared — no golden covers it. `durationSeconds` caps the codec tokens (× 25 Hz,
≤ 9 000).

**Live, one song per tier** (upstream's `examples/song.json` "city_lights", English piano pop, cot=full, seed 831001;
then the song's own planned score supplied back as `MusicRequest.score`, then nine requests that must be refused, then
evict):

| tier | phys after prepare | song | re-render of its score | peak phys (songs) | after evict | result |
|---|---|---|---|---|---|---|
| bf16 | 7.60 GB | 57.68 s, 476 score + 1442 codec tokens, rms −19.7 dBFS, **40.7 s wall (RTF 0.71)** | byte-identical WAV + score, 32.9 s | 14.39 GB | 0.30 GB | PASS |
| ar8 (8-bit AR) | 6.40 GB | 58.32 s, rms −19.7 dBFS, 29.9 s wall † | byte-identical, 24.4 s † | 12.36 GB | 0.30 GB | PASS |
| int8 (8-bit AR + NAR) | 5.08 GB | 58.32 s, rms −19.7 dBFS, 30.2 s wall † | byte-identical, 25.3 s † | 11.32 GB | 0.30 GB | PASS |
| bf16, cot=off (en golden request, CFG 1.01) | 7.60 GB | 69.44 s, 1736 codec tokens, rms −17.2 dBFS, no score returned, 56.5 s wall † | — (no score) | 14.14 GB | 0.30 GB | PASS |

Every run: one licence advisory (`CC-BY-NC-4.0 … outside permissiveOnly`), `needsDownload` false, 48 kHz × 2 ch,
`truncated` false (the song ended on MUSIC_END), the phases plan → generate → denoise (32) → decode → postprocess, and all
nine refusals (melody, cover, referenceAudio, vocalLanguage, anchors, 400 s, empty lyrics, score + off, cfg 25) thrown
as `PackageError.unsupportedRequestFeature` — the first six by the engine before admission. The re-render being
byte-identical to the planned song on every tier is the score path's receipt: a supplied score reproduces the plan's
prefix exactly, so an edited score re-renders deterministically from the same seed.

**Envelope** (`envelope` lane, the same load as `YuE2Package.load`): a 4 096-id score (the abc budget) + text = 4 144-id
prefix, 9 000 random codec frames (the 360 s cap) → one NAR chunk (AR prefill of 13 145 tokens + 32 midpoint steps over
9 002 positions) → VAE (9 tiles) → 360.0 s of audio.

| tier | resident | NAR phys peak | overall phys peak | activation over resident | NAR / VAE wall † |
|---|---|---|---|---|---|
| bf16 | 7.58 GB | 11.18 GB | **14.58 GB** | 7.00 GB | 102.6 / 8.6 s |
| ar8 | 6.38 GB | 9.67 GB | **13.27 GB** | 6.89 GB | 101.4 / 8.9 s |
| int8 | 5.06 GB | 8.36 GB | **11.92 GB** | 6.86 GB | 124.8 / 9.3 s |

The peak is the VAE (the NAR stage stays ~3.5 GB above resident); the activation is ~7 GB on every tier because it is
activations, not weights. **Declared** (`QuantFootprint` / `FootprintConfigured`): bf16 8.0 + 7.5 GB, int8 5.5 + 7.5 GB,
ar8 resident hint 7.0 GB on the shared `.int8` activation — each ≥ its envelope peak. By the engine's 0.7 × RAM rule
every tier fits a 24 GB Mac and none a 16 GB Mac.

**Lane D re-check** after the VAE tile checkpoint: audio relL2 4.6e-6 / 8.9e-7 (full), 3.0e-6 / 8.6e-7 (tiled), tiled ==
full 2.7e-6 / 4.4e-7 — unchanged from V2.

† **Speed caveat.** From 09:06:40 the E27 HeartMuLa V0 render (torch-MPS, ~22 GB phys, another process) shared the GPU.
The bf16 city_lights song (≈ 09:05:30–09:06:15) ran before it; its re-render overlapped for its last few seconds; every
other wall time above overlapped it and is inflated. phys_footprint is per-process, so the memory figures stand. No speed
is declared on the manifest.

**What V4 would publish (NOT done — the operator's decision, PORTING-SPEC §1):**
- Code: `xocialize/mlx-yue2-swift` (MIT; `YuE2Core`, `MLXYuE2`, `yue2-gates`, `Tools/`), first tag v0.1.0.
- Weights, if the CC BY-NC case is cleared for mlx-community: `mlx-community/YuE2-3B-bf16` (the converter's bundle,
  7.26 GB — `model.safetensors`, `config.json` with the `mlx_yue2` section, `qwen.tiktoken`) and
  `mlx-community/YuE2-Vae-fp32` (265 MB `decoder.safetensors`, weight_norm folded, encoder dropped) — the repo names
  `YuE2Configuration` already declares — each with the HF `LICENSE`, the GitHub `MODEL_LICENSE`, `NOTICE` (attribution:
  YuE2, the model name, the repo; a modification statement per CC BY-NC 4.0 §3(a)(1)(B)), `THIRD_PARTY_NOTICES.md` and a
  card stating non-commercial use and the individual-creator permission. Optional: `YuE2-Vae-legacy-fp32` (the paper
  decoder; the package does not load it) and pre-quantised 8-bit bundles (the tiers quantise at load today).
- Then: pin the revisions in `YuE2Configuration`, a fresh-download render through the engine, and the registry row
  (mlx-engine's `model-registry.md`, asked of mlx-engine).

## Not measured

- A sampled 3-min song (the engine songs ended on their own at 58–69 s; the 360 s envelope is teacher-forced).
- `melody` renders (not declared), CFG 1.2.
- Speed for the 8-bit tiers and the cot=off song without another GPU process (see † above).
- The 360 s envelope through the AR sampler itself (the envelope lane prefills the 13 k tokens in one pass; token-by-token
  decoding to 9 000 holds less at once).
