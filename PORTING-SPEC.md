# YuE2-3B: MLX-Swift port spec

Decision **AB-D-0073** (pass of 2026-09-11, reversed by the operator 2026-10-24), ask **AB-A-0200**, task
**AB-T-0350**. Reopens ENHANCEMENTS **E17** as a non-commercial, `named_only` engine model in the
AuK shape (E18, AB-D-0153). This is a full port from M-A-P's PyTorch release. No third-party MLX conversion is used
(`ahmadw/YuE2-3B-MLX`, `audio-cpp/Yue2-3B-GGUF`, the ComfyUI packs): read for structure at most, never copied.

Upstream is pinned at `multimodal-art-projection/YuE@1647252` (2026-10-06, `yue2-infer` 0.1.6, Apache-2.0 code).
The weights are `m-a-p/YuE2-3B@8f24312f` (model.safetensors 7.26 GB bf16, `qwen.tiktoken`, config.json),
`m-a-p/YuE2-Vae@152733a1` (531 MB fp32, the listening decoder) and `m-a-p/YuE2-Vae-legacy@d0585a9c` (the paper-metric
decoder; same architecture). The scratch side (the upstream clone, the venv, goldens, converted bundles) lives in
`mlxengine-audio/WIP/yue2-port/`; everything needed to rebuild it ships in the package's `Tools/`.

## 0. Status

| rung | outcome |
|---|---|
| V0 oracle | fp32 CPU goldens for 2 cases (zh cot=full, en cot=off with CFG); semantic capped at 60 s |
| V1 converter | bit-equal: 628 backbone tensors, 173 decoder tensors (max \|Δ\| 0); MLX conv semantics checked |
| V2 Swift core | fp32 parity on every lane, both cases: tokenizer exact; AR argmax 100 %, logits ≤ 1.8e-6; NAR ≤ 1.6e-5 (velocity) / 4.8e-5 (32 steps); VAE ≤ 4.6e-6; e2e audio ≤ 1.7e-4 |
| V2b tiers | bf16 and 8-bit (AR, NAR, both) pass the pre-registered velocity rule (≤ 1.6e-2); 4-bit AR fails through the cached prefix K/V (0.2 after 32 steps) — see MEASUREMENTS |
| V3 engine | `MLXYuE2` on `Capability.music` (contract 1.56.0): planning full / off, score in and out, cfg, a 360 s cap; CC-BY-NC-4.0 weights (one advisory, never allowlisted), named-only; 16 offline conformance tests; one song + its score re-rendered per tier through `MLXServeEngine`; footprints from phys_footprint incl. the 360 s envelope — see MEASUREMENTS §V3 |
| V4 publish | DONE 2026-10-28 on the operator's decision (AB-D-0186): github.com/xocialize/mlx-yue2-swift `v0.1.0`; `mlx-community/YuE2-3B-bf16` @ 7024750 and `mlx-community/YuE2-Vae-fp32` @ 486147e, sha256-equal to the gated bundles, pinned in `YuE2Configuration`; a fresh download passes `validate` through MLXServeEngine — MEASUREMENTS §V4 |

Package: `mlxengine-audio/PROD/YuE2/mlx-yue2-swift` (`YuE2Core`, `yue2-gates`, `Tools/`). Numbers: `MEASUREMENTS.md`.

## 1. Licence posture (decided by the operator; AB-D-0073 reversal, AUDIO-EVALUATION §4.6)

- Weights (3B, Vae, Vae-legacy): **CC BY-NC 4.0**. Code: Apache-2.0. Oobleck/SnakeBeta code: MIT (Stability AI /
  NVIDIA, `THIRD_PARTY_NOTICES.md`).
- C7: `weightLicense = "CC-BY-NC-4.0"` (an SPDX id; MLXToolKit classifies it `.rejectedWeight` — under `.advisory`
  that is a recorded advisory, under `.permissiveOnly` a refusal by construction; never allowlisted). Port code MIT.
- The offer (operator, 2026-10-24, A-6 applied — the strings live in `Sources/YuE2Core/YuE2Licence.swift`):
  `named_only: true`, `license.accepted_required: true`, **`license.commercial_use: "conditional"`** (AUDIO-EVALUATION
  A-6: allowed under a condition the engine cannot verify), `terms_url` = the HF LICENSE, `notice` = *"YuE2-3B by M-A-P
  (huggingface.co/m-a-p/YuE2-3B), CC BY-NC 4.0 with M-A-P's individual-creator permission (MODEL_LICENSE, 2026-09-16).
  Individuals acting in a personal capacity — personal users, content creators, musicians — may publish, sell and
  license songs made with it. Companies may not use the model commercially without a licence from M-A-P
  (ryuanab@connect.ust.hk); a hosted service making songs for paying users is commercial use. Responsible use is a
  condition of the permission."*, `notes` = *"Covers and supplied lyrics may need the original song's rights; this
  model's licence doesn't grant them."* `policy.license: commercial-use` admits the offer only once the operator has
  accepted the terms (A-6). The surface summaries carry the same sentence.
- **Two licence texts exist upstream (found 2026-10-24).** The HF `LICENSE` on all three weight repos is the plain
  CC BY-NC 4.0 text plus the scope/attribution header. The GitHub `MODEL_LICENSE` (same header) adds an *"Individual
  creator permission and academic use (2026-09-16)"* section: individuals acting in an individual capacity may
  generate, publish, sell and license outputs without a commercial licence; companies may not use the weights
  commercially; redistribution of the weights stays CC BY-NC (attribution to YuE2 + model name + repo). The port
  ships BOTH texts with the bundles (`LICENSE` = HF text, `MODEL_LICENSE` = GitHub text, with provenance). The
  operator chose `commercial_use: "conditional"` (A-6) on 2026-10-24, with the individual-creator condition in the
  notice (AB-A-0200 thread).
- ML[X] Audio Studio stays permissive-only (AB-D-0060): no Studio section. No MARKETING-HIGHLIGHTS entry.
- Publishing: **decided by the operator 2026-10-28 (AB-D-0186)** — "Publish YuE2 V4: the code as
  xocialize/mlx-yue2-swift v0.1.0 and the weights to mlx-community with the non-commercial cards, licence files and
  notice." Each weight repo carries the HF LICENSE, the GitHub MODEL_LICENSE, a NOTICE (attribution + the CC BY-NC 4.0
  §3(a)(1)(B) modification statement) and a card with the non-commercial terms and the individual-creator permission up
  top (`license: cc-by-nc-4.0`, `license_link` the HF LICENSE). The fleet's first publication of non-commercial weights.

## 2. Architecture as read (verified against `config.json` and `src/yue2/*.py` at the pin)

**Backbone** (`modeling_yue2.py`): Qwen3-style decoder, hidden 2048, 28 layers, 16 q / 8 kv heads × head_dim 128,
q/k RMSNorm (eps 1e-6) per head BEFORE RoPE, RoPE θ 1e6 rotate-half (NOT interleaved; cos/sin computed in fp32 and
cast to the activation dtype), SwiGLU 6144, RMSNorm 1e-6 (`x · rsqrt(mean(x²)+eps) · w`, the mean in fp32), ctx 24576,
vocab 184704, untied `lm_head`. Mixture-of-Transformers: each layer has TWO full paths — `input_layernorm` /
`self_attn` / `post_attention_layernorm` / `mlp` (AR) and `nar_input_layernorm` / `nar_self_attn` /
`nar_pre_mlp_layernorm` / `nar_mlp` (NAR) — routed per position by `ar_mask`; attention is shared over the
concatenated K/V. 3.58 B params.

**AR generation** (`sampling.generate_tokens`): prefill the prefix with the AR path (causal), then one token at a
time. Two phases, each with its own request-local RNG re-seeded from the request seed:
- abc: temperature 0.7, top-p 0.9, top-k 30, repetition 1.005 over a 100-token window, min 32, max 4096; logits
  masked to `[0, EOD)` ∪ {`</abc>`}.
- semantic: 1.0 / 0.95 / 100 / 1.2 over 50, min 200, max 9000; masked to the codec range `[151853, 151853+32768)`
  ∪ {MUSIC_END}.
- Order: mask → `-inf` on the end token while `step < min_tokens` → windowed frequency penalty (`alpha =
  penalty^freq`; `logits<0 ? logits·alpha : logits/alpha`) → temperature → top-k threshold → top-p on the sorted
  softmax with `cumsum − p > top_p` removed, keeping the first 1 (or 3 in `legacy_off`) → softmax → multinomial.
- CFG only when `cfg_scale ≠ 1`: `uncond + s·(cond − uncond)` in the model dtype; the negative prefix is
  `[EOD] + instruction` (+ `<abc>` + the SAME abc ids + `</abc>`) + MUSIC_START. Defaults: 1.0 for full/melody
  (single branch), 1.01 for off (two branches; `legacy_off` keeps scores in bf16 and top-p keeps 3).
- AB-L-0090 applies: a seeded rollout is never a gate. Gates are teacher-forced.

**Token protocol** (`protocol.py`): `[EOD] + enc("{instruction}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n") +
<abc> + abc_ids + </abc> + MUSIC_START`; `cot=off` → `<abc></abc>MUSIC_START` with no ids. EOD 151643, `<abc>`
151847, `</abc>` 151848, MUSIC_START/END 151851/2, LATENT_START/END/PAD 184621–3 (unused at inference).
Chunking: `size = (24576 − len(prefix) − 3) // 2` frames; a chunk's AR tokens are `prefix + (codec + 151853) +
[MUSIC_END]`.

**Tokenizer** (`tokenization_yue2.py`): tiktoken over `qwen.tiktoken` (151 643 ranks), the Qwen pattern, NFC,
`encode_ordinary` (specials never encoded from text; `<abc>`/`</abc>` sit at specials[204:206]). Donor: the
IndexTTS-2.5 Swift byte-level BPE (its PORTING-SPEC D1) — gate: id-exact on the instruction/tag/lyrics strings of
the goldens and on the oracle's ABC text (round trip `decode(abc_ids) == abc`, `encode(abc) == abc_ids`).

**NAR / flow matching** (`nar.py` `CachedNAR`): per chunk, prefill the AR tokens ONCE with the AR weights (causal,
positions 0…A−1) and keep each layer's K/V (post q/k-norm, post-RoPE). Noise for the WHOLE song is one
`torch.randn((frames, 64))` fp32 on the CPU from `manual_seed(seed)`, sliced per chunk — the Swift port loads the
reference noise for every gate. Each velocity eval:
- `x_nar = pad(state, 1 frame of zeros each side)` → `vae2llm` (64→2048) + `time_embedder(shifted t)` (same for
  every NAR position) + `latent_pos_embed[0…N+1]` (sinusoidal table 24576 × 2048, sin at even, cos at odd columns);
- RoPE positions `A … A+N+1` (continuing the prefix); per layer `nar_input_layernorm → nar_self_attn.project_qkv`,
  K/V = `[cached AR ; NAR]`, UNMASKED attention (NAR→AR full, NAR↔NAR bidirectional), `o_proj`, then
  `nar_mlp(nar_pre_mlp_layernorm(x))`;
- `llm2vae(model.norm(x))[1:−1]` → velocity `[N, 64]`.
- Timestep: `t = 1 − step/32`, `raw = clamp(logit(t), −20, 20)` in float64, `shifted = sigmoid(raw)` in the model
  dtype (`timestep_shift` 1.0 → identity). `TimestepEmbedder`: `freqs = exp(−ln 10000 · i/128)`, `emb = [cos ; sin]`
  (cos FIRST), Linear 256→2048 → SiLU → Linear.
- Midpoint, 32 steps, dt 1/32: `mid = x − v(x, t)·dt/2`, `x = x − v(mid, t − dt/2)·dt`. The state is kept in the
  model dtype upstream (bf16 on CUDA); the fp32 oracle keeps it fp32.
- 64 NAR forwards per chunk; no custom kernels (SDPA, silu, where).

**VAE** (`modeling_vae.py`, decoder only): Oobleck, channels 64, c_mults [1,2,4,8,16,32], strides [2,2,4,4,5,6]
(1920× → 25 latent frames/s), 48 kHz stereo, fp32 only. Layers: `WNConv1d(64→2048, k7, p3)`; six `DecoderBlock`s
in channel order 2048→1024→512→256→128→64 with strides **6,5,4,4,2,2** (`strides[i−1]` for i = 6…1): `SnakeBeta(in)
→ WNConvTranspose1d(k=2s, stride s, padding ceil(s/2)) → ResidualUnit ×3 (dilation 1,3,9: SnakeBeta → WNConv1d k7
dil d pad 3d → SnakeBeta → WNConv1d k1, + residual)`; then `SnakeBeta(64) → WNConv1d(64→2, k7, p3, no bias)`;
`final_tanh: false`. SnakeBeta is log-scale: `x + sin²(x·eᵅ) / (eᵝ + 1e−9)`. weight_norm `(g, v)` folds at
conversion. Natural output length `1920·T − 64`. Pipeline decodes tiled (core 1024 / halo 16 frames, receptive-field
derived, exact: cores are cropped, never blended) and clamps to ±1; the output is FLAC PCM_24 or WAV float.

**Not ported:** the VAE encoder (latent 128 = μ + scale, `softplus(scale)+1e−4`; only an audio-conditioned workflow
would need it), `cuda_graph.py`, `fast.py` (vLLM), `quantization.py` (fp8 AR), `progress.py`, `storage.py`.

## 3. Rungs and gates

Goldens are fp32 on the CPU (upstream disables TF32 and sets matmul precision "highest" — the oracle inherits that)
with injected noise, from two fixed requests: `tonight-awake` (the released example, `cot=full`) cut to a short
lyric, and a `cot=off` request. Per golden: the request, the exact prefix ids, the oracle's abc text + ids, the
oracle's semantic tokens (a seeded MPS/bf16 rollout — reference only, never a gate), the song noise, and:

| gate | lane | tolerance (fp32, `MLX_ENABLE_TF32=0`, AB-D-0114) |
|---|---|---|
| T | tokenizer | id-exact on every golden string; abc round trip exact |
| A | AR teacher-forced logits over `prefix + abc_ids + </abc> + MUSIC_START + 64 codec tokens` | logits relL2 ≤ 1e-3 per position; argmax agreement 100 % |
| N1 | one NAR velocity on the oracle's state at `t = 1` and at `t = 0.5` | relL2 ≤ 1e-3 |
| N2 | 32-step midpoint on the oracle's semantic tokens + noise → latents | relL2 ≤ 5e-3 (the ODE compounds 64 evals) |
| D | VAE decode of the oracle's latents, tiled and full | waveform relL2 ≤ 1e-3; tiled == full within 1e-5 |
| E | end-to-end teacher-forced: oracle tokens + noise → Swift latents → Swift VAE | waveform relL2 ≤ 1e-2; listening pass |

- **V0 oracle** — `Tools/oracle-capture/dump_goldens.py`: writes the per-stage goldens above as `.npy` + JSON.
- **V1 converter** — `Tools/convert/convert.py`: HF safetensors → MLX bundles. Backbone: names kept, bf16 as shipped
  (fp32 copy for the gate); VAE decoder: weight_norm folded, `Conv1d` weights transposed to MLX `[out, k, in]`,
  `ConvTranspose1d` to `[out, k, in]` with the kernel flipped as MLX's `ConvTransposed1d` needs, encoder dropped.
  Gate: converted tensors equal upstream's own folded tensors (max |Δ| 0 at fp32).
- **V2 Swift core** (`YuE2Core`): `YuE2Tokenizer`, `YuE2Backbone` (AR + NAR paths), `ARSampler`, `NARSolver`,
  `OobleckDecoder`, `YuE2Pipeline`; gates T/A/N1/N2/D/E through `yue2-gates`.
- **V2b tiers** — bf16 whole; 8-bit AR + bf16 NAR; 8-bit both; 4-bit AR + 8-bit NAR. The NAR path is the named
  quant risk (E17: fully 4-bit flow matching drifted to 12 % velocity error in the third-party port). Pre-registered
  rule per tier against the fp32 E lane: N1 relL2 ≤ 2e-2 (bf16) / 5e-2 (8-bit) / 1e-1 (4-bit AR only), plus a
  listening pass and an ASR check of the lyrics (mlx-whisper, WER vs the lyric sheet within 10 points of bf16).
- **V3 engine** (`MLXYuE2`): on `Capability.music` (contract 1.56.0, AB-A-0201 answered 2026-10-24; latents are not on
  `MusicResponse`). Declares only what V2 gated: `.planning` with modes `full` (zh golden) and `off` (en golden, CFG
  1.01) — `melody` has no golden and is refused —, `.score` (the A lane's prefix with supplied abc ids), `.guidanceScale`,
  `maxDurationSeconds` 360 (the 9 000-token budget). Licence per §1; footprints from `phys_footprint` (AB-L-0113)
  through MLXServeEngine plus an `envelope` lane at the 360 s / 4 096-score-token worst case; C0–C13 conformance.
- **V4 publish** — xocialize/mlx-yue2-swift; weights per §1's publishing decision.

## 4. Budget (estimate from E17; measured numbers go to MEASUREMENTS.md)

7.26 GB bf16 backbone + 0.53 GB fp32 decoder + KV 114.7 KB/token → ≈ 8.5 GB resident at bf16; ≈ 2–3 min per 3-min
song on an M5 Max (abc 10–60 s, semantic ~65 s at 70 tok/s, NAR ~45 s, VAE seconds).
