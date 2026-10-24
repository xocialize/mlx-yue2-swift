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

## Not measured

- A 3-min song (the model card's 71 s / 3.6-min song on an RTX 4090 at 139 tok/s is the comparison point; at 108 tok/s
  and 11 s per 2100 NAR frames, ≈ 65 s + 25 s + 20 s ≈ 2 min is the arithmetic, unmeasured).
- `cot=off` and `melody` renders, supplied-ABC re-renders, CFG 1.2.
- phys_footprint (needs the engine package, V3).
