# Tools

The Python and shell side of the port. Nothing here assumes a directory layout: every location is an argument.

| path | what |
|---|---|
| `oracle-capture/dump_goldens.py` | **V0 oracle.** Runs upstream `yue2` (multimodal-art-projection/YuE@1647252) with the backbone in fp32 on the CPU and dumps per-case goldens: tokenizer strings/ids, exact prefixes, a seeded rollout (reference only), the song noise, teacher-forced AR logits, NAR velocities at two t, a 32-step chunk solve, full-song latents, VAE decodes. `--model <YuE2-3B snapshot> --vae <YuE2-Vae snapshot> --out <goldens> [--cases a,b] [--semantic-cap N]` |
| `convert/convert.py` | **V1 converter.** `m-a-p/YuE2-3B` + `YuE2-Vae(-legacy)` → MLX-layout bundles: backbone keys verbatim (bf16, or fp32 for the gate); VAE decoder with weight-norm folded and convs in (O, K, I); encoder dropped; both licence texts + a NOTICE in every bundle. `--model <snap> --vae <snap> [--vae-legacy <snap>] --github <YuE clone> --out <dir> [--dtype float32\|bfloat16]` |
| `convert/gate_v1.py` | **V1 gate.** Converted tensors equal upstream's own folded weights (max \|Δ\| 0), the bf16 backbone is bit-equal to the checkpoint, and MLX's conv / conv_transpose reproduce torch on the converted layouts. Run with `MLX_ENABLE_TF32=0`. |
| `asr/score_lyrics.py` | **V2b lyric check.** mlx-whisper large-v3-turbo WER (en) / CER (zh) against the lyric sheet; the number that matters is the drift between tiers. |
| `run_v2b.sh` | **V2b matrix.** `nar` + `e2e` lanes per tier (bf16; 8-bit AR + bf16 NAR; 8-bit both; 4-bit AR + 8-bit NAR) on both goldens, then two real renders. |

Swift-side lanes are `yue2-gates` subcommands: `tokenizer`, `ar`, `nar`, `vae`, `e2e`, `render` (header of
`Sources/yue2-gates/main.swift`).

**Environment.** The oracle was validated with upstream's pins: torch 2.10.0, transformers 4.57.6, safetensors 0.7.0,
tiktoken 0.12.0, numpy 2.2.6, soundfile 0.13.1, accelerate 1.13.0, plus `pip install -e <YuE clone> --no-deps`; the V1
gate adds `mlx`. Run the venv's Python with `-I`.

Build the gate CLI with the Xcode build system so the MLX metallib ships (AB-B-0007):
`swift build -c release --build-system swiftbuild --product yue2-gates`. The exact fp32 lane needs `MLX_ENABLE_TF32=0`
(the M5 default TF32 costs ~1e-3 relL2 per GEMM, AB-D-0114).
