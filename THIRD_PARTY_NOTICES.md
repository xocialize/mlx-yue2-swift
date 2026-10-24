# Third-party notices

This package's Swift code is MIT (`LICENSE`). It is a port of, and reads its weights from, the following:

- **YuE2** — multimodal-art-projection/YuE (`src/yue2`, pinned at `1647252`), Apache-2.0 code. The port reproduces
  its architecture and protocol; no upstream code is vendored.
- **YuE2-3B, YuE2-Vae, YuE2-Vae-legacy weights** — m-a-p, **CC BY-NC 4.0** (non-commercial; attribution to YuE2,
  the model name and its repository). The upstream GitHub `MODEL_LICENSE` adds an individual-creator permission
  (2026-09-16) that the Hugging Face `LICENSE` files lack; the converter ships both texts with every bundle. These
  weights are NOT part of this package and are not relicensed by it.
- **Oobleck VAE / SnakeBeta** — the decoder design follows stable-audio-tools (Stability AI, MIT) and BigVGAN's
  SnakeBeta (NVIDIA, MIT), as upstream's `THIRD_PARTY_NOTICES.md` records; licence texts ride in each VAE bundle's
  `licenses/`.
- **tiktoken** — the byte-level BPE algorithm (OpenAI, MIT) is reimplemented in `YuE2Tokenizer.swift`; the vocabulary
  (`qwen.tiktoken`, Alibaba Qwen) ships with the checkpoint.
- **mlx-swift** — Apple, MIT.
- `Support/NPY.swift` and the WAV helpers are lifted from xocialize/mlx-step-audio-editx-swift / mlx-auk-swift (MIT).
