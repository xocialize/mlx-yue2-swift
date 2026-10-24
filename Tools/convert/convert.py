#!/usr/bin/env python3
"""V1 converter: m-a-p/YuE2-3B + YuE2-Vae(-legacy) → MLX-layout safetensors bundles for mlx-yue2-swift (AB-T-0350).

Backbone: every key is kept verbatim (isomorphic port). The shipped dtype is bf16; `--dtype float32` casts it
exactly for the parity gate. `latent_pos_embed.pe` is kept as the checkpoint's bf16-rounded table (upstream loads
it rather than recomputing it).
VAE: decoder only. weight_norm (weight_g, weight_v) is folded to `weight` with torch's own `_weight_norm` (dim 0,
as the checkpoint's weight_g shapes say); Conv1d (O, I, K) → (O, K, I); ConvTranspose1d (I, O, K) → (O, K, I);
SnakeBeta alpha/beta kept (log-scale). The encoder is dropped. Always fp32 (upstream refuses any other dtype).

Outputs under --out:
  YuE2-3B-<dtype>/model.safetensors + config.json + qwen.tiktoken + licence files
  YuE2-Vae-fp32/decoder.safetensors + config.json + licence files      (and YuE2-Vae-legacy-fp32)

    .venv/bin/python -I convert.py --model <snap> --vae <snap> [--vae-legacy <snap>] --github <YuE clone> --out <dir>
                                   [--dtype float32|bfloat16] [--parts backbone,vae,vae-legacy]
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

LICENCE_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md", "README.md")


def load_all(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def licences(snap: Path, github: Path | None, out: Path, repo: str) -> None:
    for name in LICENCE_FILES:
        if (snap / name).exists():
            shutil.copyfile(snap / name, out / name)
    if (snap / "licenses").is_dir():
        shutil.copytree(snap / "licenses", out / "licenses", dirs_exist_ok=True)
    if github is not None and (github / "MODEL_LICENSE").exists():
        shutil.copyfile(github / "MODEL_LICENSE", out / "MODEL_LICENSE")
    (out / "NOTICE").write_text(
        f"YuE2 — {repo} by M-A-P (https://huggingface.co/{repo}), weights CC BY-NC 4.0.\n"
        "This bundle is an MLX-layout conversion of the released checkpoint (dtype cast / weight_norm folded /\n"
        "conv layouts transposed); the model is otherwise unmodified. Attribution per the weights' licence:\n"
        f"YuE2, {repo.split('/')[-1]}, https://huggingface.co/{repo}.\n"
        "LICENSE is the Hugging Face repo's text (plain CC BY-NC 4.0 + scope header). MODEL_LICENSE is the GitHub\n"
        "release's text (multimodal-art-projection/YuE@1647252), which adds the 'Individual creator permission and\n"
        "academic use (2026-09-16)' section: individuals acting in a personal capacity may publish, sell and license\n"
        "songs made with the model; companies may not use it commercially without a licence from M-A-P\n"
        "(ryuanab@connect.ust.hk). Code licences: Apache-2.0 (YuE), MIT (Oobleck / SnakeBeta; licenses/).\n")


def convert_backbone(snap: Path, github: Path | None, out: Path, dtype: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sd = load_all(snap / "model.safetensors")
    target = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    tensors = {k: v.to(target).contiguous() for k, v in sd.items()}
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "mlx"})
    cfg = json.loads((snap / "config.json").read_text())
    cfg["mlx_yue2"] = {"dtype": dtype, "source_repo": "m-a-p/YuE2-3B", "source_revision": snap.name,
                       "upstream_code": "multimodal-art-projection/YuE@1647252",
                       "generation": json.loads((snap / "yue2_generation_config.json").read_text())}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    shutil.copyfile(snap / "qwen.tiktoken", out / "qwen.tiktoken")
    licences(snap, github, out, "m-a-p/YuE2-3B")
    print(f"backbone → {out}: {len(tensors)} tensors, {dtype}")


def convert_vae(snap: Path, github: Path | None, out: Path, repo: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sd = load_all(snap / "model.safetensors")
    tensors: dict[str, torch.Tensor] = {}
    folded = 0
    for k, v in sd.items():
        if not k.startswith("decoder."):
            continue
        if k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            g, vv = v, sd[base + ".weight_v"]
            w = torch._weight_norm(vv, g, 0)
            # ConvTranspose1d: torch (I, O, K) → MLX (O, K, I); Conv1d: torch (O, I, K) → MLX (O, K, I).
            # The transposed convs are exactly the `layers.1` child of each DecoderBlock (decoder.layers.{1..6}.layers.1).
            parts = base.split(".")
            is_transposed = len(parts) == 5 and parts[1] == "layers" and parts[3] == "layers" and parts[4] == "1"
            w = w.permute(1, 2, 0) if is_transposed else w.permute(0, 2, 1)
            tensors[base + ".weight"] = w.contiguous().float()
            folded += 1
        elif k.endswith(".weight_v"):
            continue
        else:
            tensors[k] = v.contiguous().float()
    assert all(t.dtype == torch.float32 for t in tensors.values())
    save_file(tensors, str(out / "decoder.safetensors"), metadata={"format": "mlx"})
    cfg = json.loads((snap / "config.json").read_text())
    cfg["mlx_yue2"] = {"dtype": "float32", "source_repo": repo, "source_revision": snap.name, "decoder_only": True,
                       "conv_layout": "OKI", "weight_norm": "folded"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    licences(snap, github, out, repo)
    print(f"vae → {out}: {len(tensors)} tensors ({folded} weight-norm folds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--vae", type=Path, required=True)
    ap.add_argument("--vae-legacy", type=Path)
    ap.add_argument("--github", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--parts", default="backbone,vae,vae-legacy")
    a = ap.parse_args()
    parts = a.parts.split(",")
    if "backbone" in parts:
        convert_backbone(a.model, a.github, a.out / f"YuE2-3B-{'fp32' if a.dtype == 'float32' else 'bf16'}", a.dtype)
    if "vae" in parts:
        convert_vae(a.vae, a.github, a.out / "YuE2-Vae-fp32", "m-a-p/YuE2-Vae")
    if "vae-legacy" in parts and a.vae_legacy:
        convert_vae(a.vae_legacy, a.github, a.out / "YuE2-Vae-legacy-fp32", "m-a-p/YuE2-Vae-legacy")


if __name__ == "__main__":
    main()
