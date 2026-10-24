#!/usr/bin/env python3
"""V1 gate: (1) converted VAE tensors equal upstream's own remove_weight_norm output, layout-adjusted back;
(2) backbone tensors are bit-equal to the checkpoint (cast only); (3) MLX conv / conv_transpose semantics with the
converted layouts reproduce torch on one DecoderBlock and the first/last convs (so the Swift port inherits a checked
layout, not a guess).   gate_v1.py --vae <snap> --model <snap> --weights <converted dir>"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", type=Path, required=True); ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    a = ap.parse_args()
    from yue2.modeling_vae import YuE2VAE
    vae = YuE2VAE.from_pretrained(a.vae, decoder_only=True, device="cpu", local_files_only=True)
    for m in vae.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
    ref = {k: v for k, v in vae.state_dict().items()}
    worst, n, got = 0.0, 0, {}
    with safe_open(str(a.weights / "YuE2-Vae-fp32" / "decoder.safetensors"), "pt") as f:
        for k in f.keys():
            got[k] = f.get_tensor(k)
    missing, extra = set(ref) - set(got), set(got) - set(ref)
    for k, v in got.items():
        if k not in ref: continue
        if v.ndim == 3:
            parts = k.split(".")
            transposed = len(parts) == 6 and parts[3] == "layers" and parts[4] == "1"
            v = v.permute(2, 0, 1) if transposed else v.permute(0, 2, 1)
        d = (v.float() - ref[k].float()).abs().max().item(); worst = max(worst, d); n += 1
    print(f"VAE: {n} tensors compared, max|Δ| = {worst:.3e}, missing={sorted(missing)[:4]} ({len(missing)}), extra={sorted(extra)[:4]} ({len(extra)})")
    ok_vae = worst == 0.0 and not missing and not extra

    # (2) backbone bit-equality (bf16 bundle vs checkpoint)
    worst_b, nb = 0.0, 0
    with safe_open(str(a.model / "model.safetensors"), "pt") as src, \
         safe_open(str(a.weights / "YuE2-3B-bf16" / "model.safetensors"), "pt") as dst:
        ks, kd = set(src.keys()), set(dst.keys())
        assert ks == kd, (ks - kd, kd - ks)
        for k in sorted(ks):
            d = (src.get_tensor(k).float() - dst.get_tensor(k).float()).abs().max().item(); worst_b = max(worst_b, d); nb += 1
    print(f"backbone bf16: {nb} tensors, max|Δ| = {worst_b:.3e}")

    # (3) MLX conv semantics on the converted layouts: the first conv (k7 p3), one DecoderBlock's ConvTranspose
    #     (k12 s6 p3 — the first block), and the final conv (k7 p3, no bias), against torch with folded weights.
    dec = vae.decoder.layers
    x = torch.randn(1, 64, 37)
    def run_conv(mod, w_key, pad, stride=1, transposed=False):
        w = got[w_key]; b = got.get(w_key.replace(".weight", ".bias"))
        xm = mx.array(x.permute(0, 2, 1).numpy())  # MLX is NLC
        if transposed:
            y = mx.conv_transpose1d(xm, mx.array(w.numpy()), stride=stride, padding=pad)
        else:
            y = mx.conv1d(xm, mx.array(w.numpy()), stride=stride, padding=pad)
        if b is not None: y = y + mx.array(b.numpy())
        with torch.no_grad(): yt = mod(x)
        return np.abs(np.array(y).transpose(0, 2, 1) - yt.numpy()).max() / (np.abs(yt.numpy()).max() + 1e-12)
    e0 = run_conv(dec[0], "decoder.layers.0.weight", 3)
    block = dec[1]  # DecoderBlock 2048→1024, stride 6: layers.1 is the ConvTranspose
    x = torch.randn(1, 2048, 9)
    e1 = run_conv(block.layers[1], "decoder.layers.1.layers.1.weight", 3, stride=6, transposed=True)
    x = torch.randn(1, 64, 200)
    e2 = run_conv(dec[-2], f"decoder.layers.{len(dec)-2}.weight", 3)
    print(f"MLX conv semantics: conv_pre relmax {e0:.2e}; ConvTranspose(k12,s6,p3) relmax {e1:.2e}; conv_post relmax {e2:.2e}")
    ok_sem = max(e0, e1, e2) < 1e-5
    print("V1 GATE:", "PASS" if ok_vae and worst_b == 0.0 and ok_sem else "FAIL")


if __name__ == "__main__":
    main()
