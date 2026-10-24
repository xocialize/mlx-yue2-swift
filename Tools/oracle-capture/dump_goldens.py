#!/usr/bin/env python3
"""V0 oracle for the YuE2-3B MLX-Swift port (AB-T-0350): fp32 CPU goldens, injected noise.

Runs upstream `yue2` (multimodal-art-projection/YuE@1647252) with the backbone in fp32 on the CPU and dumps, per
case: the tokenizer strings/ids, the exact prefixes, a seeded rollout (abc + semantic tokens; reference only), the
song noise, teacher-forced AR logits (lane A), one NAR velocity at t=1 and t=0.5 (lane N1), a 32-step midpoint
solve on a 256-frame chunk (lane N2), the full-song latents (lane E), and VAE decodes full + tiled (lane D).

  dump_goldens.py --model <YuE2-3B snapshot> --vae <YuE2-Vae snapshot> --out <goldens> [--cases a,b] [--semantic-cap N]
"""
import argparse, json, time, math
from pathlib import Path
import numpy as np
import torch

CASES = {
    # The released example (examples/tonight-awake.json, seed 12300) cut to its first four sections.
    "tonight-awake-short": dict(
        style="City Pop, upbeat, danceable, groovy bass, electric guitar, synth, energetic, joyful, neon city night",
        lyrics="[Intro]\n\n[Verse]\n路灯眨着眼睛 偷看谁的身影\n街道哼着小调 节奏多轻盈\n晚风染成霓虹 吹乱发际线\n脚步踩着鼓点 不需要终点\n\n"
               "[Pre-Chorus]\n旋转的唱片 划破了寂静\n气泡在上升 快乐在飞行\n把烦恼抛去 别再去在意\n这里的空气 充满了魔力\n\n"
               "[Chorus]\n今晚不眠 快乐无限\n城市在狂欢 我们在中间\n自由摇摆 光芒盛开\n跟着这节拍 把心打开\n",
        cot="full", seed=12300),
    # An English request without a symbolic plan: exercises cot=off (two-branch CFG at 1.01, legacy_off sampling).
    "stay-awhile-off": dict(
        style="Indie folk, acoustic guitar, warm female vocal, intimate, slow, gentle piano",
        lyrics="[Verse]\nMorning light on the kitchen floor\nCoffee steam and an open door\nYou hum a tune I've heard before\n"
               "And I don't need anything more\n\n[Chorus]\nStay a while, stay a while\nLet the quiet hold us for a while\n"
               "Stay a while, stay a while\nNothing here we have to reconcile\n",
        cot="off", seed=831001),
}
NAR_FRAMES = 256  # the N1/N2 chunk (10.24 s)
TAIL = 80         # lane A keeps full logits for the last TAIL positions (argmax for all)


def raw_t(t):
    return torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--vae", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--semantic-cap", type=int, default=1500, help="semantic max_tokens for the rollout (60 s at 1500)")
    ap.add_argument("--abc-cap", type=int, default=4096)
    args = ap.parse_args()

    from yue2.protocol import (SongRequest, Sampling, GenerationConfig, INSTRUCTIONS, token_prefixes, negative_prefix,
                               CODEC_OFFSET, MUSIC_END)
    from yue2.sampling import generate_tokens
    from yue2.nar import song_chunks, CachedNAR, synthesize
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.modeling_vae import YuE2VAE
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    from dataclasses import replace

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tok = YuE2TextTokenizer(args.model / "qwen.tiktoken")
    t0 = time.perf_counter()
    model = YuE2ForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float32).eval()
    assert next(model.parameters()).dtype == torch.float32
    vae = YuE2VAE.from_pretrained(args.vae, decoder_only=True, device="cpu", local_files_only=True)
    print(f"loaded fp32 backbone + decoder in {time.perf_counter()-t0:.1f}s; threads {torch.get_num_threads()}", flush=True)
    gen = GenerationConfig()

    for name in args.cases.split(","):
        spec = CASES[name]
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        req = SongRequest(**spec)
        timing = {}
        abc_s = replace(gen.abc, max_tokens=args.abc_cap)
        sem_s = replace(gen.semantic, max_tokens=args.semantic_cap)

        # --- plan (abc) ---
        base_prefix = token_prefixes(req, tok)
        abc_ids, abc_text, truncated = [], None, {}
        if req.cot != "off":
            t1 = time.perf_counter()
            abc_ids, tm, tr = generate_tokens(model, base_prefix, abc_s, req.seed, "abc", use_cuda_graph=False)
            timing["abc"] = tm; truncated["abc"] = tr
            abc_text = tok.decode(abc_ids)
            print(f"[{name}] abc: {len(abc_ids)} tokens in {time.perf_counter()-t1:.0f}s (truncated={tr})", flush=True)
            assert tok.encode(abc_text) == abc_ids, "abc round trip is not exact"
        prefix = token_prefixes(req, tok, abc_ids)
        negative = negative_prefix(req, tok, abc_ids) if req.guidance != 1 else None

        # --- semantic rollout (reference only; AB-L-0090) ---
        t1 = time.perf_counter()
        sem_ids, tm, tr = generate_tokens(model, prefix, sem_s, req.seed, "semantic", negative=negative,
                                          cfg_scale=req.guidance, legacy_off=req.cot == "off", use_cuda_graph=False)
        timing["semantic"] = tm; truncated["semantic"] = tr
        codec = [i - CODEC_OFFSET for i in sem_ids]
        assert all(0 <= c < 32768 for c in codec)
        print(f"[{name}] semantic: {len(codec)} tokens in {time.perf_counter()-t1:.0f}s (truncated={tr})", flush=True)

        # --- tokenizer + protocol goldens ---
        strings = [req.text(), INSTRUCTIONS[req.cot], req.style, req.lyrics] + ([abc_text] if abc_text else [])
        json.dump({"strings": strings, "ids": [tok.encode(s) for s in strings]}, (out / "tokenizer.json").open("w"),
                  ensure_ascii=False)
        json.dump({**spec, "cfg_scale": req.guidance, "semantic_cap": args.semantic_cap, "abc_cap": args.abc_cap,
                   "truncated": truncated, "instruction": INSTRUCTIONS[req.cot]}, (out / "request.json").open("w"),
                  ensure_ascii=False, indent=1)
        np.save(out / "prefix.npy", np.asarray(prefix, dtype=np.int32))
        np.save(out / "abc_ids.npy", np.asarray(abc_ids, dtype=np.int32))
        if abc_text is not None:
            (out / "abc.txt").write_text(abc_text, encoding="utf-8")
        if negative is not None:
            np.save(out / "negative_prefix.npy", np.asarray(negative, dtype=np.int32))
        np.save(out / "semantic.npy", np.asarray(codec, dtype=np.int32))

        # --- song noise (torch.randn fp32 CPU from the seed, the whole song) ---
        chunks = song_chunks(prefix, codec, req.seed)
        noise = torch.cat([c.noise for c in chunks]).numpy()
        np.save(out / "song_noise.npy", noise)

        # --- lane A: teacher-forced AR logits over prefix + 64 codec tokens (and the negative branch) ---
        def lane_a(ids, tag):
            with torch.inference_mode():
                logits = model(torch.tensor([ids]), use_cache=False, logits_to_keep=0).logits[0].float()
            np.save(out / f"ar_tf_{tag}_ids.npy", np.asarray(ids, dtype=np.int32))
            np.save(out / f"ar_tf_{tag}_argmax.npy", logits.argmax(-1).numpy().astype(np.int32))
            np.save(out / f"ar_tf_{tag}_logits_tail.npy", logits[-TAIL:].numpy())
        t1 = time.perf_counter()
        lane_a(prefix + [c + CODEC_OFFSET for c in codec[:64]], "pos")
        if negative is not None:
            lane_a(negative + [c + CODEC_OFFSET for c in codec[:64]], "neg")
        print(f"[{name}] lane A in {time.perf_counter()-t1:.0f}s", flush=True)

        # --- lane E: full-song latents (all chunks, injected noise inside synthesize) + VAE ---
        t1 = time.perf_counter()
        latents = synthesize(model, prefix, codec, req.seed, steps=32).numpy()
        timing["nar_full_seconds"] = time.perf_counter() - t1
        np.save(out / "latents_full.npy", latents)
        print(f"[{name}] full NAR: {latents.shape} in {timing['nar_full_seconds']:.0f}s", flush=True)
        t1 = time.perf_counter()
        z = torch.from_numpy(latents).T[None]
        audio = vae.decode_tiled(z, core_frames=1024, halo_frames=16)[0].clamp(-1, 1).numpy()
        timing["vae_full_seconds"] = time.perf_counter() - t1
        np.save(out / "audio_ref.npy", audio)
        import soundfile as sf
        sf.write(out / "audio_ref.flac", audio.T, 48000, subtype="PCM_24")
        print(f"[{name}] VAE: {audio.shape} in {timing['vae_full_seconds']:.0f}s", flush=True)

        # --- lanes N1/N2/D on a 256-frame chunk ---
        n = min(NAR_FRAMES, len(codec))
        from yue2.nar import Chunk
        chunk = Chunk(prefix + [c + CODEC_OFFSET for c in codec[:n]] + [MUSIC_END], torch.from_numpy(noise[:n]).clone())
        np.save(out / "nar_chunk_ids.npy", np.asarray(chunk.ar_tokens, dtype=np.int32))
        t1 = time.perf_counter()
        eng = CachedNAR(model, chunk)
        s1 = chunk.noise.clone()
        v1 = eng.velocity(s1, raw_t(1.0))
        s05 = 0.5 * chunk.noise + 0.5 * torch.from_numpy(latents[:n])
        v05 = eng.velocity(s05, raw_t(0.5))
        np.save(out / "nar_state_t1.npy", s1.numpy()); np.save(out / "nar_v_t1.npy", v1.float().numpy())
        np.save(out / "nar_state_t05.npy", s05.numpy()); np.save(out / "nar_v_t05.npy", v05.float().numpy())
        lat_n = eng.solve(32)
        eng.close()
        np.save(out / "nar_latents.npy", lat_n.numpy())
        timing["nar_chunk_seconds"] = time.perf_counter() - t1
        json.dump({"raw_t1": raw_t(1.0), "raw_t05": raw_t(0.5), "frames": n, "ar_len": len(chunk.ar_tokens)},
                  (out / "nar_meta.json").open("w"))
        with torch.inference_mode():
            zc = lat_n.T[None]
            full = vae.decode(zc)[0].numpy()
            tiled = vae.decode_tiled(zc, core_frames=64, halo_frames=16)[0].numpy()
        np.save(out / "vae_audio_full.npy", full); np.save(out / "vae_audio_tiled64.npy", tiled)
        print(f"[{name}] N1/N2/D in {timing['nar_chunk_seconds']:.0f}s; tiled-vs-full max|Δ| {np.abs(full-tiled).max():.2e}",
              flush=True)
        json.dump(timing, (out / "timing.json").open("w"), indent=1, default=float)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
