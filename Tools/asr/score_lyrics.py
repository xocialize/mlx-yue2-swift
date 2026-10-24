#!/usr/bin/env python3
"""V2b lyric check: transcribe song renders with mlx-whisper large-v3-turbo (temperature 0) and score against the lyric
sheet — WER (en, Whisper's EnglishTextNormalizer-like lowercase/punct strip) or CER (zh, whitespace/punct stripped).
Sung lyrics over a full mix are hard for ASR, so the number that matters is the DRIFT between tiers and the fp32 lane,
not the absolute rate (pre-registered rule: within 10 points of bf16).

  score_lyrics.py --lyrics <request.json> --lang zh|en <wav|flac>...
"""
import argparse, json, re, string, glob, os
import jiwer, mlx_whisper

WHISPER = (glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/*/"))
           or glob.glob("/Volumes/Satechi/Models/models--mlx-community--whisper-large-v3-turbo/snapshots/*/"))[0]

def norm(text, lang):
    text = re.sub(r"\[[^\]]*\]", " ", text)                     # [Verse] markers
    if lang == "zh":
        return re.sub(r"[\s%s，。！？、：；“”‘’（）…—]+" % re.escape(string.punctuation), "", text)
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation.replace("'", "")), " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lyrics", required=True); ap.add_argument("--lang", required=True, choices=["zh", "en"])
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    lyrics = norm(json.load(open(a.lyrics, encoding="utf-8"))["lyrics"], a.lang)
    for f in a.files:
        r = mlx_whisper.transcribe(f, path_or_hf_repo=WHISPER, language=a.lang, temperature=0.0, condition_on_previous_text=False)
        hyp = norm(r["text"], a.lang)
        if a.lang == "zh":
            rate = jiwer.cer(lyrics, hyp) if hyp else 1.0
        else:
            rate = jiwer.wer(lyrics, hyp) if hyp else 1.0
        print(f"{os.path.basename(f):48s} {'CER' if a.lang=='zh' else 'WER'} {rate*100:6.1f} %   | {hyp[:110]}")

if __name__ == "__main__":
    main()
