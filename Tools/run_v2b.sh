#!/bin/bash
# V2b tier matrix (PORTING-SPEC §3): nar + e2e lanes per tier on both goldens, then two real renders (bf16; 8-bit AR + bf16 NAR).
cd "$(dirname "$0")"
export MLX_ENABLE_TF32=0
B=/Volumes/Satechi/Development/mlxengine-audio/PROD/YuE2/mlx-yue2-swift/.build/out/Products/Release/yue2-gates
OUT=receipts/v2b-tiers.txt; : > "$OUT"
run() { echo "== $*" | tee -a "$OUT"; "$B" "$@" 2>&1 | grep -E "max\||LANE|NAR \(|solve|ERROR|loaded" | tee -a "$OUT"; }
for tier in "bf16:--dtype bf16" "ar8-narbf16:--dtype bf16 --ar-bits 8" "ar8-nar8:--dtype bf16 --ar-bits 8 --nar-bits 8" "ar4-nar8:--dtype bf16 --ar-bits 4 --nar-bits 8"; do
  name=${tier%%:*}; flags=${tier#*:}
  for c in tonight-awake-short stay-awhile-off; do
    echo "### tier $name case $c" | tee -a "$OUT"
    run $flags nar oracle/goldens/$c weights/YuE2-3B-bf16
    run $flags e2e oracle/goldens/$c weights/YuE2-3B-bf16 weights/YuE2-Vae-fp32 v2b/$c-$name-e2e.wav
  done
done
echo "### renders" | tee -a "$OUT"
for tier in "bf16:--dtype bf16" "ar8-narbf16:--dtype bf16 --ar-bits 8"; do
  name=${tier%%:*}; flags=${tier#*:}
  echo "== render $name" | tee -a "$OUT"
  "$B" $flags render weights/YuE2-3B-bf16 weights/YuE2-Vae-fp32 oracle/goldens/tonight-awake-short/render-request.json v2b/render-tonight-awake-$name.wav 2>&1 | tee -a "$OUT"
done
echo V2B-DONE | tee -a "$OUT"
