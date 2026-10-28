#!/bin/zsh
# V3 live lanes, one model process at a time: `validate` (through MLXServeEngine) then `envelope`, per tier; OFF=1 adds the
# cot=off request through `validate` at bf16.
#   run_v3.sh <work-dir>   — <work-dir> holds weights/{YuE2-3B-bf16,YuE2-Vae-fp32}, v3/{city_lights,stay_awhile_off}.json,
#                            a score at ref/YuE/examples/score.abc; logs go to <work-dir>/receipts, audio to <work-dir>/v3/out*
#   TIERS="bf16 ar8 int8" (default), OFF=0|1
set -u
W=${1:?usage: run_v3.sh <work-dir>}
G=${YUE2_GATES:-${0:A:h}/../.build/release/yue2-gates}
export MLX_ENABLE_TF32=0
for t in ${=TIERS:-bf16 ar8 int8}; do
  $G --tier $t --cache-gb 1 validate $W/weights/YuE2-3B-bf16 $W/weights/YuE2-Vae-fp32 $W/v3/city_lights.json $W/v3/out > $W/receipts/v3-validate-$t.log 2>&1
  echo "validate $t exit $?"
  $G --tier $t --cache-gb 1 envelope $W/weights/YuE2-3B-bf16 $W/weights/YuE2-Vae-fp32 $W/ref/YuE/examples/score.abc > $W/receipts/v3-envelope-$t.log 2>&1
  echo "envelope $t exit $?"
done
if [[ ${OFF:-0} == 1 ]]; then
  $G --tier bf16 --cache-gb 1 validate $W/weights/YuE2-3B-bf16 $W/weights/YuE2-Vae-fp32 $W/v3/stay_awhile_off.json $W/v3/out-off > $W/receipts/v3-validate-bf16-off.log 2>&1
  echo "validate bf16 off exit $?"
fi
