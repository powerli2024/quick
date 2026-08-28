#!/usr/bin/env bash
set -euo pipefail

# One-command A3 deployment.  Raw s1/s7 files are read from POS_NEG; previous
# MossFormer-48k files are reused by hash when PRECOMPUTED_SE_DIR is present.
QUICK_DIR="${QUICK_DIR:-/root/quick}"
POS_NEG="${POS_NEG:?set POS_NEG to the extract-sep full-audio root}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/kws_a3_route}"
S1_ARM="${S1_ARM:-s1_onnx_full}"
S7_ARM="${S7_ARM:-s7_cv_then_onnx_gate/thr_a}"
EXPECTED_UIDS="${EXPECTED_UIDS:-0}"
DEVICE="${DEVICE:-cuda:0}"
SENSEVOICE_DIR="${SENSEVOICE_DIR:-/root/autodl-tmp/quick_models/a3/SenseVoiceSmall}"
WENET_DIR="${WENET_DIR:-/root/autodl-tmp/quick_models/a3/wenet_aishell_u2pp_conformer_exp}"
PRECOMPUTED_SE_DIR="${PRECOMPUTED_SE_DIR:-/root/autodl-tmp/kws_se48k}"

if [[ ! -d "$SENSEVOICE_DIR" ]]; then
  echo "missing SenseVoice model: $SENSEVOICE_DIR (run scripts/download_a3_models.sh)" >&2
  exit 2
fi

ASR_COMMAND="${ASR_COMMAND:-python $QUICK_DIR/scripts/score_sensevoice_manifest.py --manifest {manifest} --output {output} --model-dir $SENSEVOICE_DIR --device $DEVICE}"

args=(
  "$QUICK_DIR/scripts/run_s1_s7_route.py"
  --pos-neg "$POS_NEG" --s1-arm "$S1_ARM" --s7-arm "$S7_ARM"
  --expected-uids "$EXPECTED_UIDS" --work-dir "$WORK_DIR"
  --asr-command "$ASR_COMMAND" --asr-model-dir "$SENSEVOICE_DIR"
  --asr-context-mode none --qkw-switch-margin "${QKW_SWITCH_MARGIN:-0.01}"
  --alias-json "${ALIAS_JSON:-$QUICK_DIR/configs/english_alias.json}"
  --policy-json "${POLICY_JSON:-$QUICK_DIR/configs/route_policy.json}"
  --mossformer-model-hash "${MOSSFORMER_MODEL_HASH:-a3_reused_precomputed_se}"
  --inference-signature "${INFERENCE_SIGNATURE:-a3_sensevoice_wenet_ctc}"
  --selected-only-dir "${SELECTED_ONLY_DIR:-$WORK_DIR/best_sep_selected}"
)

if [[ -n "${QKW_JSONL:-}" ]]; then
  args+=(--qkw-jsonl "$QKW_JSONL")
else
  WENET_DECODE_COMMAND="${WENET_DECODE_COMMAND:-python $QUICK_DIR/scripts/wenet_decode_manifest.py --manifest {manifest} --output {output} --model-dir $WENET_DIR --wenet-repo ${WENET_REPO:-/root/wenet}}"
  QKW_COMMAND="python $QUICK_DIR/scripts/score_wenet_ctc_manifest.py --manifest {manifest} --output {output} --model-dir $WENET_DIR --decode-command \"$WENET_DECODE_COMMAND\" --enabled-langs ${WENET_ENABLED_LANGS:-zh}"
  args+=(--qkw-command "$QKW_COMMAND")
fi

if [[ -d "$PRECOMPUTED_SE_DIR" ]]; then
  args+=(--se-backend precomputed --precomputed-se-dir "$PRECOMPUTED_SE_DIR")
elif [[ -n "${SE_BATCH_COMMAND:-}" ]]; then
  args+=(--se-backend command --se-batch-command "$SE_BATCH_COMMAND")
else
  echo "no reusable SE cache: set PRECOMPUTED_SE_DIR or SE_BATCH_COMMAND" >&2
  exit 2
fi

[[ "${RESUME:-1}" == "1" ]] || args+=(--no-resume)
exec python "${args[@]}"
