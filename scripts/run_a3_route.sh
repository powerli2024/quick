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
PRECOMPUTED_SE_DIR="${PRECOMPUTED_SE_DIR:-/root/autodl-tmp/kws_se_route/se_wav}"

if [[ ! -d "$SENSEVOICE_DIR" ]]; then
  echo "missing SenseVoice model: $SENSEVOICE_DIR (run scripts/download_a3_models.sh)" >&2
  exit 2
fi

if [[ -z "${ASR_COMMAND:-}" || "$ASR_COMMAND" != *"{manifest}"* || "$ASR_COMMAND" != *"{output}"* ]]; then
  # Do not use ${VAR:-...} here: Bash treats braces in the default value as
  # parameter-expansion syntax and turns {manifest} into {manifest.
  ASR_COMMAND="python ${QUICK_DIR}/scripts/score_sensevoice_manifest.py --manifest {manifest} --output {output} --model-dir ${SENSEVOICE_DIR} --device ${DEVICE}"
fi

args=(
  "$QUICK_DIR/scripts/run_s1_s7_route.py"
  --pos-neg "$POS_NEG" --s1-arm "$S1_ARM" --s7-arm "$S7_ARM"
  --expected-uids "$EXPECTED_UIDS" --work-dir "$WORK_DIR"
  --asr-model-dir "$SENSEVOICE_DIR"
  --asr-context-mode none --qkw-switch-margin "${QKW_SWITCH_MARGIN:-0.01}"
  --alias-json "${ALIAS_JSON:-$QUICK_DIR/configs/english_alias.json}"
  --policy-json "${POLICY_JSON:-$QUICK_DIR/configs/route_policy.json}"
  --mossformer-model-hash "${MOSSFORMER_MODEL_HASH:-a3_reused_precomputed_se}"
  --inference-signature "${INFERENCE_SIGNATURE:-a3_sensevoice_wenet_ctc}"
  --selected-only-dir "${SELECTED_ONLY_DIR:-$WORK_DIR/best_sep_selected}"
)

if [[ -n "${ASR_JSONL:-}" && -f "$ASR_JSONL" ]]; then
  echo "[A3][reuse] SenseVoice sidecar: $ASR_JSONL" >&2
  args+=(--asr-jsonl "$ASR_JSONL")
else
  args+=(--asr-command "$ASR_COMMAND")
fi

if [[ -n "${QKW_JSONL:-}" ]]; then
  if [[ -f "$QKW_JSONL" ]]; then
    args+=(--qkw-jsonl "$QKW_JSONL")
  else
    echo "[A3][WARN] QKW_JSONL not found; continue without CTC sidecar: $QKW_JSONL" >&2
  fi
else
  WENET_READY=0
  WENET_REPO_PATH="${WENET_REPO:-/root/wenet}"
  if [[ -f "$WENET_REPO_PATH/wenet/bin/recognize.py" ]]; then
    # The wrapper also checks that both files exist; this cheap probe avoids
    # aborting after SenseVoice has already finished when CTC is unavailable.
    WENET_CONFIG="$(find "$WENET_DIR" -type f -name train.yaml -print -quit 2>/dev/null || true)"
    WENET_CHECKPOINT="$(find "$WENET_DIR" -type f -name final.pt -print -quit 2>/dev/null || true)"
    if [[ -n "$WENET_CONFIG" && -n "$WENET_CHECKPOINT" ]]; then
      WENET_READY=1
    fi
  fi
  if [[ -n "${WENET_DECODE_COMMAND:-}" || "$WENET_READY" == "1" ]]; then
    if [[ -z "${WENET_DECODE_COMMAND:-}" ]]; then
      WENET_DECODE_COMMAND="python ${QUICK_DIR}/scripts/wenet_decode_manifest.py --manifest {manifest} --output {output} --model-dir ${WENET_DIR} --wenet-repo ${WENET_REPO_PATH}"
    fi
    QKW_COMMAND="python $QUICK_DIR/scripts/score_wenet_ctc_manifest.py --manifest {manifest} --output {output} --model-dir $WENET_DIR --decode-command \"$WENET_DECODE_COMMAND\" --enabled-langs ${WENET_ENABLED_LANGS:-zh}"
    args+=(--qkw-command "$QKW_COMMAND")
  else
    echo "[A3][WARN] WeNet CTC unavailable; continue with SenseVoice only (set QKW_JSONL or install model/source to enable CTC)" >&2
  fi
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
