#!/usr/bin/env bash
# Frozen s1/s7 + MossFormer route on AutoDL Linux.
# SE cosine/CER pairing is diagnostic only. s7_arm=auto is forbidden.
set -euo pipefail
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

# Some AutoDL images inherit an invalid OMP_NUM_THREADS value.  libgomp emits
# an error before Python starts, so normalize it to a safe integer.
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

REPO_DIR="${REPO_DIR:-/root/quick}"
KWS_DIR="${KWS_DIR:-/root/kws}"
KWS_SRC="${KWS_SRC:-$KWS_DIR/src}"
POS_NEG="${POS_NEG:-/root/autodl-tmp/kws_sep}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/quick_s1_s7}"
AUDIO_CACHE_ROOT="${AUDIO_CACHE_ROOT:-/root/autodl-tmp/quick_audio_cache}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-/root/autodl-tmp/Qwen3-ASR-1.7B}"
EXTRACT_MAIN="${EXTRACT_MAIN:-/root/extract-main}"
CLEARVOICE_ROOT="${CLEARVOICE_ROOT:-/root/autodl-tmp/ClearerVoice-Studio}"
S1_ARM="${S1_ARM:-s1_onnx_full}"
S7_ARM="${S7_ARM:-}"
SE_BACKEND="${SE_BACKEND:-command}"
SE_COMMAND="${SE_COMMAND:-}"
SE_BATCH_COMMAND="${SE_BATCH_COMMAND:-}"
PRECOMPUTED_SE_DIR="${PRECOMPUTED_SE_DIR:-}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EXPECTED_UIDS="${EXPECTED_UIDS:-1838}"
RESUME="${RESUME:-1}"
WITH_NLL="${WITH_NLL:-1}"
WITH_QWEN_Q1="${WITH_QWEN_Q1:-0}"
WITH_EMBED="${WITH_EMBED:-0}"
ASR_CONTEXT_MODE="${ASR_CONTEXT_MODE:-none}"
ASR_CACHE_DIR="${ASR_CACHE_DIR:-/root/autodl-tmp/quick_asr_cache}"
DURATION_BUCKET_SEC="${DURATION_BUCKET_SEC:-0.5}"
EXTRACT_SEP_RUN_ID="${EXTRACT_SEP_RUN_ID:-}"
ALIAS_JSON="${ALIAS_JSON:-$REPO_DIR/configs/english_alias.json}"
QKW_JSONL="${QKW_JSONL:-}"
QKW_CALIBRATED="${QKW_CALIBRATED:-0}"
QKW_CALIBRATOR_JSON="${QKW_CALIBRATOR_JSON:-}"
INFERENCE_SIGNATURE="${INFERENCE_SIGNATURE:-mossformer2_se_48k_full_waveform_v1}"

if [[ -z "$S7_ARM" || "$S7_ARM" == "auto" ]]; then
  echo "[ERR] lock S7_ARM to an exact pos/neg-identical label; auto is forbidden" >&2
  exit 1
fi
test -d "$REPO_DIR"
test -d "$POS_NEG/pos"
test -d "$POS_NEG/neg"
test -d "$KWS_SRC/kws"
cd "$REPO_DIR"
export KWS_SRC DEVICE ASR_MODEL_DIR ASR_CACHE_DIR
export PYTHONPATH="${REPO_DIR}/src:${KWS_SRC}${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$SE_BACKEND" == "command" && -z "$SE_COMMAND" && -z "$SE_BATCH_COMMAND" ]]; then
  SE_BATCH_COMMAND="python ${KWS_DIR}/scripts/extract_main_se48k_manifest.py --manifest {manifest} --extract-main ${EXTRACT_MAIN} --clearvoice-root ${CLEARVOICE_ROOT} --device ${DEVICE}"
fi

if [[ -z "${ASR_COMMAND:-}" || "$ASR_COMMAND" != *"{manifest}"* || "$ASR_COMMAND" != *"{output}"* ]]; then
  # Do not put {manifest}/{output} inside ${VAR:-...}; Bash treats the
  # placeholder braces as part of the parameter expansion and drops/moves
  # them in the resulting command.  Also replace any inherited malformed
  # command instead of passing it into the pipeline.
  ASR_COMMAND="python ${REPO_DIR}/scripts/score_asr_manifest.py --manifest {manifest} --output {output} --model-dir ${ASR_MODEL_DIR} --device ${DEVICE} --batch-size ${BATCH_SIZE} --context-mode ${ASR_CONTEXT_MODE} --cache-dir ${ASR_CACHE_DIR} --duration-bucket-sec ${DURATION_BUCKET_SEC}"
fi

args=(
  python scripts/run_s1_s7_route.py
  --pos-neg "$POS_NEG"
  --work-dir "$WORK_DIR"
  --audio-cache-root "$AUDIO_CACHE_ROOT"
  --s1-arm "$S1_ARM"
  --s7-arm "$S7_ARM"
  --expected-uids "$EXPECTED_UIDS"
  --se-backend "$SE_BACKEND"
  --asr-command "$ASR_COMMAND"
  --asr-model-dir "$ASR_MODEL_DIR"
  --asr-context-mode "$ASR_CONTEXT_MODE"
  --alias-json "$ALIAS_JSON"
  --selected-only-dir "$WORK_DIR/best_sep_selected"
)

if [[ -n "$EXTRACT_SEP_RUN_ID" ]]; then
  args+=(--extract-sep-run-id "$EXTRACT_SEP_RUN_ID")
fi
args+=(--inference-signature "$INFERENCE_SIGNATURE")
if [[ "$SE_BACKEND" == "command" ]]; then
  if [[ -n "$SE_BATCH_COMMAND" ]]; then
    args+=(--se-batch-command "$SE_BATCH_COMMAND")
  else
    test -n "$SE_COMMAND"
    args+=(--se-command "$SE_COMMAND")
  fi
elif [[ "$SE_BACKEND" == "precomputed" ]]; then
  test -n "$PRECOMPUTED_SE_DIR"
  args+=(--precomputed-se-dir "$PRECOMPUTED_SE_DIR")
fi
if [[ "$RESUME" != "1" ]]; then
  args+=(--no-resume)
fi
if [[ "$WITH_NLL" == "1" ]]; then
  args+=(--nll-command "python ${REPO_DIR}/scripts/score_nll_manifest.py --manifest {manifest} --output {output} --model-dir ${ASR_MODEL_DIR} --device ${DEVICE} --batch-size ${BATCH_SIZE}")
fi
if [[ "$WITH_EMBED" == "1" ]]; then
  args+=(--embedding-command "python ${REPO_DIR}/scripts/score_embed_manifest.py --manifest {manifest} --output {output} --device ${DEVICE}")
fi
if [[ -n "$QKW_JSONL" ]]; then
  test -f "$QKW_JSONL"
  args+=(--qkw-jsonl "$QKW_JSONL")
fi
if [[ "$QKW_CALIBRATED" == "1" ]]; then
  if [[ -z "$QKW_JSONL" || -z "$QKW_CALIBRATOR_JSON" || ! -f "$QKW_CALIBRATOR_JSON" ]]; then
    echo "[ERR] QKW_CALIBRATED=1 requires complete QKW_JSONL and QKW_CALIBRATOR_JSON" >&2
    exit 1
  fi
  args+=(--qkw-calibrated --qkw-calibrator-json "$QKW_CALIBRATOR_JSON")
fi
if [[ -n "${CMD_RESULT_JSON:-}" ]]; then
  args+=(--cmd-result-json "$CMD_RESULT_JSON")
fi
if [[ -n "${PRESENCE_RESULT_JSON:-}" ]]; then
  args+=(--presence-result-json "$PRESENCE_RESULT_JSON")
fi
if [[ -n "${CONTEST_RESULT_JSON:-}" ]]; then
  args+=(--contest-result-json "$CONTEST_RESULT_JSON")
fi

status=0
"${args[@]}" || status=$?

if [[ ! -f "$WORK_DIR/report.json" ]]; then
  echo "[ERR] quick pipeline failed before report.json was written (status=$status)" >&2
  exit "${status:-1}"
fi

if [[ "$WITH_QWEN_Q1" == "1" ]]; then
  q1_args=(
    --manifest "$WORK_DIR/asr_manifest.jsonl"
    --output "$WORK_DIR/asr_q1_context.jsonl"
    --model-dir "$ASR_MODEL_DIR" --device "$DEVICE"
    --batch-size "$BATCH_SIZE" --context-mode wake --cache-dir "$ASR_CACHE_DIR" --duration-bucket-sec "$DURATION_BUCKET_SEC"
  )
  python "${REPO_DIR}/scripts/score_asr_manifest.py" "${q1_args[@]}"
  python "${REPO_DIR}/scripts/audit_qwen_asr.py" \
    --manifest "$WORK_DIR/asr_manifest.jsonl" \
    --free "$WORK_DIR/asr.jsonl" --context "$WORK_DIR/asr_q1_context.jsonl" \
    --output "$WORK_DIR/qwen_asr_audit.json"
fi

python - "$WORK_DIR/report.json" "$EXPECTED_UIDS" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
report = json.load(open(path, encoding="utf-8"))
local = report["phases"]["I4_I5_local"]
sep = report["phases"]["I1_inventory"]["audio_cache"]
se = report["phases"]["I2_se"]
assert local["n_uid"] == expected, local
assert report["production_approved"] is False or report["status"] == "PASS"
print("[SEP_REUSE] hit/miss/fresh", sep["n_cache_hit"], sep["n_cache_miss"], sep["n_fresh"], sep["root"])
print("[SE_REUSE] hit/miss/fresh", se.get("n_cache_hit"), se.get("n_cache_miss"), se.get("n_fresh"), se.get("cache_root"))
print("[OK]", report["status"], "uid", local["n_uid"], "flat", report["paths"]["flat_dir"])
PY

echo "[DONE] $WORK_DIR/report.md"
exit "$status"
