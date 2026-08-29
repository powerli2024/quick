#!/usr/bin/env bash
set -euo pipefail

# One-command A3 deployment.  Raw s1/s7 files are read from POS_NEG; previous
# MossFormer-48k files are reused by hash when PRECOMPUTED_SE_DIR is present.
QUICK_DIR="${QUICK_DIR:-/root/quick}"
POS_NEG="${POS_NEG:?set POS_NEG to the extract-sep full-audio root}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/kws_a3_route}"
AUDIO_CACHE_ROOT="${AUDIO_CACHE_ROOT:-/root/autodl-tmp/quick_audio_cache}"
S1_ARM="${S1_ARM:-s1_onnx_full}"
S7_ARM="${S7_ARM:-s7_cv_then_onnx_gate/thr_a}"
EXPECTED_UIDS="${EXPECTED_UIDS:-0}"
DEVICE="${DEVICE:-cuda:0}"
ASR_BACKEND="${ASR_BACKEND:-sensevoice}"
SENSEVOICE_DIR="${SENSEVOICE_DIR:-/root/autodl-tmp/quick_models/a3/SenseVoiceSmall}"
PARAFORMER_ZH_DIR="${PARAFORMER_ZH_DIR:-/root/autodl-tmp/quick_models/a3/Paraformer-zh}"
PARAFORMER_EN_DIR="${PARAFORMER_EN_DIR:-/root/autodl-tmp/quick_models/a3/Paraformer-en}"
WENET_DIR="${WENET_DIR:-/root/autodl-tmp/quick_models/a3/wenet_aishell_u2pp_conformer_exp}"
PRECOMPUTED_SE_DIR="${PRECOMPUTED_SE_DIR:-/root/autodl-tmp/kws_se_route/se_wav}"

if [[ "$ASR_BACKEND" == "sensevoice" ]]; then
  if [[ ! -d "$SENSEVOICE_DIR" ]]; then
    echo "missing SenseVoice model: $SENSEVOICE_DIR (run scripts/download_a3_models.sh)" >&2
    exit 2
  fi
elif [[ "$ASR_BACKEND" == "paraformer" ]]; then
  if [[ ! -d "$PARAFORMER_ZH_DIR" || ! -d "$PARAFORMER_EN_DIR" ]]; then
    echo "missing Paraformer zh/en model dirs: $PARAFORMER_ZH_DIR / $PARAFORMER_EN_DIR" >&2
    echo "run scripts/download_a3_models.sh or set PARAFORMER_ZH_DIR/PARAFORMER_EN_DIR" >&2
    exit 2
  fi
else
  echo "unsupported ASR_BACKEND=$ASR_BACKEND (choose sensevoice or paraformer)" >&2
  exit 2
fi

if [[ -z "${ASR_COMMAND:-}" || "$ASR_COMMAND" != *"{manifest}"* || "$ASR_COMMAND" != *"{output}"* ]]; then
  # Do not use ${VAR:-...} here: Bash treats braces in the default value as
  # parameter-expansion syntax and turns {manifest} into {manifest.
  if [[ "$ASR_BACKEND" == "paraformer" ]]; then
    ASR_COMMAND="python ${QUICK_DIR}/scripts/score_paraformer_manifest.py --manifest {manifest} --output {output} --zh-model-dir ${PARAFORMER_ZH_DIR} --en-model-dir ${PARAFORMER_EN_DIR} --device ${DEVICE}"
  else
    ASR_COMMAND="python ${QUICK_DIR}/scripts/score_sensevoice_manifest.py --manifest {manifest} --output {output} --model-dir ${SENSEVOICE_DIR} --device ${DEVICE}"
  fi
fi
if [[ "$ASR_BACKEND" == "paraformer" ]]; then
  ASR_MODEL_DIR="$PARAFORMER_ZH_DIR"
  if [[ -z "${ASR_MODEL_HASH:-}" ]]; then
    ASR_MODEL_HASH="$(PYTHONPATH="${QUICK_DIR}/src" python - "$PARAFORMER_ZH_DIR" "$PARAFORMER_EN_DIR" <<'PY'
import hashlib, sys
from quick.signatures import hash_model_dir
h = hashlib.sha256()
for path in sys.argv[1:]:
    h.update(path.encode())
    h.update(b"\0")
    h.update((hash_model_dir(path) or "missing").encode())
    h.update(b"\n")
print(h.hexdigest())
PY
)"
  fi
else
  ASR_MODEL_DIR="$SENSEVOICE_DIR"
fi

args=(
  "$QUICK_DIR/scripts/run_s1_s7_route.py"
  --pos-neg "$POS_NEG" --s1-arm "$S1_ARM" --s7-arm "$S7_ARM"
  --expected-uids "$EXPECTED_UIDS" --work-dir "$WORK_DIR"
  --audio-cache-root "$AUDIO_CACHE_ROOT"
  --asr-model-dir "$ASR_MODEL_DIR"
  --asr-context-mode none --qkw-switch-margin "${QKW_SWITCH_MARGIN:-0.01}"
  --alias-json "${ALIAS_JSON:-$QUICK_DIR/configs/english_alias.json}"
  --policy-json "${POLICY_JSON:-$QUICK_DIR/configs/route_policy.json}"
  --inference-signature "${INFERENCE_SIGNATURE:-a3_${ASR_BACKEND}_ctc_optional}"
  --selected-only-dir "${SELECTED_ONLY_DIR:-$WORK_DIR/best_sep_selected}"
)
if [[ -n "${ASR_MODEL_HASH:-}" ]]; then
  args+=(--asr-model-hash "$ASR_MODEL_HASH")
fi

if [[ -n "${ASR_JSONL:-}" && -f "$ASR_JSONL" ]]; then
  echo "[A3][reuse] $ASR_BACKEND sidecar: $ASR_JSONL" >&2
  args+=(--asr-jsonl "$ASR_JSONL")
else
  args+=(--asr-command "$ASR_COMMAND")
fi

if [[ "${ENABLE_CTC:-0}" != "1" || "${DISABLE_CTC:-0}" == "1" ]]; then
  echo "[A3][INFO] WeNet CTC disabled by default; continue with $ASR_BACKEND (set ENABLE_CTC=1 only with a matched model/repo)" >&2
elif [[ -n "${QKW_JSONL:-}" ]]; then
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
    echo "[A3][WARN] WeNet CTC unavailable; continue with $ASR_BACKEND only (set QKW_JSONL or install a matched model/source to enable CTC)" >&2
  fi
fi

if [[ "${QKW_CALIBRATED:-0}" == "1" ]]; then
  if [[ -z "${QKW_JSONL:-}" || ! -f "$QKW_JSONL" || -z "${QKW_CALIBRATOR_JSON:-}" || ! -f "$QKW_CALIBRATOR_JSON" ]]; then
    echo "[A3][ERR] QKW_CALIBRATED=1 requires complete QKW_JSONL and QKW_CALIBRATOR_JSON" >&2
    exit 2
  fi
  args+=(--qkw-calibrated --qkw-calibrator-json "$QKW_CALIBRATOR_JSON")
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
