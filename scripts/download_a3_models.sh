#!/usr/bin/env bash
set -euo pipefail

# Download A3 weights without placing them in the git checkout.  SenseVoice is
# fetched from its official ModelScope repository; the default WeNet checkpoint
# is the official AISHELL U2++ Conformer package listed by WeNet.  Override the
# URLs/paths for an internal mirror without changing the pipeline.
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/quick_models/a3}"
SENSEVOICE_DIR="${SENSEVOICE_DIR:-$MODEL_ROOT/SenseVoiceSmall}"
WENET_DIR="${WENET_DIR:-$MODEL_ROOT/wenet_aishell_u2pp_conformer_exp}"
SENSEVOICE_ID="${SENSEVOICE_ID:-iic/SenseVoiceSmall}"
WENET_URL="${WENET_URL:-https://wenet.org.cn/downloads?models=wenet&version=aishell_u2pp_conformer_exp.tar.gz}"
WENET_REPO="${WENET_REPO:-/root/wenet}"
mkdir -p "$MODEL_ROOT"

if [[ ! -f "$SENSEVOICE_DIR/model.pt" && ! -f "$SENSEVOICE_DIR/config.yaml" ]]; then
  python - "$SENSEVOICE_ID" "$SENSEVOICE_DIR" <<'PY'
import sys
model_id, dest = sys.argv[1:]
try:
    from modelscope import snapshot_download
except Exception as exc:
    raise SystemExit("Install requirements-a3.txt before downloading SenseVoice") from exc
snapshot = snapshot_download(model_id, local_dir=dest)
print(f"SenseVoice downloaded to {snapshot}")
PY
else
  echo "[A3][reuse] SenseVoice: $SENSEVOICE_DIR"
fi

if [[ ! -d "$WENET_DIR" || -z "$(find "$WENET_DIR" -maxdepth 1 -type f 2>/dev/null | head -n 1)" ]]; then
  tmp="$MODEL_ROOT/wenet_aishell.tar.gz.part"
  echo "[A3] downloading WeNet AISHELL checkpoint"
  wget -c "$WENET_URL" -O "$tmp"
  mkdir -p "$WENET_DIR"
  tar -xzf "$tmp" -C "$WENET_DIR" --strip-components=1
  rm -f "$tmp"
else
  echo "[A3][reuse] WeNet checkpoint: $WENET_DIR"
fi

if [[ ! -f "$WENET_REPO/wenet/bin/recognize.py" ]]; then
  echo "[A3] cloning WeNet runtime source into $WENET_REPO"
  git clone --depth 1 https://github.com/wenet-e2e/wenet.git "$WENET_REPO"
else
  echo "[A3][reuse] WeNet source: $WENET_REPO"
fi
if [[ "${INSTALL_WENET:-1}" == "1" ]]; then
  if ! python -c 'import wenet' >/dev/null 2>&1; then
    python -m pip install -e "$WENET_REPO"
  else
    echo "[A3][reuse] WeNet Python package"
  fi
fi

cat <<EOF
[A3] ready
  SENSEVOICE_DIR=$SENSEVOICE_DIR
  WENET_DIR=$WENET_DIR
  WENET_REPO=$WENET_REPO
  Next: set WENET_DECODE_COMMAND to a WeNet CTC decoder that writes {output}.
EOF
