# quick：s1/s7 + MossFormer SE 严格选路（AutoDL Linux）

`quick` 按 [`docs/S1_S7选路与平铺审阅导出方案.md`](docs/S1_S7选路与平铺审阅导出方案.md) 落地 I0–I8。
输入约定对齐 `kws` 的 extract-sep 目录；ASR / NLL / embedding / MossFormer 通过 sidecar 或
command 复用 `/root/kws` 与 `/root/extract-main`，不修改 kws。

目标环境是 **AutoDL Linux**（`/root/...`、`cuda:0`、bash）。`s7_arm=auto` 在冻结复验中禁止。

## 本地检查

```bash
cd /root/quick
python -m pytest -q
python scripts/run_s1_s7_route.py --help
```

## AutoDL 正式跑

冻结 s7 arm 后：

```bash
cd /root/quick
export POS_NEG=/root/autodl-tmp/kws_sep
export WORK_DIR=/root/autodl-tmp/quick_s1_s7
export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
export S7_ARM=s7_cv_then_onnx_gate/thr_a   # 开发集锁定的精确标签
export KWS_DIR=/root/kws
export EXTRACT_MAIN=/root/extract-main
export CLEARVOICE_ROOT=/root/autodl-tmp/ClearerVoice-Studio
bash scripts/run_s1_s7_route.sh
```

默认会：

1. 用 kws 的 `extract_main_se48k_manifest.py` 对 unique raw PCM 跑 MossFormer2_SE_48K
2. 用 `scripts/score_asr_manifest.py` 调 kws `Qwen3ASRTranscriber`（按 pcm+wake+lang 只转写一次）
3. `WITH_NLL=1` 时写 target NLL sidecar（只破同分，不是校准 q_kw）
4. 导出 `review_flat/` 与 `best_sep_selected/`
5. `SE_BACKEND=spectral` 最多本地验收，**永远不会** `production_approved=true`
6. I8 `PASS` 要求生产 SE（command/precomputed）+ 带 schema/bindings/UID 指纹/逐语言指标的 CMD/Presence/contest JSON

摸底闭环（不跑神经 SE）可设 `SE_BACKEND=spectral`。中断后续跑保持同一 `WORK_DIR` 并 `RESUME=1`。

正式门还要求：`n_selected_finite_cer = n_baseline_finite_cer = n_paired = expected_uid`，且禁止 audit_fallback / missing ASR。`--qkw-calibrated` 必须同时提供 `--qkw-calibrator-hash`。策略默认读 `configs/route_policy.json`，可用 `--policy-json` 覆盖。

## 输入约定

`--pos-neg` 与 kws 相同：`pos/`、`neg/`，每个 arm 下有 `index.jsonl` 和 `wav/`。
index 至少要有 `uid`、`wake_text`（兼容 `唤醒文本`/`wake`/`text`）、`streams`；缺少 `lang` 时按
唤醒文本推导。`original` 在 WAV 文件名中可对应 `peak`。s7 某 UID 缺失记为 `s7_available=false`。

## 阶段产物

| 阶段 | 产物 |
|---|---|
| I0 | `signatures.json` |
| I1 | `file_sha256` + `pcm_sha256` registry |
| I2 | unique raw 一次 SE；失败写 8000 占位 JSON |
| I3 | unique `(pcm, wake, lang)` 一次 ASR/NLL |
| I4–I5 | 冻结 `reason_code` 与 `decision_trace` |
| I6 | `review_flat/`：`0000__SELECTED` 在组内第一条 |
| I7 | 从 0000 物化 selected-only，不再选路 |
| I8 | CMD FRR/FAR + Presence + contest |

平铺目录只用于听审，不能直接冒充 extract-main 的 selected-only `best_sep`。
