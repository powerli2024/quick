# quick：s1/s7 + MossFormer SE 严格选路（AutoDL Linux）

`quick` 按 [`docs/S1_S7选路与平铺审阅导出方案.md`](docs/S1_S7选路与平铺审阅导出方案.md) 落地 I0–I8。
当前代码逐条执行逻辑、两种 q_kw 模式和合理性审查见 [`docs/当前选路逻辑与合理性审查.md`](docs/当前选路逻辑与合理性审查.md)。
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
export AUDIO_CACHE_ROOT=/root/autodl-tmp/quick_audio_cache
# 可选：有稳定权重 hash 时再设置；不设置也不阻断运行/I8
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
也可以换新的 `WORK_DIR`：只要 `AUDIO_CACHE_ROOT` 和 SE 模型/推理签名不变，音频仍会命中固定缓存。

## 固定音频缓存契约

正式脚本默认使用 `/root/autodl-tmp/quick_audio_cache`，也可显式设置 `AUDIO_CACHE_ROOT`。目录不再依赖某一轮 `WORK_DIR`：

```text
$AUDIO_CACHE_ROOT/
├── sep_pcm/<pcm前2位>/<pcm_sha256>.wav
└── se48k/<SE变换签名>/<raw_pcm前2位>/<raw_pcm_sha256>.wav
```

- `sep_pcm` 是 s1/s7 分离结果的内容寻址副本；原始 extract-sep 文件不被修改。
- `se48k` 由 backend、命令、推理签名和全长输出契约隔离；`MOSSFORMER_MODEL_HASH` 仅作为可选 provenance 记录，不再作为 I8 硬门槛。更换 SE 模型时仍应使用新的 `INFERENCE_SIGNATURE` 或新的缓存目录。
- 每轮在 `sep_audio_cache_manifest.jsonl`、`inventory_meta.json`、`se_meta.json` 和 `report.json` 记录 `hit/miss/fresh`。
- 命中项仍校验 PCM、可解码性和 SE 前后时长；文件存在不等于自动信任。
- `WORK_DIR` 只保存本轮清单、sidecar、评分、选路和导出；`input_signatures.json` 用于安全续跑，`signatures.json` 绑定最终评分证据。

正式门还要求：`n_selected_finite_cer = n_baseline_finite_cer = n_paired = expected_uid`，且禁止 audit_fallback / missing ASR。策略默认读 `configs/route_policy.json`，可用 `--policy-json` 覆盖。

## q_kw 的正式契约

`q_kw` 是“目标唤醒词存在”的校准概率，不等于 `1-CER`、CTC 对齐分、ASR 置信度或 `-NLL`。正式启用时必须同时满足：

1. 校准数据来自与最终验收集隔离的开发集，标签固定为 `1=目标唤醒词存在`、`0=不存在`；中文和英文分别拟合。
2. 对本轮每个 unique `(pcm_sha256, wake_text, lang)` 都有一条记录，不能只覆盖容易样本。
3. 每条记录包含 `score_kind=calibrated_qkw`、`0<=q_kw<=1`，以及相同的 `qkw_calibrator_hash`。
4. 运行时给出的校准器 hash 必须逐条匹配；任一缺失、越界或签名不一致直接终止。
5. 未启用 `--qkw-calibrated` 时，即使 sidecar 自称带 `q_kw`，该值也不会进入选路。未校准 CTC/NLL 只允许在同 UID、同 CER 下按冻结 margin 破同分，不能使用跨 UID 绝对阈值。

拟合和应用冻结校准器：

```bash
python scripts/calibrate_qkw.py \
  --input /root/autodl-tmp/qkw_dev_labeled.jsonl \
  --output /root/autodl-tmp/qkw_calibrator.json \
  --score-field ctc_align_score

python scripts/apply_qkw_calibrator.py \
  --input /root/autodl-tmp/wenet_ctc_scores.jsonl \
  --calibrator /root/autodl-tmp/qkw_calibrator.json \
  --output /root/autodl-tmp/qkw_calibrated.jsonl

python scripts/run_s1_s7_route.py ... \
  --qkw-jsonl /root/autodl-tmp/qkw_calibrated.jsonl \
  --qkw-calibrated \
  --qkw-calibrator-json /root/autodl-tmp/qkw_calibrator.json
```

如果原始分数是 NLL，拟合时增加 `--score-field nll --lower-is-better`。`report.json` 的 `I3_score` 必须显示 `qkw_valid=qkw_expected` 且 `qkw_coverage=1.0`，否则不可使用 q_kw 结论。

## 输入约定

`--pos-neg` 与 kws 相同：`pos/`、`neg/`，每个 arm 下有 `index.jsonl` 和 `wav/`。
index 至少要有 `uid`、`wake_text`（兼容 `唤醒文本`/`wake`/`text`）、`streams`；缺少 `lang` 时按
唤醒文本推导。`original` 在 WAV 文件名中可对应 `peak`。s7 某 UID 缺失记为 `s7_available=false`。

## 阶段产物

| 阶段 | 产物 |
|---|---|
| I0 | `signatures.json` |
| I1 | `file_sha256` + `pcm_sha256` registry，并物化固定 `sep_pcm` 缓存 |
| I2 | unique raw 一次 SE，写入模型签名隔离的固定 `se48k` 缓存；失败写 8000 占位 JSON |
| I3 | unique `(pcm, wake, lang)` 一次 ASR/NLL |
| I4–I5 | 冻结 `reason_code` 与 `decision_trace` |
| I6 | `review_flat/`：`0000__SELECTED` 在组内第一条 |
| I7 | 从 0000 物化 selected-only，不再选路 |
| I8 | CMD FRR/FAR + Presence + contest |

平铺目录只用于听审，不能直接冒充 extract-main 的 selected-only `best_sep`。
# A3：SenseVoice + WeNet CTC 部署

quick 的 A3 主链使用 SenseVoiceSmall 做无提示自由转写，避免 Qwen3 的
wake/context 提示放大幻觉；WeNet CTC 作为独立已知唤醒词证据。CTC 适配器输出
`ctc_align_score=1-cer_route`，只在同一 UID 的 CER 并列时作为 tie-breaker，
不是跨录音概率；拿到开发集校准器后再写入 `q_kw/qkw_calibrated`。

在 AutoDL（`/root/quick` 为本仓库）首次部署：

```bash
cd /root/quick
pip install -r requirements-a3.txt
bash scripts/download_a3_models.sh
```

随后复用旧分离与 MossFormer-48k 音频。`POS_NEG` 指向已有 extract-sep
full-audio 根目录，`PRECOMPUTED_SE_DIR` 指向旧 SE 根目录；quick 会按
`pcm_sha256` 建索引，支持 `se_wav/<hash[:2]>/<hash>.wav`、扁平 hash 以及
更深层目录，不重复处理已存在音频：

```bash
POS_NEG=/root/autodl-tmp/kws_sep_fullaudio_v1 \
PRECOMPUTED_SE_DIR=/root/autodl-tmp/kws_se_route/se_wav \
WENET_DECODE_COMMAND='python /root/your_wenet_decode.py --manifest {manifest} --output {output} --model-dir {model_dir}' \
bash scripts/run_a3_route.sh
```

默认调用 quick 自带的 `wenet_decode_manifest.py`（需要 `/root/wenet` 源码树和
下载的 `train.yaml/final.pt`）；如使用其他解码器，`WENET_DECODE_COMMAND`
必须产生 JSONL（`score_key`+`hyp`）或 Kaldi 风格 `key text` 文件。也可以直接给已有 CTC 转写：

若 WeNet 权重或源码尚未就绪，`run_a3_route.sh` 会输出 warning 并继续完成
SenseVoice + s1/s7/SE 选路，不会丢弃已生成的 ASR 缓存；安装完成后重新运行即可补上 CTC tie-breaker。
已知 checkpoint 配置不兼容时可设置 `DISABLE_CTC=1`，强制只跑 SenseVoice 主链。

若 SenseVoice 已经跑完但后续阶段失败，可直接复用已有 sidecar，避免重新识别：

```bash
ASR_JSONL=/root/autodl-tmp/kws_a3_route_v2/asr.jsonl \
QKW_JSONL=/root/autodl-tmp/empty_qkw.jsonl \
WORK_DIR=/root/autodl-tmp/kws_a3_route_v3 \
bash scripts/run_a3_route.sh
```

```bash
QKW_JSONL=/root/autodl-tmp/wenet_decode.jsonl bash scripts/run_a3_route.sh
```

如果旧 SE 缓存不存在，改为设置 `SE_BATCH_COMMAND`，其命令需接受
`{manifest}` 并为每行写出声明的 `output`。最终证据位于
`$WORK_DIR/report.json`、`scored_candidates.jsonl` 和 `review_flat/`；严格
生产验收仍需独立 CMD/Presence/contest 结果，不能用 spectral 或合成结果放行。
