# Quick 项目中英文 CTC 与 q_kw 落地技术方案

> 版本：v1.0  
> 日期：2026-08-29  
> 范围：用注册文本对 s1/s7、raw/SE 候选计算真正的文本条件声学分，并校准为 `q_kw`  
> ASR 主链见《ASR落地技术方案》，音频裁剪见《注册音频裁剪落地技术方案》。

## 实现基线状态（2026-08-29）

已落地可复用的基线是：`quick.ctc_align.forced_align_subsequence`（模型输出
`log_probs` 后的 CTC Viterbi 子串对齐）、`scripts/score_ctc_logprobs_manifest.py`
（`.npy/.npz` logits sidecar、去重和缓存）、现有中英文独立的一维 Platt
校准器，以及 `scripts/reselect_route.py`（所有 sidecar 完成后只重算评分/选路/导出，
不重复分离、SE 或 ASR）。该基线不伪造 acoustic model logits；MMS-FA、Zipformer
和 FastConformer 的 logits 提取仍需外部模型适配器，WeNet 自由解码仍只能作为
未校准对照，不能标记为真正 CTC。

## 1. 结论

停止把：

```text
ctc_align_score = 1 - 自由转写 CER
```

称为 CTC 对齐分。该值只是第二套 ASR 文本证据，不是注册文本在音频中的 CTC 路径概率，也不是 `q_kw`。

新方案分三层：

```text
T0：MMS-FA 中英统一 forced-alignment 基线
T1：中文 Zipformer CR-CTC + 自有 CTC DP
T2：英文 FastConformer CTC + 自有 CTC DP
T3：中英文分别校准，统一输出 q_kw=P(目标词真实存在|音频,文本,语言)
```

MMS-FA 用于快速统一实现和独立对照；效果候选使用中文、英文各自更强的声学模型。路由层只消费统一 schema，不感知模型内部框架。

## 2. 目标与非目标

### 2.1 目标

1. 对已知中英文注册文本计算真正的 frame/token forced-alignment 特征。
2. 对 `hicolmo` 等品牌词支持正发音 alias 和混淆词 margin。
3. 用独立正负开发集把原始特征校准成可解释的 `q_kw`。
4. 与 Qwen context 解耦，成为独立防幻觉证据。
5. 支持所有音频长度、raw/SE、噪声、音乐和竞争人声。
6. 输出可冻结、可复算、可去重的 sidecar。

### 2.2 非目标

1. CTC 不取代主 CER；当前路由仍以低 CER 为第一硬指标。
2. 原始 path log-prob 不能直接称为概率。
3. 强制对齐成功不能单独证明目标存在。
4. 不把旧 WeNet checkpoint 强行加载到不匹配的新 WeNet 代码。

## 3. 当前问题

### 3.1 旧 WeNet checkpoint/runtime 漂移

当前 WeNet 方案使用旧 AISHELL U2++ checkpoint 和新版 `recognize.py`，已经出现 missing/unexpected tensor。除非拿到原训练 commit、配置和词表并冻结整个运行时，否则该路径不具备可复现性。

### 3.2 当前分数不是 forced alignment

`scripts/score_wenet_ctc_manifest.py` 当前流程：

```text
WeNet自由解码 -> hyp
CER(hyp, wake_text)
1-CER -> ctc_align_score
```

它与主 ASR CER 高度相关，无法提供：

```text
逐token概率
目标区间
最弱token
blank比例
目标与混淆词margin
路径概率
```

### 3.3 单模型覆盖中英文不现实

中文强模型和英文强模型的 tokenizer、训练数据、声学分布不同。强行使用一个旧中文 WeNet 覆盖英文会造成系统缺口。

新方案允许模型分语言，但要求最终特征和 `q_kw` 契约统一。

## 4. 模型选择

### 4.1 T0：MMS-FA

用途：

1. 中英文统一的第一版 forced alignment；
2. 对中文/英文效果模型提供独立对照；
3. 为裁剪输出 token span 和 token score。

中文需要冻结的拼音/罗马化；英文直接规范化。TorchAudio 运行时必须固定版本，必要时把 emission 和 DP 逻辑封装在独立环境。

### 4.2 T1：中文 Zipformer CR-CTC

选择 2025 年 AISHELL Zipformer CR-CTC 预训练模型作为中文效果分支。目标不是复用 icefall 自由解码脚本，而是：

```text
冻结模型/词表/前端
导出或封装 frame log-prob
quick 自己执行 CTC forced-alignment DP
```

这样模型仓库更新不会改变 scoring contract。

### 4.3 T2：英文 FastConformer CTC

选择 NeMo FastConformer CTC 或 Hybrid CTC-Transducer 的 CTC 分支，提取英文 frame log-prob。NeMo Forced Aligner 可作参考，但正式输出仍应进入 quick 统一特征 schema。

### 4.4 Qwen3-ForcedAligner 的位置

Qwen3-ForcedAligner 输出时间戳但官方接口不输出路径置信分，因此：

```text
适合裁剪
不适合作为唯一 q_kw 原始分
```

它不替代 T0/T1/T2。

## 5. 文本、token 与 alias

### 5.1 中文

根据模型 tokenizer 生成：

```text
字符序列
无调拼音序列
模型需要时的罗马字序列
```

多音字由冻结词典覆盖。中文主 CER 继续使用无调拼音；CTC 模型内部 token 可以不同，但必须在输出中保存实际 token 序列和 normalizer hash。

### 5.2 英文

保存：

```text
标准拼写
positive aliases
发音变体
confusables
```

`hicolmo` 示例：

```text
positive: hicolmo, hi colmo, hey colmo, heycolmo
confusable: hi como, hey como
```

每个正 alias 独立计算路径分，取最好分；混淆词不能混入正 alias。

## 6. CTC forced-alignment 算法

### 6.1 输入

```text
log_probs: [T, V]
target tokens: y_1 ... y_U
blank_id
```

先构造插 blank 序列：

```text
[blank, y1, blank, y2, ..., yU, blank]
```

动态规划在时间轴上保留：

1. 留在当前 token；
2. 前进一个状态；
3. token 不重复时跨过一个 blank。

所有计算在 log-space 完成，避免长音频下溢。

### 6.2 部分文本搜索

注册词通常只占整段音频的一部分，不能要求目标覆盖整段音频。DP 必须允许：

```text
目标前自由 blank/非目标区域
目标后自由 blank/非目标区域
在全时间轴搜索最佳目标子区间
```

MMS `<star>` 可以吸收无关片段，但必须记录 `star_ratio`，避免大部分音频都被 wildcard 吸收仍得到高分。

### 6.3 多次出现

保留 top-K 非重叠路径：

```text
occurrence_start/end
path_score
token_scores
```

两个接近路径不能静默只留第一个；需输出 `multiple_occurrences` 供裁剪和审阅。

## 7. 原始特征

对每个正 alias 输出：

```text
align_logp_sum
align_logp_mean       = sum / target_token_count
align_logp_per_frame  = sum / aligned_frame_count
token_score_min
token_score_q10
token_score_mean
coverage
blank_ratio
star_ratio（若有）
aligned_duration_sec
duration_per_token
start_sec/end_sec
```

目标集合聚合：

```text
target_score = max_a score(alias_a)
best_alias = argmax_a score(alias_a)
```

混淆 margin：

```text
confusable_score = max_c score(confusable_c)
path_margin = target_score - confusable_score
```

还需加入音频级诊断：

```text
speech_ratio
p_music
p_overlap
snr_vad_db
clip_rate
duration_sec
```

## 8. 为什么不能直接阈值化 path score

强制对齐器会在任意音频中寻找“最不差”的目标路径。分数受以下因素影响：

```text
模型
语言
tokenizer
目标长度
音频长度
说话速度
噪声和音乐
blank先验
```

所以：

```text
align_logp_mean > 某个拍脑袋阈值
```

不能直接用于生产。必须经过中英文独立标注集校准。

## 9. q_kw 定义与校准

### 9.1 定义

```text
q_kw = P(注册唤醒词真实存在 | 音频, 注册文本, 语言)
```

### 9.2 特征

第一版建议输入：

```text
align_logp_mean
token_score_q10
coverage
path_margin
blank_ratio
star_ratio
aligned_duration / expected_duration
Qwen target NLL
Qwen NLL margin
Qwen free alias CER
独立 ASR 音素 CER
speech_ratio
p_music
p_overlap
```

不能把 Qwen 带 context 的 CER 当作无偏声学特征；可以加入 `context_gain` 和 `context_only_hit` 作为幻觉风险特征。

### 9.3 校准器

当前一维 Platt 工具适合单一 score 初版。多特征正式版建议：

1. 正则 Logistic Regression 作为可解释基线；
2. Isotonic 只用于单调二次校准且需要足够数据；
3. 若使用 GBDT，必须额外检查过拟合、语言分层和可靠性。

中文、英文分别训练：

```text
calibrator_zh
calibrator_en
```

两者最终都输出 `[0,1]` 概率，但阈值和 switch margin 分语言冻结。

### 9.4 标注集

正样本：

```text
目标词干净语音
口音、快慢、弱音
raw/SE
不同 s1/s7 流
音乐、噪声、竞争人声中仍真实包含目标
```

负样本：

```text
静音和纯噪声
音乐
其他人声
同音/近音/少字/多字
其他唤醒词
目标词只出现在 context、音频不存在的样本
分离出的干扰说话人
```

必须按 UID/说话人/原始音频分组切分，避免同源 raw、SE 或 s1/s7 泄漏到训练和验证两侧。

## 10. q_kw 验收指标

按中文、英文分别报告：

```text
AUROC
PR-AUC
EER
FRR@固定FAR
FAR@固定FRR
Brier score
ECE
可靠性曲线
bootstrap置信区间
```

同时分层：

```text
raw / SE
s1 / s7
噪声 / 音乐 / 重叠人声
短 / 长音频
目标长度
品牌词 / 普通词
```

只看平均 CER 或 AUROC 不足以批准绝对阈值。

## 11. Sidecar 契约

```json
{
  "schema": "quick_ctc_qkw/v2",
  "score_key": "...",
  "candidate_id": "...",
  "pcm_sha256": "...",
  "wake_text": "hicolmo",
  "lang": "en",
  "backend": "fastconformer_ctc_en",
  "best_alias": "hi colmo",
  "tokens": ["h", "i", "c", "o", "l", "m", "o"],
  "align_logp_mean": -0.42,
  "token_score_q10": -1.31,
  "coverage": 1.0,
  "blank_ratio": 0.64,
  "star_ratio": null,
  "path_margin": 0.38,
  "start_sec": 0.41,
  "end_sec": 1.28,
  "occurrence_count": 1,
  "q_kw": 0.93,
  "qkw_calibrated": true,
  "qkw_calibrator_hash": "...",
  "model_hash": "...",
  "runtime_hash": "...",
  "normalizer_hash": "...",
  "alias_hash": "..."
}
```

未校准时：

```text
q_kw = null
qkw_calibrated = false
score_kind = ctc_forced_alignment_uncalibrated
```

禁止把原始分复制到 `q_kw` 字段。

## 12. 去重与缓存

```text
ctc_key = SHA256(
  canonical_pcm_sha256,
  wake_text,
  lang,
  alias_hash,
  confusable_hash,
  frontend_hash,
  model_hash,
  runtime_hash,
  dp_policy_hash
)
```

相同 `ctc_key` 只计算一次。candidate_id、arm、raw/SE 路径仅作为引用映射。

若同一 `ctc_key` 出现不同结果，属于 fatal evidence conflict，写入：

```text
same_ctc_key_conflicts.jsonl
```

## 13. 长音频

不得把音频硬裁到 3 秒后再算 CTC。

优先顺序：

1. 声学模型支持时整段提取 logits；
2. 超长时用模型原生 streaming/chunked encoder；
3. 不支持流式时使用重叠窗口，全时间轴覆盖；
4. 每个窗口独立 forced alignment，映射回全局时间；
5. 保留跨窗口边界保护和 top-K occurrence。

窗口大小由模型上下文和注册词最长预期时长决定，不使用与业务无关的固定 3 秒上限。

## 14. 与当前选路的结合

保持主规则：

```text
L1 最低主 ASR CER
L2 calibrated q_kw
L3 未校准时仅同 UID、同 CER 比较 CTC/NLL margin
L4 speaker_ref_score
L5 噪声、音乐、重叠和质量
L6 确定性保守顺序
```

跨 s1/s7：

```text
CER_s7 < CER_s1 -> 可切 s7
CER_s7 > CER_s1 -> 保持 s1
CER相同 -> q_kw达到冻结margin才切
```

未校准 CTC 不得启用 `qkw_low_thr`，也不能跨 UID 解释为存在概率。

完成校准后，低 `q_kw` 是否触发 s7 仍需独立验证，默认保持关闭。

## 15. 建议新增文件

```text
configs/ctc_backends.json
configs/ctc_dp_policy.json
configs/wake_aliases.json
scripts/download_ctc_models.sh
scripts/score_mms_fa_manifest.py
scripts/score_zipformer_ctc_manifest.py
scripts/score_fastconformer_ctc_manifest.py
scripts/build_qkw_dataset.py
scripts/train_qkw_calibrator.py
scripts/apply_qkw_calibrator_v2.py
scripts/audit_ctc_qkw.py
src/quick/ctc_align.py
src/quick/ctc_contract.py
src/quick/qkw_features.py
src/quick/qkw_calibration_v2.py
tests/test_ctc_dp.py
tests/test_ctc_contract.py
tests/test_qkw_split_leakage.py
```

现有 `calibrate_qkw.py` 和 `apply_qkw_calibrator.py` 保留为一维基线，不静默改变其 schema。

## 16. 分阶段落地

### T0：MMS-FA shadow

全量输出中英文 span 和特征，不参与选路，验证 coverage、缓存和长音频。

### T1：中文 Zipformer CR-CTC

先导出/封装稳定 logits 接口，再复用相同 DP 和 schema。

### T2：英文 FastConformer CTC

同 T1，禁止英文继续依赖中文 WeNet。

### T3：单分数校准基线

先使用 `align_logp_mean + path_margin` 构造简单 score，与现有一维 Platt 流程对接。

### T4：多特征 q_kw

建立按源分组的标注集，训练中英文多特征校准器。

### T5：shadow route

比较：

```text
CER-only
CER + raw CTC
CER + calibrated q_kw
CER + q_kw + NLL
CER + q_kw + crop
```

### T6：冻结阈值

基于独立开发集冻结：

```text
qkw_switch_margin_zh/en
qkw_low_thr_zh/en（若证据足够，否则保持null）
calibrator hash
model/runtime/frontend/alias/DP hash
```

## 17. 预期命令接口

```bash
python scripts/score_mms_fa_manifest.py \
  --manifest "$WORK_DIR/candidate_refs_with_se.jsonl" \
  --output "$WORK_DIR/ctc_mms_fa.jsonl" \
  --aliases configs/wake_aliases.json \
  --device cuda:0

python scripts/score_zipformer_ctc_manifest.py \
  --manifest "$WORK_DIR/candidate_refs_with_se.jsonl" \
  --output "$WORK_DIR/ctc_zh.jsonl" \
  --model-dir /root/autodl-tmp/quick_models/ctc/zipformer_cr_ctc_zh \
  --enabled-langs zh \
  --device cuda:0

python scripts/score_fastconformer_ctc_manifest.py \
  --manifest "$WORK_DIR/candidate_refs_with_se.jsonl" \
  --output "$WORK_DIR/ctc_en.jsonl" \
  --model-dir /root/autodl-tmp/quick_models/ctc/fastconformer_ctc_en \
  --enabled-langs en \
  --device cuda:0

python scripts/train_qkw_calibrator.py \
  --labeled-jsonl /root/autodl-tmp/qkw_dev/labeled_features.jsonl \
  --output /root/autodl-tmp/qkw_dev/qkw_calibrator_v2.json \
  --group-field source_uid \
  --languages zh,en
```

其中 `score_ctc_logprobs_manifest.py` 和 `calibrate_qkw.py` 已可运行；
MMS-FA、Zipformer、FastConformer 的模型提取命令仍是外部适配目标，未安装对应
模型时不得把 WeNet 自由解码结果冒充为这些模型的 forced alignment。

正式 route 只接受合并且完整的 `qkw.jsonl`：

```bash
QKW_CALIBRATED=1 \
QKW_JSONL=/root/autodl-tmp/qkw_dev/qkw_full.jsonl \
QKW_CALIBRATOR_JSON=/root/autodl-tmp/qkw_dev/qkw_calibrator_v2.json \
bash scripts/run_a3_route.sh
```

若 ASR、CTC、q_kw、声纹和噪声结果是分阶段生成的，全部完成后可只重新选路：

```bash
python scripts/reselect_route.py \
  --work-dir /root/autodl-tmp/quick_a3_route \
  --output-dir /root/autodl-tmp/quick_a3_reselect_v1 \
  --qkw-jsonl /root/autodl-tmp/quick_a3_route/qkw_calibrated.jsonl
```

该命令读取既有 `candidate_refs_with_se.jsonl`，重新计算 CER/q_kw/NLL/质量排序，
并生成新的 `review_flat`、`best_sep_selected`、`route_decisions.jsonl` 和
`report.json`；源工作目录不被覆盖。

## 18. 验收标准

### 18.1 单元和契约

1. 人工小矩阵可验证 DP 路径和 blank/repeat 规则；
2. 相同输入重复运行分数一致；
3. 中英文每个非 fatal score_key 全覆盖；
4. 任何缺失、非有限、hash 冲突都会失败；
5. 未校准输出不含伪 `q_kw`；
6. alias/confusable 和 normalizer hash 完整。

### 18.2 效果

1. MMS、中文 CTC、英文 CTC 分语言报告；
2. `hicolmo` 正样本和近音负样本必须单列；
3. q_kw 在独立集满足预先冻结的 EER、FRR@FAR、PR-AUC、ECE/Brier 门；
4. 相对 CER-only route 的每 UID 退化为零或进入明确人工批准清单；
5. q_kw 引发的 s1→s7 切换必须逐条有正收益证据；
6. 裁剪后重新校准或至少验证分布不漂移。

### 18.3 生产

即使 local q_kw 通过，仍需：

```text
full UID coverage
CMD FRR/FAR
Presence
contest/预期测试集
模型和阈值冻结
selected index/hash绑定
```

在此之前：

```text
production_approved = false
```

## 19. 回退策略

1. 某语言效果 CTC 不可用：该语言回退 MMS-FA shadow 或无 q_kw，不用另一语言模型顶替；
2. 校准器 coverage 不完整：整个 calibrated q_kw 模式关闭；
3. raw CTC 分可用但校准失败：仅同 UID、同 CER tie-break；
4. 模型/runtime 更新：新 hash、新 work dir，旧证据保留；
5. CTC 与主 CER 强冲突：进入审阅，不允许 CTC 无条件翻盘；
6. 长音频窗口不完整：该 PCM 失败并保留 full/主 CER，不能静默截断。

## 20. 主要外部依据

- MMS 多语言 forced alignment：https://docs.pytorch.org/audio/2.7.0/tutorials/forced_alignment_for_multilingual_data_tutorial.html
- TorchAudio frame/token forced alignment：https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
- icefall Zipformer CR-CTC AISHELL 结果：https://github.com/k2-fsa/icefall/blob/master/egs/aishell/ASR/RESULTS.md
- NeMo Forced Aligner：https://docs.nvidia.com/nemo/speech/nightly/tools/nemo_forced_aligner.html
- Qwen3-ASR / ForcedAligner：https://github.com/QwenLM/Qwen3-ASR
