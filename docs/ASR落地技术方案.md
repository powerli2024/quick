# Quick 项目 ASR 落地技术方案

> 版本：v1.0  
> 日期：2026-08-29  
> 范围：`D:\gpt\quick` 的 s1/s7、raw/MossFormer-SE 注册音频选路  
> 本文只定义 ASR 层；注册音频裁剪见《注册音频裁剪落地技术方案》，CTC 与 `q_kw` 见《CTC与q_kw落地技术方案》。

## 1. 结论

主 ASR 恢复使用 `Qwen3-ASR-1.7B`，但不恢复原来的“把注册文本直接作为 context 后用单次结果计算硬 CER”的方法。

正式方案固定为：

```text
Q0：Qwen3-ASR 无提示自由转写        -> 主 CER
Q1：Qwen3-ASR 保守词表上下文转写    -> 品牌词召回和拼写辅助
Q2：Qwen3-ASR 注册文本 teacher-forced NLL -> 目标条件分
Q3：SenseVoice/Paraformer 无提示转写 -> 独立音素证据和对照
```

其中只有 Q0 的自由转写可以直接产生主 CER。Q1 含有注册文本信息，存在标签泄漏风险，不能单独把候选变成 CER0。Q2 在独立标注集校准前只允许同 UID 排序。Q3 不要求输出品牌标准拼写，主要判断发音是否相近。

## 2. 目标与非目标

### 2.1 目标

1. 中英文注册词统一全覆盖，支持 `hicolmo` 等品牌词。
2. 降低 Qwen3-ASR 在静音、噪声、回声、音乐和弱人声上的幻觉。
3. 保留 Qwen3-ASR 对品牌词和复杂声学环境的优势。
4. 所有结果可复算、可去重、可追踪模型与运行时签名。
5. 与现有 `quick` 的 CER 第一、s1 优先、s7 条件切换策略兼容。

### 2.2 非目标

1. ASR 不直接判断最终生产可用；仍需 CMD、Presence、FRR/FAR 等下游验证。
2. 带 context 的转写不能替代 `q_kw`。
3. SenseVoice/Paraformer 不承担品牌标准拼写裁判职责。
4. 本阶段不以 vLLM 吞吐为第一目标，先冻结 Transformers 基线效果。

## 3. 当前实现的关键问题

### 3.1 注册文本直接注入

历史版本的 `scripts/score_asr_manifest.py` 默认 `--context-mode wake`，`kws/asr_transcribe.py` 将整个 `wake_text` 直接传给 `context`。当前已改为 Q0 默认 `none`；Q1 只能通过显式 `--context-mode wake` 或双 sidecar 脚本生成。Qwen 官方实现会把 context 放入 system message；它不是传统、可量化权重的热词 WFST。

风险：弱语音、近音甚至噪声可能被补成注册词。因此必须拆成 Q0/Q1 两个独立 sidecar。

### 3.2 异长 batch

历史版本同一 `(wake_text, lang)` 下按最多 8 条音频直接批处理，不按时长分桶。当前 Qwen 路由默认 batch=1；显式增大 batch 时按 `duration_bucket_sec`（默认 0.5 秒）分桶，并将分桶参数写入 runtime hash。

效果基线必须使用：

```text
batch_size = 1
```

只有逐 PCM 验证 batch=1 与分桶 batch 完全一致后，才能启用批量加速。

### 3.3 48 kHz 到 16 kHz 的线性重采样

`D:\gpt\kws\src\kws\audio.py` 已改为优先 soxr HQ、其次 scipy polyphase；仅当两者均不可用时才使用带诊断标记的线性兜底。MossFormer 48 kHz 音频在下采样到 16 kHz 时因此具有抗混叠处理。

正式 ASR 输入统一改用以下任一高质量实现，并冻结版本：

1. `soxr`，质量档 `HQ`；
2. `scipy.signal.resample_poly`；
3. `torchaudio.transforms.Resample` 的 sinc 实现。

不得让 raw、SE、Q0、Q1、Q2 使用不同的重采样器。

### 3.4 context 静默降级

适配器已移除上述静默降级逻辑。不同版本不支持 `context` 时直接失败，要求固定并升级到兼容的 Qwen3-ASR 包。

新实现必须：

```text
请求 context 且运行时不支持 -> 明确失败
请求 none                    -> 明确传空 context
```

禁止静默删除或更名参数。

## 4. 输入音频规范

### 4.1 canonical ASR PCM

```text
sample_rate = 16000
channels = 1
dtype = float32
range = [-1, 1]
resampler = frozen high-quality sinc/polyphase
```

生成：

```text
asr_pcm_sha256 = SHA256(canonical_16k_float32_PCM)
```

`asr_pcm_sha256` 与源文件 SHA256 分开保存。相同 canonical PCM 即使 WAV 头、文件路径或候选别名不同，也只推理一次。

### 4.2 输入审计

每个 unique PCM 在 ASR 前记录：

```text
duration_sec
source_sample_rate
sample_count
peak
rms_dbfs
clip_rate
dc_offset
speech_ratio
silence_ratio
non_finite_count
```

非有限数值、空波形、采样率无法解析属于 fatal；低语音占比不是自动 fatal，但必须进入幻觉诊断。

## 5. 文本规范与 alias

### 5.1 中文

主 CER 继续使用无声调拼音 CER。保留原始汉字、规范汉字和人工确认的多音字读法：

```text
wake_text_raw
wake_text_normalized
pinyin_toneless
pinyin_with_tone_optional
```

多音字和品牌读法必须来自冻结配置，不允许运行时根据 ASR 输出自动扩充。

### 5.2 英文

英文同时计算：

1. 标准化字符 CER；
2. 人工确认 alias CER；
3. 独立模型音素 CER。

`hicolmo` 的配置示例：

```json
{
  "wake_text": "hicolmo",
  "lang": "en",
  "display": "Hi Colmo",
  "positive_aliases": ["hicolmo", "hi colmo", "hey colmo", "heycolmo"],
  "confusables": ["hi como", "hey como"]
}
```

`hey colmo` 是否属于正别名必须由业务确认。ASR 新出现的错误词只能先进入审阅，不能自动成为正别名。

## 6. Q0：无提示自由转写

### 6.1 固定设置

```text
model = Qwen3-ASR-1.7B
backend = transformers
dtype = bfloat16
decode = greedy / do_sample=false
context = ""
batch_size = 1（效果基线）
language = manifest 指定的 Chinese 或 English
```

官方公开榜单使用 BF16、greedy；本项目必须将生成配置写入签名，不能只依赖 checkpoint 的隐含 `generation_config`。

### 6.2 language 策略

主结果强制 manifest 语言：

```text
lang=zh -> Chinese
lang=en -> English
```

另行保留 `language=None` 审计分支。若 auto 与 forced 差异显著，记录 `language_instability=true`，不能静默选更有利的结果。

### 6.3 max_new_tokens

短注册音频默认上限可从 64 起步；若输出 token 数命中上限，必须标记并用更高上限重跑。不得把可能截断的文本当正常 CER。

重复“嗯、啊、嗨”即使未命中上限，也由重复检测器标记。

## 7. Q1：wake_text 上下文转写（当前临时基线）

### 7.1 context 模板

当前实现只传入当前样本的 `wake_text`，不扩展语言级词表，也不拼接额外说明。示例：

```text
Hi Colmo
```

中文示例：

```text
海尔
```

后续如要引入 alias/词表列表，必须另建 Q1 版本和缓存命名空间，不能覆盖此基线。

### 7.2 使用边界

Q1 只产生：

```text
cer_context
context_gain = cer_free - cer_context
context_copy_risk
```

规则：

```text
Q0命中 + Q1命中 -> 强文本一致性
Q0近音 + Q1标准拼写 -> 可能是合理拼写修正
Q0不命中 + Q1命中 -> 召回候选，必须由 Q2/CTC/独立ASR确认
Q0空文本 + Q1命中 -> 高风险 context 注入
```

Q1 不允许直接覆盖 Q0 的主 CER。

## 8. Q2：teacher-forced 注册文本 NLL

现有 `kws/qkw_nll.py` 保持“目标文本只作为 labels、不进入 context”的原则。

### 8.1 alias 与混淆词

对每个正别名分别计算：

```text
nll_target = min_a NLL(a | audio), a in positive_aliases
best_alias = argmin_a NLL(a | audio)
```

对混淆词计算：

```text
nll_confusable = min_c NLL(c | audio)
nll_margin = nll_confusable - nll_target
```

`nll_margin` 越大，说明音频更支持目标读法而不是最强混淆读法。

### 8.2 使用限制

未校准时：

1. 只能同 UID、同主 CER 内比较；
2. 不能设置跨 UID 的绝对拒绝阈值；
3. 不能把 `exp(-NLL)` 直接称为 `q_kw`。

校准方法与验收见《CTC与q_kw落地技术方案》。

## 9. Q3：独立 ASR 证据

SenseVoice/Paraformer 作为独立模型保留，但调整其职责：

1. 中文继续计算无声调拼音 CER；
2. 英文除 alias 字符 CER 外增加音素 CER；
3. 品牌拼写错误不等于声学完全错误；
4. 不使用 Qwen context，也不使用 Qwen 的转写作为提示。

Q3 不直接推翻更低的 Q0 主 CER，只进入 `q_kw` 特征、同 CER 排序和审阅解释。

## 10. 幻觉与异常检测

每条 Q0/Q1 输出计算：

```text
speech_ratio
silence_ratio
output_char_count
output_tokens_per_sec
repeat_ngram_ratio
common_filler_ratio
requested_language
returned_language
context_target_hit
free_target_hit
batch_reference_match
```

至少定义以下原因码：

```text
ASR_EMPTY_VALID
ASR_EMPTY_ON_SPEECH
ASR_CONTEXT_ONLY_TARGET
ASR_TARGET_ON_LOW_SPEECH
ASR_REPETITION
ASR_OUTPUT_TOO_LONG
ASR_LANGUAGE_MISMATCH
ASR_HIT_MAX_TOKENS
ASR_BATCH_NONINVARIANT
ASR_RUNTIME_INCOMPATIBLE
```

这些原因首先作为诊断和 `q_kw` 特征。非有限输出、运行时不兼容、结果数不匹配属于 fatal；单独的疑似幻觉不删除证据，而是禁止该视图产生可信 CER0。

## 11. Sidecar 契约

建议 schema：

```json
{
  "schema": "quick_asr/v2",
  "score_key": "...",
  "candidate_id": "...",
  "pcm_sha256": "...",
  "asr_pcm_sha256": "...",
  "wake_text": "hicolmo",
  "lang": "en",
  "mode": "qwen_free",
  "hyp": "Hi Colmo",
  "returned_language": "English",
  "cer_route": 0.0,
  "best_alias": "hi colmo",
  "phoneme_cer": null,
  "context_hash": null,
  "hallucination_flags": [],
  "duration_sec": 1.42,
  "model_hash": "...",
  "runtime_hash": "...",
  "preprocess_hash": "..."
}
```

去重键：

```text
ASR key = SHA256(asr_pcm_sha256, wake_text, lang, mode,
                 alias_hash, model_hash, runtime_hash, preprocess_hash)
```

同一 PCM 在 Q0/Q1 中必须分别缓存，因为 context 不同。

## 12. 建议新增和修改的文件

### 12.1 新增（已落地/保留扩展）

```text
scripts/score_qwen_dual_manifest.py       # 已落地：Q0+Q1
（独立音素 ASR、幻觉审计和集中式 schema 校验仍作为下一阶段扩展）
```

### 12.2 修改

```text
kws/src/kws/audio.py
  -> 高质量重采样，或 quick 新建独立 canonical ASR 预处理，避免改坏其他项目

kws/src/kws/asr_transcribe.py
  -> 删除静默 context/prompt 降级；返回 language 和生成诊断

kws/src/kws/qkw_nll.py
  -> 支持正 alias、混淆词和 margin；保留无 context

scripts/run_a3_route.sh
  -> 已新增 qwen backend；主 CER 绑定 Q0
```

优先选择在 `quick` 内新增预处理与适配器，避免未经完整回归直接改变 `kws` 的共享行为。

## 13. 分阶段执行

### A0：冻结现状

保存当前 `context=wake,batch=8` sidecar 和签名，只作为对照，不覆盖。

### A1：修正预处理

对 raw、48 kHz SE 生成统一 16 kHz canonical PCM，审计时长和哈希。

### A2：Q0 全量基线

```bash
python scripts/score_asr_manifest.py \
  --manifest /root/autodl-tmp/quick_qwen_v2/candidate_manifest.jsonl \
  --output /root/autodl-tmp/quick_qwen_v2/asr_qwen_free.jsonl \
  --model-dir /root/autodl-tmp/Qwen3-ASR-1.7B \
  --device cuda:0 \
  --batch-size 1 \
  --context-mode none
```

当前实现已使用 `kws.audio.load_wav_mono` 的 soxr HQ/scipy polyphase canonical PCM，并写入
`model_hash/runtime_hash/context_mode/resampler/duration`。重复运行可通过 `--cache-dir`
复用，不会把旧 context 结果混入 Q0。

```bash
python scripts/score_asr_manifest.py \
  --manifest /root/autodl-tmp/quick_qwen_v2/candidate_manifest.jsonl \
  --output /root/autodl-tmp/quick_qwen_v2/asr_qwen_free.jsonl \
  --model-dir /root/autodl-tmp/Qwen3-ASR-1.7B \
  --batch-size 1 --context-mode none \
  --cache-dir /root/autodl-tmp/quick_asr_cache
```

### A3：Q1 全量辅助分支

运行 `wake_text` context，输出独立 sidecar，不覆盖 A2：

```bash
python scripts/score_qwen_dual_manifest.py \
  --manifest /root/autodl-tmp/quick_qwen_v2/candidate_manifest.jsonl \
  --free-output /root/autodl-tmp/quick_qwen_v2/asr_qwen_free.jsonl \
  --context-output /root/autodl-tmp/quick_qwen_v2/asr_qwen_context.jsonl \
  --model-dir /root/autodl-tmp/Qwen3-ASR-1.7B \
  --batch-size 1 --duration-bucket-sec 0.5 \
  --cache-dir /root/autodl-tmp/quick_asr_cache
```

`free-output` 是主 CER 输入；`context-output` 只能作为召回、alias 和
“是否依赖注册词”诊断输入。Qwen 适配器对 `context` 参数不支持时直接失败，
不再静默改用 prompt 或无上下文。

当前 Q1 暂时只把每条样本自己的 `wake_text` 作为 context；不注入语言级词表列表，
以便先测量 context 带来的召回和幻觉。`--vocab-json` 暂停使用，传入会直接报错。

在完整 s1/s7 路由中，Q0 可直接这样启用（`run_a3_route.sh` 已支持）：

```bash
cd /root/quick
ASR_BACKEND=qwen \
QUICK_DIR=/root/quick \
POS_NEG=/root/autodl-tmp/kws_sep_fullaudio_v1 \
WORK_DIR=/root/autodl-tmp/quick_a3_qwen_v1 \
AUDIO_CACHE_ROOT=/root/autodl-tmp/quick_audio_cache_a3_v1 \
PRECOMPUTED_SE_DIR=/root/autodl-tmp/quick_audio_cache_a3_v1/se48k \
QWEN_ASR_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B \
ASR_CACHE_DIR=/root/autodl-tmp/quick_asr_cache \
EXPECTED_UIDS=1838 \
QWEN_BATCH_SIZE=1 \
DURATION_BUCKET_SEC=0.5 \
bash /root/quick/scripts/run_a3_route.sh
```

这里 `POS_NEG` 只负责读取 s1/s7 索引和确认 UID；raw 音频优先复用
`AUDIO_CACHE_ROOT/sep_pcm`，MossFormer 音频从精确的
`AUDIO_CACHE_ROOT/se48k` 目录递归按 PCM 哈希复用。不能把包含 `sep_pcm`
和 `se48k` 的缓存根目录整体填给 `PRECOMPUTED_SE_DIR`。

双 sidecar 完成后可只做审计、不重新推理：

```bash
python scripts/audit_qwen_asr.py \
  --manifest /root/autodl-tmp/quick_qwen_v2/candidate_manifest.jsonl \
  --free /root/autodl-tmp/quick_qwen_v2/asr_qwen_free.jsonl \
  --context /root/autodl-tmp/quick_qwen_v2/asr_qwen_context.jsonl \
  --output /root/autodl-tmp/quick_qwen_v2/qwen_asr_audit.json
```

### A4：Q2 alias NLL

对全部 unique `(PCM, wake_text, lang)` 计算正 alias NLL、混淆词 NLL 和 margin。

### A5：独立 ASR

复用 SenseVoice/Paraformer 结果，补充英文音素 CER。

### A6：ASR 报告

按中文、英文、raw/SE、s1/s7、音乐、噪声、重叠人声分层报告。

## 14. 批处理加速规则

效果基线通过后，允许按以下顺序提速：

1. 按 canonical sample count 严格相同分组；
2. 再尝试窄时长桶；
3. 每个桶抽取 batch=1 参考结果；
4. 文本、语言、CER 必须与 batch=1 一致；
5. 出现任一不一致，该时长桶回退 batch=1。

不得直接把全量 `batch_size` 从 1 调回 8。

## 15. 验收标准

### 15.1 数据与契约

1. `qwen_free` 覆盖全部非 fatal unique `score_key`；
2. 输出条数、candidate alias 映射与 manifest 一致；
3. 相同 ASR key 不出现不同文本；
4. 模型、运行时、context、预处理和 alias hash 非空；
5. 没有静默 fallback。

### 15.2 效果

必须与现有 SenseVoice/Paraformer 和旧 Qwen 配置做全量配对比较：

```text
mean CER
CER0 / CER0 rate
hicolmo 正样本 alias recall
no-target / noise / music target hallucination rate
context-only target rate
batch non-invariance rate
中英文分别统计
```

最低要求：

1. Q0 相对当前替代 ASR 的主 CER 有明确改善；
2. Q1 不能单独造成生产 CER0；
3. batch=1 基准中无结果数错位；
4. 所有 UID 相对基线的退化清单可逐条审阅；
5. 下游仍满足现有 full coverage 和逐 UID 不退化门。

### 15.3 生产边界

ASR 本地通过后状态仍应为：

```text
LOCAL_PASS_NEEDS_QKW_CMD_PRESENCE
production_approved = false
```

只有 CTC/`q_kw` 校准、裁剪复算、CMD、Presence、FRR/FAR 和最终数据集验证全部通过后才能批准。

## 16. 回退策略

1. Qwen 运行失败：回退冻结的 SenseVoice/Paraformer sidecar，不临时重新推理。
2. Q1 幻觉过高：关闭 Q1，保留 Q0/Q2。
3. 新重采样器异常：保留旧 PCM 证据并使用新 work dir，不覆盖缓存。
4. 分桶 batch 不一致：仅回退 batch=1，不改变模型和文本策略。
5. Qwen 新版本漂移：固定旧容器、包版本和模型 revision，重新建 runtime hash 后再评估。

## 17. 主要外部依据

- Qwen3-ASR 官方实现与使用说明：https://github.com/QwenLM/Qwen3-ASR
- Qwen3-ASR-1.7B 模型卡：https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- context 注入报告：https://github.com/QwenLM/Qwen3-ASR/issues/186
- Transformers 异长 batch 报告：https://github.com/QwenLM/Qwen3-ASR/issues/207
- 噪声/回声幻觉报告：https://github.com/QwenLM/Qwen3-ASR/issues/165
- vLLM 与直接推理差异报告：https://github.com/QwenLM/Qwen3-ASR/issues/188
