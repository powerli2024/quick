# quick：s1/s7 + MossFormer SE 严格选路

`quick` 是按 `D:\gpt\kws\docs\S1_S7选路与平铺审阅导出方案.md` 落地的独立实现，不修改
`kws` 项目。它完成 I0–I8：输入和签名审计、PCM 去重、raw/SE 评分、s1/s7 选路、逐 UID reason JSON、
平铺审阅导出、selected-only 导出，以及 CMD/Presence/contest 验收。

正式运行需要提供冻结 ASR sidecar（或 `--asr-command`）、MossFormer SE 批处理命令/预计算目录，以及可选
的 q_kw、embedding、噪声模型 sidecar。没有这些外部模型结果时程序会保留明确的 `PENDING_EXTERNAL` 或
`NO_GO`，不会把缺失模型当成通过。

## 快速检查

```powershell
cd D:\gpt\quick
python -m pytest -q
python scripts\run_s1_s7_route.py --help
```

## 输入约定

`--pos-neg` 与 kws 相同，根目录包含 `pos/`、`neg/`，每个阶段 arm 下有 `index.jsonl` 和 `wav/`。
index 中至少要有 `uid`、`wake_text`（旧字段 `唤醒文本`/`wake`/`text` 也可）、`streams`；缺少
`lang` 时按唤醒文本推导。s1/s7 arm 必须显式传入，最终禁止 `auto`。

## 设计文档

完整的候选排序、s7 触发、SE 解释、平铺文件名和 JSON 契约见：

`D:\gpt\kws\docs\S1_S7选路与平铺审阅导出方案.md`
