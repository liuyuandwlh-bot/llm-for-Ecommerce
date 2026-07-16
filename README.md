# LLM Portfolio Platform（LLM 全栈作品集）

> 电商客服领域微调 + 金融研报 RAG 一体化骨架。

本仓库实现了一个**可验证、可复现、可评测**的 LLM 系统骨架，覆盖两条主线：

1. **电商客服（E-commerce Customer Service）**——基于自有虚构 3C 数码店铺 SOP、确定性生成的合成对话、可复现的训练-评测流水线。
2. **金融研报检索增强问答（Financial RAG）**——结构化分块 + 混合检索（BM25 + 向量 + RRF） + 父-子上下文扩展。

代码优先解决**真实运行起来发现的问题**，而不是堆砌新空模块。每一行非平凡逻辑都对应一份 `pytest` 用例或 CLI smoke。

---

## 项目状态（截至第二轮整改）

| 模块                              | 状态 |
| -------------------------------- | ---- |
| 框架（sop / cases / conv / pipeline / train / eval / rag） | ✅ 实现 + 单测 + 集成测试 |
| 训练数据 | ✅ 仅自有 fixture（`owned_sop_v1`） |
| 真实模型训练 | ⏸️ 本轮不出模型（受硬件约束，未跑 8B LoRA） |
| 公开数据（公告、年报） | ⏸️ 本轮未爬取，只留 schema + manifest 接口 |
| DPO 偏好训练 | ⏸️ 明确标注 experimental，仅做 schema 校验 |

> 详见 [`LICENSE`](./LICENSE) 中的 third-party 段落。**不**随仓库分发任何第三方数据或权重；只有自写 fixture。

---

## 1. 目录结构

```
src/
├── common/                 # 哈希 / 日志 / PII / 注册 / 顶层 CLI
│   ├── cli.py              # `llm-portfolio` 顶层入口
│   ├── hashing.py          # 稳定 SHA-256 跨进程 ID
│   ├── near_dedup.py       # n-gram Jaccard + 冲突隔离
│   ├── pii.py              # 手机/身份证/邮箱检测与掩码
│   └── schemas.py          # FastAPI 入参/出参
├── ecommerce/
│   ├── dataset/            # 全部离线数据生成
│   │   ├── sop_builder.py          # 14 条自有 SOP + registry 校验
│   │   ├── policy_engine.py        # 确定性匹配 + 冲突/缺失/歧义
│   │   ├── canonical_cases.py      # 26 个跨意图结构化测试样本 + 验证器
│   │   ├── conversation_generator.py # 78 条确定性对话 + 多策略改写
│   │   ├── registry.py             # 数据来源登记
│   │   └── pipeline.py             # 9 阶段漏斗 + 严格分组切分 + 泄漏报告
│   ├── train/              # 训练脚本
│   │   ├── sft_trainer.py          # 自定义 assistant-only collator + dry-run
│   │   ├── sft_config.py           # 严格 YAML + 锁定依赖版本
│   │   └── dpo_trainer.py          # 实验性：拒绝训练，仅 schema 校验
│   └── eval/               # 评测
│       ├── metrics.py              # intent/slot/policy/decision/handoff/tool/safety
│       └── evaluator.py            # fixture_oracle / mock / real_model 三后端
├── finance_rag/            # 金融 RAG
│   ├── pdf_parser.py       # PDF → 结构化（docling/pdfplumber）
│   ├── chunker.py          # 父子分块 + 表格 + 稳定 chunk_id
│   ├── retriever.py        # BM25 + dense(fake/real) + RRF + parent 扩展
│   ├── document_ingest.py  # manifest / checksum / 版权登记
│   └── answer_engine.py    # 抽取式 / 计算 / 局限说明
└── serving/
    └── gateway/            # FastAPI 网关（health / customer-service / finance-rag / batch）
configs/
├── train/                  # Qwen3-8B / Llama3-8B YAML 配置（锁定 model.revision）
└── rag/                    # 检索参数
data/
└── fixtures/finance/parsed/fixture_a_2024.json  # 自写 RAG demo 文档
tests/
├── unit/                   # PolicyEngine / Cases / ConvGen / Pipeline / Schema
├── integration/            # API + 完整流水线往返
└── adversarial/            # PII 泄漏 / 注入检测
```

---

## 2. 快速开始

### 2.1 安装

```bash
git clone <your-repo>
cd llm-portfolio-platform
pip install -e ".[dev]"
```

> 仅离线骨架跑得通；SFT/DPO 真实训练需要额外 `pip install -e ".[train]"`，并自行下载 Qwen3-8B / Llama-3-8B 权重（需接受对应许可）。

### 2.2 一行命令跑通完整 E-commerce 流水线

```bash
python -m src.common.cli sop-build   --output data/processed/policies.json --validate --registry data/processed/registry.json
python -m src.common.cli cases-build --policies data/processed/policies.json --output data/processed/cases.jsonl --validate
python -m src.common.cli conv-build  --policies data/processed/policies.json --cases data/processed/cases.jsonl --output data/processed/conv.jsonl --mode fixture
python -m src.common.cli pipeline    --input data/processed/conv.jsonl --policies data/processed/policies.json --registry data/processed/registry.json --output data/processed/out
```

或用顶层脚本：

```bash
llm-portfolio sop-build   --output data/processed/policies.json --validate --registry data/processed/registry.json
```

### 2.3 RAG smoke（无需 GPU）

```bash
python -m src.finance_rag.chunker   --input data/fixtures/finance/parsed --output data/processed/chunks.jsonl
python -m src.finance_rag.retriever --chunks data/processed/chunks.jsonl --backend fake --output data/processed/rag_index --smoke-query "示例公司营业收入是多少"
```

---

## 3. 设计契约（Contract）

### 3.1 `PolicyEngine`

* 输入：`policy` 列表 + `context` 字典
* 输出：`PolicyMatch`，包含 `policy_id / decision / missing_slots / conflicted_policies / candidate_policy_ids / ambiguity / requires_human`
* 行为：
  * **冲突优先**：当某策略的条件与上下文显式冲突时，从候选中剔除
  * **歧义检测**：多个非冲突策略同时满分 → `ambiguity=True`
  * **缺槽策略**：返回得分最高的「部分匹配」，告知需要补哪些槽
  * **行为意图**（injection / privacy / out_of_scope / small_talk）走 `match_behavior()`，不入主决策流

### 3.2 `CanonicalCase` / `validate_canonical_cases`

* 每个 case 至少包含：`case_id / intent / case_type / turns / context / expected_decision / expected_policy_ids`
* 验证器做四件事：
  1. `case_id` 唯一
  2. `expected_policy_ids ⊆ 所有 policy.policy_id`
  3. 把 case 的 `context` 喂给 `PolicyEngine`，结果必须匹配 `expected_decision` 与 `expected_missing_slots`
  4. 行为意图（`is_behavior=True`）的 `tool_expectation` 必须与 `BehaviorIntent` 规则一致
* **当前：26 个用例全部通过验证（severity=0）**

### 3.3 `ConversationGenerator`

* 三种改写策略注册为策略表 (`_STRATEGY_REGISTRY`)：`original / colloquial / emotional / missing_slot_followup / split_slots / short_alias`
* 每条 case 默认产出 3 个变体（共 **78 条对话**），通过同一 `seed` 跨进程可复现
* `LLMGenerator` 显式抛出 `LLMUnavailableError`（必须设置 `OPENAI_API_KEY` 才允许加载）

### 3.4 数据漏斗

9 阶段，全部计数 + 第一失败原因归因：

```
input → schema_pass → registry_pass → pii_masked → policy_pass
      → role_pass → quality_pass → review_pass → final
```

去重环节：
* `exact_dedup_removed`：完全相同 `sample_id` 计数
* `near_dedup_removed`：按用户业务文本 n-gram Jaccard ≥ 0.7 合簇；冲突标签自动隔离到 `quarantined.jsonl`

切分：
* 按 `group_id = hash(case_id, template_family, rewrite_strategy)` 分组，**同组只能落一个 split**
* `train > dev ≥ test`；`stratified` 按 intent 平衡
* `leak_count == 0` 时整流 `exit_ok=True`

### 3.5 SFT / DPO

* **SFT**：`tokenize_with_assistant_mask` + `PadCollator` —— 仅 assistant token 参与 loss
* `dry-run` 不加载模型，只做：
  1. 配置严格校验（未知字段抛 `ValueError`）
  2. 训练数据路径与 tokenization 烟雾测试（用 `QwenTokenizer` / `LlamaTokenizer` 兼容的 `_FakeTokenizer`）
* **DPO**：本轮明确转为 experimental。CLI 只有 `--check-schema` / `--validate-data`，不会真训练。

---

## 4. 评测

```bash
python -m src.ecommerce.eval.evaluator \
  --backend fixture_oracle \
  --test-data data/processed/cases.jsonl \
  --output-dir data/processed/eval
```

输出 `summary.json` 中关键字段：

```json
{
  "intent": { "macro_f1": 1.0, "micro_f1": 1.0, "exact_match": 1.0 },
  "slot":    { "macro_f1": 1.0, "micro_f1": 1.0, "exact_match": 1.0 },
  "policy_id_set_exact_match": 1.0,
  "decision_exact_match": 1.0,
  "handoff_accuracy": 1.0,
  "tool": { "name_accuracy": 1.0, "argument_accuracy": 1.0, "n": 26 },
  "safety": { "pii_leak_count": 0, "injection_compliance_count": 0, "unauthorized_commitment_count": 0 },
  "backend_type": "fixture_oracle",
  "is_model_result": false,
  "n_samples": 26
}
```

* `backend_type ∈ {fixture_oracle, mock, real_model}` **永远显式标注**，杜绝把 mock 结果当真模型成绩。
* `is_model_result=false` 表示本次结果由 oracle/启发式产生，不能与真实模型直接比较。

---

## 5. 服务

```bash
uvicorn src.serving.gateway.main:app --reload --port 8000
curl http://localhost:8000/health
```

`/health` 响应字段：
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "mode": "mock",
  "models_loaded": [],
  "backend_ready": false,
  "cache_hits": 0,
  "cache_misses": 0,
  "uptime_seconds": 0.42
}
```

* `mode` 由环境变量 `LLM_PORTFOLIO_REAL_BACKEND=1` 切到 `real`，否则为 `mock`
* CORS 默认只放行 `http://localhost:3000`、`http://127.0.0.1:3000`；通过 `LLM_PORTFOLIO_ALLOWED_ORIGINS` 注入额外源
* 所有 API 路径：`POST /api/v1/customer-service`、`POST /api/v1/finance-rag`、`POST /api/v1/batch`

---

## 6. 测试

```bash
ruff check src/ --no-cache
python -m pytest tests/ -q
```

本轮交付：

| 项 | 数量 |
| --- | --- |
| ruff 错误 | **0** |
| pytest 通过 | **39 / 39** |
| 验收端到端 | SOP → Cases(26) → Convs(78) → Pipeline(train 27 / dev 7 / test 5)，`leak_count=0` |

---

## 7. 已知约束 / 不在本轮范围

* **不跑 8B 训练**：本轮硬件为单/双 A6000，Qwen3-8B 全量 SFT + eval 远超合理时长，dry-run 已验证 collator / 配置正确，真实训练请在后续轮次执行。
* **不爬真实金融公告**：`data/fixtures/finance/parsed/` 中只有一份自写的示例文档。`document_ingest.py` 提供 manifest 接口，添加真实 PDF 时请同步填写 publisher / license_note / checksum。
* **DPO 占位**：未提供示例偏好数据；偏好收集流程请在数据来源明确后再实现。
* **稀疏-稠密混合检索**：已实现 BM25 + Dense(fake) + RRF；reranker 接口预留，未实装。

---

## 8. 许可

代码部分遵循 [MIT](./LICENSE)。仓库中**不含**任何受版权保护的第三方数据或模型权重；用户需自行获取并遵守对应许可：

* **Qwen3-8B** — Apache-2.0
* **Llama-3.1-8B-Instruct** — Llama 3 Community License
* **BAAI/bge-m3** — MIT
* **金融公告 / 研报** — 来自对应发布方，使用前请登记 `license_note` 并确保 `allowed_train / allowed_evaluate` 字段正确