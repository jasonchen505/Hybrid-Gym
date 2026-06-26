# Hybrid-Gym 复现过程增量学习笔记

> 相比前两轮（项目概览 + 面试准备），本轮深入算力评估和全流程复现规划中获得的新认知

---

## 一、训练算力层面的新认知

### 1.1 论文训练配置的精确细节

**前两轮只知道：** "用 Qwen2.5Coder-7B/32B 做 SFT"

**本轮深入发现：**

| 维度 | 7B | 32B |
|------|-----|------|
| GPU | 8x A6000 (48GB each) | 2x H100 (80GB each) |
| 总 VRAM | 384 GB | 160 GB |
| 学习率 | 5e-5（从 {5e-5, 1e-4} 搜索） | 5e-5（固定，计算受限） |
| Epochs | 5（从 {3,5} 搜索） | 5（固定） |
| Batch Size | 8（从 {8,16} 搜索） | 16（固定） |
| 超参搜索 | 做了 grid search | 没做（计算不够） |

**关键 insight：** 32B 模型因为 H100 资源有限，没有做超参搜索，直接用了 7B 搜索出的最优配置。这意味着论文的 32B 结果可能还有提升空间。

### 1.2 8x4090 vs 论文配置的精确差距

**前两轮只知道：** "4090 VRAM 比 A6000 少"

**本轮精确计算：**

```
7B 模型全参数 SFT 的 VRAM 需求估算：
- 模型权重 (bf16): 7B × 2 bytes = 14 GB
- AdamW optimizer states: 14 GB × 2 = 28 GB (fp32 momentum + variance)
- 梯度 (bf16): 14 GB
- 激活值: 取决于 batch size 和 sequence length
- 总计: ~56-70 GB (不含激活值)

单卡 4090 (24GB) 无法承载 → 必须 ZeRO-3 分片

ZeRO-3 分片后：
- 模型权重: 14 GB / 8 = 1.75 GB/卡
- Optimizer states: 28 GB / 8 = 3.5 GB/卡 (或 offload 到 CPU)
- 梯度: 14 GB / 8 = 1.75 GB/卡
- 激活值: 需要 gradient checkpointing 控制
- 总计: ~7-10 GB/卡 (offload optimizer) 或 ~15-20 GB/卡 (不 offload)
```

**结论：** 8x4090 用 ZeRO-3 + CPU offload optimizer 可以训练 7B，甚至 32B 也可以尝试。

### 1.3 32B 在 8x4090 上的可行性

**前两轮认为：** "32B 可能不行"

**本轮精确分析：**

```
32B 模型全参数 SFT 的 VRAM 需求估算：
- 模型权重 (bf16): 32B × 2 bytes = 64 GB
- ZeRO-3 分片后: 64 GB / 8 = 8 GB/卡
- AdamW optimizer states: 128 GB → 必须 offload 到 CPU
- 梯度分片: 64 GB / 8 = 8 GB/卡
- 激活值: gradient checkpointing 必须开启

总计: ~20-22 GB/卡 (offload optimizer + gradient checkpointing)
→ 刚好在 4090 的 24GB 限制内，但非常紧张
```

**可行但有风险：** batch size 只能为 1，需要大 gradient accumulation (16)，训练会很慢（~12-20 小时）。建议先尝试，如果 OOM 则用 QLoRA。

---

## 二、数据构建层面的新认知

### 2.1 数据预处理不依赖 GPU

**前两轮只知道：** "需要预处理数据"

**本轮发现：** 数据预处理主要是 CPU 任务 + API 调用：
- `get_docstring.py`: 用 AST 解析 Python 文件 + GPT-4o-mini 生成描述
- `build_dataset.py`: 用 Jedi 静态分析，**完全不需要 LLM**
- `convert_to_masked.py`: 用 LLM 生成 masked 描述

**关键发现：** Dependency Search 的 ground truth 是通过 Jedi（静态分析工具）生成的，不是 LLM。这意味着 dep_search 的数据构建是免费的。

### 2.2 Teacher 轨迹生成是主要成本

**前两轮只知道：** "需要 teacher 模型生成轨迹"

**本轮精确估算：**

```
每条轨迹的成本 = (平均步数 × 每步 token 数 × API 单价)

以 Claude-Sonnet 为例：
- 平均步数: 26.9 (dep_search) ~ 52.4 (issue_localize)
- 每步 prompt: ~2000 tokens
- 每步 response: ~500 tokens
- Claude-Sonnet-4.5: ~$3/1M input, $15/1M output

每条轨迹估算:
- Input: 50 steps × 2000 tokens = 100K tokens → $0.30
- Output: 50 steps × 500 tokens = 25K tokens → $0.375
- 总计: ~$0.675/条

但考虑到成功率（~50%），实际需要 2x rollout:
- 4470 条成功轨迹 × 2 = ~8940 次 rollout
- 总成本: 8940 × $0.675 ≈ $6,033

这个数字远高于论文报告的 $0.07/条环境成本，因为论文只计算了 Docker 环境成本，不包括 API 调用。
```

**关键 insight：** 论文报告的 "0.07¢/instance" 只是环境搭建成本（Docker），不包括 teacher 模型的 API 调用费用。实际复现的主要成本是 API 调用。

### 2.3 数据转换中的 FunctionCallConversionError

**前两轮只知道：** "有些轨迹转换会失败"

**本轮深入发现：** `convert_data.py` 中的转换失败率可能很高：
- 多 tool call → 单 tool call 转换可能失败
- fncall → non-fncall 转换可能失败
- 失败的轨迹被完全丢弃

**实际影响：** 如果 teacher 模型生成了 8000 条轨迹，但只有 4470 条成功 + 转换成功，最终训练数据可能只有 3000-4000 条。

---

## 三、评估层面的新认知

### 3.1 SWE-Bench 评估需要专用 Docker 镜像

**前两轮只知道：** "评估在 Docker 中运行"

**本轮发现：** SWE-Bench 的评估使用 **instance-specific 的 Docker 镜像**（每个 issue 一个预构建的镜像）。这些镜像包含了：
- 正确版本的代码仓库
- 正确版本的依赖
- 预配置的测试环境

**这意味着：** 本地评估 SWE-Bench 需要拉取大量 Docker 镜像（每个 instance 约 1-5 GB）。500 个 instance 可能需要 500GB-2.5TB 磁盘空间。

**替代方案：** 使用 All Hands 的 remote runtime API（`https://runtime.eval.all-hands.dev`），避免本地存储大量镜像。

### 3.2 评估超时机制的多层设计

**前两轮只知道：** "有超时机制"

**本轮发现：** 超时有 5 层：
1. **LLM API 超时：** `LLMConfig.timeout`（可配置）
2. **LLM 重试总超时：** 5+10+20+30 = 65 秒（指数退避）
3. **Docker 健康检查超时：** 120 秒（`wait_until_alive`）
4. **Action 执行超时：** 120 秒（`SandboxConfig.timeout`）
5. **评估实例超时：** 8 小时（`signal.SIGALRM`）

**关键发现：** 评估超时使用 `signal.SIGALRM`，这是 Unix-only 的，不支持 Windows。而且 `SIGALRM` 在多线程环境中不安全。

### 3.3 Dep-Search 评估的 5-line Tolerance

**前两轮只知道：** "评估检查注释位置"

**本轮发现：** 注释位置允许 ±5 行的偏移。这是因为 agent 添加注释时可能有微小的位置偏差。评估算法：
1. 用 `compute_line_offset` 计算行偏移（因为添加注释会改变后续行号）
2. 检查注释是否在 `(adjusted_expected_line ± 5)` 行范围内
3. 用模糊匹配验证注释内容（必须包含目标函数名 + 关系指示词）

---

## 四、工程架构层面的新认知

### 4.1 OpenHands 支持 8 种 Runtime

**前两轮只知道：** "用 Docker runtime"

**本轮发现：** OpenHands 支持 8 种 runtime 后端：
1. Docker（默认）
2. Remote（All Hands 云）
3. E2B（云沙箱）
4. Modal（Serverless）
5. Local（本地直接执行）
6. CLI（命令行）
7. Daytona（云开发环境）
8. Runloop（云环境）

**实际意义：** 如果本地 Docker 资源不足，可以用 remote runtime。但需要 All Hands API key。

### 4.2 LLM 集成使用 LiteLLM

**前两轮只知道：** "支持多种 LLM"

**本轮发现：** OpenHands 使用 **LiteLLM** 作为统一 LLM 网关，这意味着：
- 支持 100+ LLM 提供商
- 自动处理 API 格式差异
- 统一的重试和错误处理
- 支持 function calling 自动转换

**实际意义：** 复现时可以灵活选择 LLM 提供商。如果 Claude 太贵，可以用 OpenRouter 或其他代理。

### 4.3 Condenser 系统的 LLM 调用开销

**前两轮只知道：** "有 9 种压缩策略"

**本轮发现：** LLM-based condensers（LLMAttention、LLMSummarizing、StructuredSummary）需要额外的 LLM 调用。这意味着：
- 每次压缩可能需要 1 次额外的 LLM API 调用
- 对于长轨迹（50+ 步），可能需要多次压缩
- 这会显著增加 API 成本

**实际意义：** 评估时应该用 `NoOpCondenser` 或 `RecentEventsCondenser`，避免额外的 LLM 调用开销。

---

## 五、复现策略层面的新认知

### 5.1 论文的数据规模 vs 复现数据规模

**前两轮只知道：** "4470 条轨迹"

**本轮分析：**

```
论文数据规模:
- Func-Localize: 1,438 条 (从 2,300 个实例筛选)
- Issue-Localize: 1,978 条 (从 3,000 个实例筛选)
- Dep-Search: 502 条 (从 800 个实例筛选)
- Func-Gen: 552 条 (从 2,000 个实例筛选)
- 总计: 4,470 条

复现建议规模 (MVP):
- 每个任务: 200 条成功轨迹
- 总计: ~800 条
- 需要 rollout: ~1600 次 (假设 50% 成功率)
- API 成本: ~$1,000 (Claude) 或 ~$100 (GPT-4o-mini 作为 teacher)
```

**关键 insight：** 论文的 Scaling Law 实验显示，即使只有 250 条轨迹（5%），也能获得显著提升。所以 MVP 的 800 条应该足够验证 pipeline。

### 5.2 可以用更便宜的 Teacher 模型

**前两轮只知道：** "用 Claude 和 Qwen3 作为 teacher"

**本轮发现：** 论文中 Qwen3-235B 也被用作 teacher 模型（用于 dep_search 和部分 func_localize）。这意味着：
- 可以用更便宜的模型作为 teacher
- 甚至可以用本地部署的模型（如果有足够 GPU）
- 成功率可能更低，但成本也更低

**复现策略：** 先用 GPT-4o-mini 作为 teacher（最便宜），验证 pipeline 后再升级到 Claude。

### 5.3 训练框架不在项目中

**前两轮只知道：** "用 SFT 训练"

**本轮发现：** 项目本身 **不包含训练代码**。训练需要外部框架：
- LLaMA-Factory（推荐，支持 DeepSpeed）
- OpenRLHF
- transformers + trl
- Axolotl

**实际影响：** 需要额外安装和配置训练框架，数据格式需要转换。

---

## 六、关键代码路径索引（复现用）

### 数据预处理
- `evaluation/benchmarks/hybrid_gym_func_localize/preprocess/get_docstring.py` — 函数描述生成
- `evaluation/benchmarks/hybrid_gym_func_localize/preprocess/convert_to_masked.py` — Masked 描述生成
- `evaluation/benchmarks/hybrid_gym_dep_search/preprocess/build_dataset.py` — Jedi 依赖分析

### 轨迹生成
- `evaluation/benchmarks/hybrid_gym_func_gen/run_infer_no_image.py` — Func-Gen 推理
- `evaluation/benchmarks/hybrid_gym_func_localize/run_infer_no_image.py` — Func-Localize 推理
- `evaluation/benchmarks/hybrid_gym_dep_search/run_infer.py` — Dep-Search 推理
- `evaluation/benchmarks/hybrid_gym_issue_localize/run_infer.py` — Issue-Localize 推理

### 数据转换
- `evaluation/combine_final_completions.py` — 轨迹压缩
- `evaluation/convert_data.py` — 训练数据格式转换

### 评估
- `evaluation/benchmarks/swe_bench/run_infer.py` — SWE-Bench 推理
- `evaluation/benchmarks/swe_bench/eval_infer.py` — SWE-Bench 评估
- `evaluation/benchmarks/hybrid_gym_*/eval_*.py` — 各任务评估

### 配置
- `config.template.toml` — 配置模板
- `openhands/core/config/llm_config.py` — LLM 配置
- `openhands/core/config/sandbox_config.py` — Sandbox 配置

### Docker
- `containers/app/Dockerfile` — 应用容器
- `openhands/runtime/utils/runtime_templates/Dockerfile.j2` — Runtime 容器模板
- `containers/build.sh` — 构建脚本

---

## 七、与前两轮文档的对比总结

| 维度 | INTERVIEW_PREP.md | INTERVIEW_DEEP_DIVE.md | 本文档 REPRODUCTION_PLAN.md |
|------|-------------------|----------------------|---------------------------|
| **视角** | 项目概览 | 面试应对 | 实际复现 |
| **深度** | 中等 | 深（原理+代码） | 最深（算力+成本+时间） |
| **新信息** | 论文结果 | 代码细节 | 训练配置精确参数、成本估算、可行性分析 |
| **实用性** | 了解项目 | 面试准备 | 实际动手复现 |

**本文档独有的增量认知：**
1. 论文 32B 训练没做超参搜索（计算受限）
2. 数据预处理主要是 CPU 任务，dep_search 完全不需要 LLM
3. 论文的 0.07¢/instance 只是环境成本，不含 API 费用
4. 评估需要 instance-specific Docker 镜像（磁盘需求巨大）
5. 项目不含训练代码，需要外部 SFT 框架
6. 8x4090 用 ZeRO-3 可以训练 7B，32B 需要 CPU offload
7. LLM-based condensers 有额外 API 调用开销
8. Scaling Law 显示 250 条轨迹就能获得显著提升（MVP 可行）
