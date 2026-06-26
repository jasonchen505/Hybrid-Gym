# Hybrid-Gym 全流程复现计划

> 基于 8x NVIDIA RTX 4090 (24GB VRAM each, 192GB total) 的实际算力评估与复现方案

---

## 一、算力评估总览

### 1.1 硬件对比

| 维度 | 论文配置 | 你的配置 | 差异分析 |
|------|---------|---------|---------|
| **7B 训练** | 8x A6000 (48GB each, 384GB total) | 8x 4090 (24GB each, 192GB total) | VRAM 减半，需要 ZeRO-3 或 gradient checkpointing |
| **32B 训练** | 2x H100 (80GB each, 160GB total) | 8x 4090 (24GB each, 192GB total) | 总 VRAM 相当，但单卡 VRAM 差 3.3x，必须 ZeRO-3 |
| **推理/评估** | 未明确指定 | 8x 4090 | 充足，评估主要瓶颈是 API 调用和 Docker |
| **数据生成** | API 调用（Claude/GPT/Qwen3） | API 调用 | 与 GPU 无关，需要 API key 和预算 |

### 1.2 可行性判断

| 阶段 | 可行性 | 关键挑战 | 解决方案 |
|------|--------|---------|---------|
| **环境搭建** | ✅ 完全可行 | Python 3.12 + Poetry + Docker | 标准流程 |
| **数据预处理** | ✅ 完全可行 | CPU + 少量 API 调用 | GPT-4o-mini 成本极低 |
| **Teacher 轨迹生成** | ⚠️ 需要 API | Claude/Qwen3 API 费用 | 估算 ~$50-200 总成本 |
| **7B SFT 训练** | ✅ 可行 | 单卡 24GB 需要优化 | ZeRO-3 + gradient checkpointing |
| **32B SFT 训练** | ⚠️ 困难但可行 | 单卡 24GB 极度紧张 | ZeRO-3 + offload + gradient accumulation |
| **评估推理** | ✅ 可行 | Docker 容器并行 | 本地 Docker runtime |
| **下游评估** | ⚠️ 需要 API | SWE-Bench 需要 instance-specific Docker images | 可用 remote runtime 替代 |

---

## 二、全流程复现计划

### Phase 0: 环境搭建 (1-2 天)

#### 0.1 基础环境
```bash
# Python 3.12
conda create -n hybridgym python=3.12 -y
conda activate hybridgym

# Poetry
pip install poetry

# Node.js 22 (for frontend, optional for evaluation)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -bash
sudo apt-get install -y nodejs

# Docker
# 确保 Docker 已安装且当前用户有权限
docker --version
```

#### 0.2 项目依赖
```bash
cd /data/home/yizhou/Hybrid-Gym
poetry install --with dev,test,runtime,evaluation
```

#### 0.3 Docker 镜像拉取
```bash
# 评估所需镜像
docker pull python:3.11-bookworm
docker pull yiqingxyq/repost:v0
docker pull ghcr.io/all-hands-ai/runtime:0.40-nikolaik
```

#### 0.4 API Key 配置
```bash
# 创建 config.toml
cat > config.toml << 'EOF'
[llm]
model = "gpt-4o-mini"
api_key = "sk-..."  # OpenAI API key

[llm.claude]
model = "claude-sonnet-4-20250514"
api_key = "sk-ant-..."  # Anthropic API key

[sandbox]
timeout = 300
base_container_image = "python:3.11-bookworm"
EOF
```

**需要的 API Key：**
- OpenAI API Key（GPT-4o-mini 用于数据预处理）
- Anthropic API Key（Claude-Sonnet 用于 teacher 轨迹生成）
- 可选：Qwen3-235B API endpoint（用于部分 teacher 轨迹）

---

### Phase 1: 数据预处理 (1-3 天)

#### 1.1 Function Localization 数据构建

```bash
cd /data/home/yizhou/Hybrid-Gym

# Step 1: 从 SWE-Gym-Raw 提取函数并生成描述
python evaluation/benchmarks/hybrid_gym_func_localize/preprocess/get_docstring.py \
    --dataset SWE-Gym/SWE-Gym-Raw \
    --mode all_func \
    --sample-size 300 \
    --num-workers 4 \
    --output evaluation/benchmarks/hybrid_gym_func_localize/resource/func_localize_data.jsonl

# Step 2: 生成 masked 版本（去除函数名泄露）
python evaluation/benchmarks/hybrid_gym_func_localize/preprocess/convert_to_masked.py \
    --input evaluation/benchmarks/hybrid_gym_func_localize/resource/func_localize_data.jsonl \
    --output evaluation/benchmarks/hybrid_gym_func_localize/resource/func_localize_masked.jsonl \
    --num-workers 4
```

**成本估算：** GPT-4o-mini 调用 ~3000 次，~$0.50

#### 1.2 Dependency Search 数据构建

```bash
# 使用 Jedi 静态分析，不需要 LLM API
python evaluation/benchmarks/hybrid_gym_dep_search/preprocess/build_dataset.py \
    --dataset SWE-Gym/SWE-Gym-Raw \
    --sample-size 300 \
    --min-deps 1 \
    --max-deps 5 \
    --max-instances 1000 \
    --output evaluation/benchmarks/hybrid_gym_dep_search/resource/dep_search_data.jsonl
```

**成本估算：** $0（纯 CPU 计算）

#### 1.3 Issue Localization 数据

直接使用 SWE-Gym/SWE-Gym-Raw 数据集，无需预处理。

#### 1.4 Function Generation 数据

直接使用 HuggingFace 数据集 `hybrid-gym/hybrid_gym_func_gen`。

---

### Phase 2: Teacher 轨迹生成 (3-7 天)

> 这是最耗时和最昂贵的阶段。需要 teacher 模型（Claude-Sonnet）在 OpenHands 框架中执行任务，生成成功轨迹。

#### 2.1 评估 API 预算

| 任务 | 实例数 | 成功率(估) | 需要 rollout 数 | 每条成本(估) | 总成本(估) |
|------|--------|-----------|----------------|-------------|-----------|
| Func-Localize | 500 | ~60% | ~833 | $0.10 | $83 |
| Issue-Localize | 500 | ~50% | ~1000 | $0.15 | $150 |
| Dep-Search | 500 | ~70% | ~714 | $0.05 | $36 |
| Func-Gen | 500 | ~40% | ~1250 | $0.20 | $250 |
| **总计** | | | | | **~$519** |

> **注意：** 上述为粗略估计。实际成本取决于 Claude API 的 token 用量。每条轨迹平均 26-52 步，每步包含 prompt + response。

**降成本策略：**
1. 先用小规模（100 条/任务）验证 pipeline，再扩大规模
2. 使用 Qwen3-235B（如果能获得便宜的 API）替代部分 Claude 调用
3. 使用 Claude-Sonnet-3.7 替代 Claude-Sonnet-4.5（如果 3.7 更便宜）
4. 分批生成，先跑通再批量

#### 2.2 分步生成轨迹

**Step 1: Function Localization 轨迹生成**
```bash
cd /data/home/yizhou/Hybrid-Gym

# 先小规模测试
python evaluation/benchmarks/hybrid_gym_func_localize/run_infer_no_image.py \
    --llm-config claude \
    --agent-class CodeActAgent \
    --eval-num-workers 3 \
    --eval-n-limit 10 \
    --max-iterations 30 \
    --eval-output-dir evaluation/evaluation_outputs/func_localize_test

# 验证成功后扩大规模
python evaluation/benchmarks/hybrid_gym_func_localize/run_infer_no_image.py \
    --llm-config claude \
    --agent-class CodeActAgent \
    --eval-num-workers 5 \
    --eval-n-limit 500 \
    --max-iterations 30 \
    --eval-output-dir evaluation/evaluation_outputs/func_localize
```

**Step 2: Issue Localization 轨迹生成**
```bash
python evaluation/benchmarks/hybrid_gym_issue_localize/run_infer.py \
    --llm-config claude \
    --agent-class CodeActAgent \
    --eval-num-workers 3 \
    --eval-n-limit 500 \
    --max-iterations 100 \
    --eval-output-dir evaluation/evaluation_outputs/issue_localize
```

**Step 3: Dependency Search 轨迹生成**
```bash
python evaluation/benchmarks/hybrid_gym_dep_search/run_infer.py \
    --llm-config claude \
    --agent-class CodeActAgent \
    --eval-num-workers 5 \
    --eval-n-limit 500 \
    --max-iterations 30 \
    --eval-output-dir evaluation/evaluation_outputs/dep_search
```

**Step 4: Function Generation 轨迹生成**
```bash
python evaluation/benchmarks/hybrid_gym_func_gen/run_infer_no_image.py \
    --llm-config claude \
    --agent-class CodeActAgent \
    --eval-num-workers 3 \
    --eval-n-limit 500 \
    --max-iterations 30 \
    --eval-output-dir evaluation/evaluation_outputs/func_gen
```

#### 2.3 过滤成功轨迹并转换格式

```bash
# 对每个任务的输出进行压缩和转换
for TASK in func_localize issue_localize dep_search func_gen; do
    SUCCESS_FILE="evaluation/evaluation_outputs/${TASK}/output.jsonl"
    
    # 压缩成功轨迹
    poetry run python evaluation/combine_final_completions.py $SUCCESS_FILE
    
    # 转换为训练格式
    python evaluation/convert_data.py \
        ${SUCCESS_FILE%.jsonl}.with_completions.jsonl.gz
done
```

---

### Phase 3: 模型训练 (1-2 天)

#### 3.1 7B 模型训练（主要方案）

**训练框架选择：** 项目本身不包含训练代码，需要使用外部 SFT 框架。推荐使用 **LLaMA-Factory** 或 **OpenRLHF**。

**方案 A: 使用 LLaMA-Factory（推荐）**
```bash
# 安装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 准备训练数据（合并所有任务）
# 将 convert_data.py 输出的 JSONL 转换为 LLaMA-Factory 格式
```

**训练配置（7B, 8x4090）：**
```yaml
# configs/hybrid_gym_7b.yaml
model_name_or_path: Qwen/Qwen2.5-Coder-7B-Instruct
stage: sft
do_train: true
finetuning_type: full  # 全参数微调
deepspeed: configs/ds_z3_config.json  # DeepSpeed ZeRO-3

# 数据
dataset: hybrid_gym
template: qwen
cutoff_len: 8192
max_samples: 5000
overwrite_cache: true

# 训练超参（与论文一致）
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # 等效 batch_size = 8
learning_rate: 5.0e-5
num_train_epochs: 5
lr_scheduler_type: cosine
warmup_ratio: 0.1
gradient_checkpointing: true

# 输出
output_dir: outputs/hybrid_gym_7b
logging_steps: 10
save_strategy: steps
save_steps: 500
```

**DeepSpeed ZeRO-3 配置：**
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6
    },
    "bf16": {"enabled": true},
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0
}
```

**启动训练：**
```bash
cd LLaMA-Factory
deepspeed --num_gpus 8 src/train.py \
    configs/hybrid_gym_7b.yaml \
    --deepspeed configs/ds_z3_config.json
```

**预估训练时间：** ~4-6 小时（4470 条轨迹，5 epochs，8x4090 with ZeRO-3）

#### 3.2 32B 模型训练（挑战方案）

**32B 在 8x4090 上的可行性分析：**
- 模型参数：32B × 2 bytes (bf16) = 64 GB 模型权重
- 8 卡 × 24GB = 192GB 总 VRAM
- ZeRO-3 可以将模型权重分片到 8 卡：64GB / 8 = 8GB/卡
- 但 optimizer states (AdamW) 需要 2× 模型大小 = 128GB → 需要 offload 到 CPU
- 梯度：64GB → ZeRO-3 分片后 8GB/卡
- 激活值：需要 gradient checkpointing

**结论：** 32B 可以在 8x4090 上训练，但需要：
1. DeepSpeed ZeRO-3 + CPU offload optimizer
2. Gradient checkpointing
3. 小 batch size (1) + 大 gradient accumulation (16)
4. bf16 精度

**训练配置（32B, 8x4090）：**
```yaml
model_name_or_path: Qwen/Qwen2.5-Coder-32B-Instruct
stage: sft
do_train: true
finetuning_type: full
deepspeed: configs/ds_z3_cpu_offload_config.json

dataset: hybrid_gym
template: qwen
cutoff_len: 8192
max_samples: 5000

per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # 等效 batch_size = 16
learning_rate: 5.0e-5
num_train_epochs: 5
lr_scheduler_type: cosine
warmup_ratio: 0.1
gradient_checkpointing: true

output_dir: outputs/hybrid_gym_32b
logging_steps: 10
save_strategy: steps
save_steps: 500
```

**预估训练时间：** ~12-20 小时（4470 条轨迹，5 epochs，8x4090 with ZeRO-3 + CPU offload）

**替代方案：QLoRA（如果全参数训练太慢）**
```yaml
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_target: all
quantization_bit: 4  # 4-bit 量化
```
QLoRA 可以大幅降低 VRAM 需要，训练速度更快，但效果可能略差于全参数微调。

---

### Phase 4: 评估验证 (2-5 天)

#### 4.1 SWE-Bench Verified 评估

```bash
# 使用训练好的 7B 模型评估
python evaluation/benchmarks/swe_bench/run_infer.py \
    --agent-class CodeActAgent \
    --llm-model outputs/hybrid_gym_7b \
    --eval-num-workers 3 \
    --eval-n-limit 50 \
    --max-iterations 50 \
    --eval-output-dir evaluation/evaluation_outputs/swe_bench_7b

# 评估结果
python evaluation/benchmarks/swe_bench/eval_infer.py \
    --output-file evaluation/evaluation_outputs/swe_bench_7b/output.jsonl
```

#### 4.2 Hybrid-Gym 自评估

```bash
# 在 Hybrid-Gym 的 4 个任务上评估训练后的模型
for TASK in func_gen func_localize dep_search issue_localize; do
    python evaluation/benchmarks/hybrid_gym_${TASK}/run_infer*.py \
        --agent-class CodeActAgent \
        --llm-model outputs/hybrid_gym_7b \
        --eval-num-workers 3 \
        --eval-n-limit 100 \
        --max-iterations 30 \
        --eval-output-dir evaluation/evaluation_outputs/${TASK}_eval
done
```

#### 4.3 预期结果对标

| 指标 | 论文 (7B + Hybrid-Gym) | 你的目标 (首次复现) |
|------|----------------------|-------------------|
| SWE-Bench Verified Resolved | 15.0% | >10% |
| SWE-Bench Verified Localized | 62.4% | >50% |
| SWE-Bench Verified Non-Loop | 97.6% | >90% |

> **注意：** 首次复现不需要完全匹配论文结果。论文使用了完整的 4470 条轨迹和精确的训练超参搜索。首次复现重点是验证 pipeline 可以跑通。

---

## 三、关键风险与应对

### 3.1 VRAM 不足

**风险：** 7B 全参数 SFT 在 8x4090 上可能 OOM

**应对：**
1. 优先使用 DeepSpeed ZeRO-3（推荐）
2. 启用 gradient checkpointing
3. 减小 per_device_train_batch_size 到 1
4. 增大 gradient_accumulation_steps 保持等效 batch size
5. 如仍 OOM，使用 ZeRO-3 + CPU offload optimizer

### 3.2 API 成本超预算

**风险：** Teacher 轨迹生成 API 费用超出预期

**应对：**
1. 先用 10 条实例测试 pipeline，估算单条成本
2. 使用更便宜的 teacher（如 Qwen3-235B 替代 Claude）
3. 分批生成，每批 100 条，监控成本
4. 使用 local LLM（如果有足够 GPU）替代 API 调用

### 3.3 评估超时

**风险：** 某些评估实例超时（8 小时限制）

**应对：**
1. 减少 eval-n-limit，先评估小样本
2. 减少 max-iterations（从 30 降到 20）
3. 检查 Docker 容器资源是否充足

### 3.4 训练不收敛

**风险：** 训练后模型效果差

**应对：**
1. 检查训练数据格式是否正确
2. 检查 loss 曲线是否正常下降
3. 尝试不同学习率（5e-5 vs 1e-4）
4. 检查是否有数据质量问题（空 trajectory、格式错误）

---

## 四、时间线总览

| 阶段 | 任务 | 时间 | 依赖 |
|------|------|------|------|
| Phase 0 | 环境搭建 | 1-2 天 | - |
| Phase 1 | 数据预处理 | 1-3 天 | Phase 0 |
| Phase 2 | Teacher 轨迹生成 | 3-7 天 | Phase 1, API keys |
| Phase 3a | 7B 模型训练 | 0.5-1 天 | Phase 2 |
| Phase 3b | 32B 模型训练 | 1-2 天 | Phase 2 |
| Phase 4 | 评估验证 | 2-5 天 | Phase 3 |
| **总计** | | **8-20 天** | |

---

## 五、最小可行复现 (MVP)

如果时间/预算有限，可以做最小复现：

1. **只训练 7B 模型**（跳过 32B）
2. **每个任务只生成 200 条轨迹**（总计 ~800 条，API 成本 ~$100）
3. **只评估 SWE-Bench Verified 的前 50 个实例**
4. **使用 QLoRA 替代全参数微调**（更快，VRAM 更省）

MVP 时间线：**5-7 天**

---

## 六、检查清单

### Phase 0 完成标准
- [ ] Python 3.12 + Poetry 环境正常
- [ ] Docker 可以正常拉取和运行容器
- [ ] `poetry install` 成功
- [ ] API key 配置完成（OpenAI + Anthropic）

### Phase 1 完成标准
- [ ] Func-Localize 预处理数据生成（~1000 条）
- [ ] Dep-Search 预处理数据生成（~1000 条）
- [ ] 数据格式验证通过

### Phase 2 完成标准
- [ ] 至少 1 个任务的 teacher 轨迹生成成功
- [ ] 成功轨迹过滤和格式转换完成
- [ ] 训练数据合并完成（~500-2000 条）

### Phase 3 完成标准
- [ ] 7B 模型训练完成（loss 正常下降）
- [ ] 模型 checkpoint 保存成功

### Phase 4 完成标准
- [ ] SWE-Bench 评估 pipeline 跑通
- [ ] 至少有 1 个评估指标的结果
