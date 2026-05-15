# Poet-mini 训练工程

Poet-mini 是一个面向中文古诗词生成任务的轻量级语言模型训练工程。本目录是可执行训练工程根目录，包含数据处理、tokenizer 构建、最小 GPT 训练、采样生成和容器化入口。

---

## 1. 运行环境

推荐在 Linux / WSL / Docker 环境中运行。

宿主机只需要准备：

- Docker
- WSL 或 Linux shell
- 如需 GPU 训练，需确保 Docker 支持 `--gpus all`

进入 `train` 目录：

```bash
cd /mnt/e/Poet-mini/train
```

---

## 2. 进入容器

普通容器：

```bash
bash makeshell
```

清理旧容器并重建：

```bash
bash makeshell -clean
```

GPU 容器：

```bash
bash makeshell -clean -gpu
```

进入容器后，默认工作目录为：

```text
/workspace/train
```

---

## 3. 准备训练环境

容器构建时会自动安装 `requirements.txt` 中声明的依赖，`build.sh` 只负责检查环境。

```bash
./build.sh
```

正常情况下会输出：

```text
Python: ...
Torch: ...
CUDA available: True 或 False
```

如果显示 `CUDA available: False`，训练会自动走 CPU；如需 GPU，请确认容器是通过 `bash makeshell -clean -gpu` 创建的，并且宿主机 Docker GPU 透传可用。

---

## 4. 数据目录结构

```text
data/
├── sample/          # 用户复制的原始样本数据，默认不提交 Git
├── interim/         # 清洗后的 JSONL 和纯文本语料，默认不提交 Git
└── processed/       # tokenizer 和 tokenized 数据
```

默认样本目录：

```text
data/sample/tangshi/
```

可放入：

```text
poet.tang.*.json
poet.song.*.json
唐诗三百首.json
唐诗补录.json
```

`data/sample/tangshi/error`、`authors.*.json` 和 `表面结构字.json` 默认不进入训练语料。

---

## 5. 构建训练数据

完整数据处理入口：

```bash
./build_data.sh
```

该命令会执行：

```text
scan -> build-jsonl -> build-text -> inspect
```

生成：

```text
data/interim/poems.jsonl
data/interim/poetry_corpus.txt
outputs/logs/corpus_scan.json
outputs/logs/corpus_stats.json
```

只执行 tokenizer 和 token 序列化：

```bash
./build_data.sh --steps train-tokenizer,prepare-tokenized
```

生成：

```text
data/processed/tokenizer.json
data/processed/train.bin
data/processed/val.bin
outputs/logs/tokenizer_stats.json
```

如需从头完整重建：

```bash
./build_data.sh --steps scan,build-jsonl,build-text,inspect,train-tokenizer,prepare-tokenized
```

---

## 6. 训练配置

模型结构配置：

```text
configs/model.yaml
```

训练流程配置：

```text
configs/train.yaml
```

常用训练参数位于 `configs/train.yaml`：

```yaml
train:
  device: auto
  batch_size: 32
  max_steps: 1000

runtime:
  eval_interval: 100
  log_interval: 10
  save_interval: 500
  sample_interval: 200
```

`device: auto` 会优先使用 CUDA；如果 CUDA 不可用，则使用 CPU。

---

## 7. 开始训练

默认训练：

```bash
./run.sh
```

指定运行 ID：

```bash
./run.sh --run-id cuda-test
```

指定配置：

```bash
./run.sh --model-config configs/model.yaml --config configs/train.yaml --run-id cuda-test
```

从 checkpoint 恢复：

```bash
./run.sh --run-id resume-test --resume outputs/checkpoints/latest.pt
```

训练输出：

```text
outputs/checkpoints/  # checkpoint
outputs/logs/         # metrics.jsonl
outputs/samples/      # 定期采样文本
```

---

## 8. 简单测试

### 8.1 冒烟训练

如果只是验证链路，建议先使用小步数：

```yaml
train:
  batch_size: 4
  max_steps: 10

runtime:
  eval_interval: 5
  eval_iters: 2
  log_interval: 1
  save_interval: 10
  sample_interval: 10
```

运行：

```bash
./run.sh --run-id smoke-test
```

检查是否生成：

```bash
ls -lh outputs/checkpoints
ls -lh outputs/logs
ls -lh outputs/samples
```

### 8.2 查看训练指标

```bash
tail -n 50 outputs/logs/metrics.jsonl
```

重点检查：

- loss 是否为正常数字
- 是否出现 `nan`
- step 是否持续增长
- 是否写入 train / val 指标

### 8.3 查看生成样例

```bash
cat outputs/samples/step_000010.txt
```

早期训练样例质量较差是正常现象，主要看是否能正常解码、是否出现大量 `<UNK>` 或乱码。

### 8.4 手动采样

```bash
python -m poet_mini.generation.sample \
  --checkpoint outputs/checkpoints/latest.pt \
  --prompt $'<BOS>标题：秋夜\n作者：李白\n' \
  --max-new-tokens 120 \
  --output outputs/samples/manual_sample.txt
```

查看输出：

```bash
cat outputs/samples/manual_sample.txt
```

注意：在 shell 中传入换行 prompt 时，推荐使用 `$'...'` 格式。

---

## 9. 工程结构

```text
poet_mini/
├── data/
│   └── token_dataset.py   # token bin 读取和 batch 构造
├── model/
│   └── gpt.py             # 最小 GPT 模型结构
├── training/
│   └── train.py           # 训练循环、评估、checkpoint、metrics
└── generation/
    └── sample.py          # checkpoint 采样生成
```

训练链路：

```text
data/processed/train.bin + val.bin
  -> poet_mini.data.TokenBatcher
  -> poet_mini.model.GPT
  -> poet_mini.training.train
  -> checkpoint / metrics / samples
```

生成链路：

```text
checkpoint + tokenizer.json + prompt
  -> poet_mini.generation.sample
  -> outputs/samples/*.txt
```

---

## 10. 常见问题

### CUDA 不可用

检查：

```bash
./build.sh
```

如果显示：

```text
CUDA available: False
```

请确认：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

如果 Docker GPU 可用，再重新创建 GPU 容器：

```bash
bash makeshell -clean -gpu
```

### prompt 中出现 `<UNK>`

手动采样时不要直接复制含 `\n` 的普通字符串，使用：

```bash
--prompt $'<BOS>标题：秋夜\n作者：李白\n'
```

### checkpoint / 日志不要提交

以下目录默认不纳入 Git：

```text
outputs/checkpoints/
outputs/logs/
outputs/samples/
data/interim/
data/processed/train.bin
data/processed/val.bin
```

---

## 11. 参考项目和文献

### 参考项目

- `chinese-poetry`：中文古诗词开源数据集，Poet-mini 当前语料来源。[https://github.com/chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- `nanoGPT`：极简 GPT 训练工程，Poet-mini 的最小训练闭环、`train.bin` / `val.bin` 数据组织和自回归训练流程主要参考其工程思路。[https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- `LitGPT`：LLM 训练、微调、评估和部署工作流项目，Poet-mini 的 YAML 配置分层和后续训练工作流规划参考其 recipe 思路。[https://github.com/Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt)
- `RL-LocalServer`：本地强化学习训练和可视化框架，Poet-mini 的容器入口、脚本契约和后续 Dashboard 观测思路参考该项目。[https://github.com/Achbite/RL-LocalServer](https://github.com/Achbite/RL-LocalServer)
- `DeepSeek-V3`：DeepSeek-V3 模型开源仓库，Poet-mini 后续分布式参数、训练阶段划分和大模型工程边界参考其公开说明。
  [https://github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)

### 参考文献

- Vaswani et al., 2017. **Attention Is All You Need**.Transformer 架构原始论文，Poet-mini 的 GPT 模型结构基于 causal self-attention 思路。[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Radford et al., 2019. **Language Models are Unsupervised Multitask Learners**.GPT-2 技术报告，自回归语言模型、next-token prediction 和生成式预训练的重要参考。[https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Brown et al., 2020. **Language Models are Few-Shot Learners**.GPT-3 论文，展示自回归语言模型规模化后的 few-shot 能力。[https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
- DeepSeek-AI, 2024. **DeepSeek-V3 Technical Report**.大规模 MoE 语言模型训练、预训练 / SFT / RL 阶段划分和训练稳定性参考。[https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek-AI, 2025. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**.
  后训练、强化学习激发推理能力和能力蒸馏方向的后续参考。
  [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
