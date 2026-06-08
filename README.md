<div align="center">

# MSE-Adapter

**A Lightweight Plugin Endowing LLMs with the Capability to Perform Multimodal Sentiment Analysis and Emotion Recognition**

国科大情感计算课程大作业

<p align="center">
  <a href="https://arxiv.org/abs/2502.12478"><img src="https://img.shields.io/badge/arXiv-2502.12478-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/34755"><img src="https://img.shields.io/badge/AAAI-2025-003973?style=flat-square" alt="AAAI 2025"></a>
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square">
  <img src="https://img.shields.io/badge/PyTorch-2.8.0%2Bcu128-orange?style=flat-square">
</p>

</div>

---

## 简介

本仓库是论文 [MSE-Adapter: A Lightweight Plugin Endowing LLMs with the Capability to Perform Multimodal Sentiment Analysis and Emotion Recognition](https://arxiv.org/abs/2502.12478)（AAAI 2025）的课程复现与改进实验。

在原论文基础上，本仓库对以下三种 LLM 骨干进行了适配与实验：

| 目录 | 骨干模型 | 参数量 |
|------|---------|--------|
| `MSE-Llama2-7B` | Llama-2-7B-Chat | 7B |
| `MSE-Qwen-1.8B` | Qwen-1.8B | 1.8B |
| `MSE-ChatGLM3-6B` | ChatGLM3-6B | 6B |

---

## 相比原论文的改进

原论文的 MSE-Adapter 采用**生成式**范式：将融合后的多模态特征作为软提示注入 LLM，由 LLM 自回归生成情感标签文本。

本仓库在此基础上引入了**判别式**变体，以更适配情感识别（Emotion Recognition）任务：

- **新增可学习 CLS token**：在 LLM 输入序列末尾追加一个随机初始化的可学习 token `[CLS]`，经 LLM 前向传播后提取其对应的隐藏状态作为序列表示。

- **新增分类头（`cmcm_cls`）**：在 CLS token 的隐藏状态上接两层 MLP，将隐藏维度（如 4096）映射至情绪类别数（MELD/CHERMA 为 7 类），以 CrossEntropy Loss 监督训练，完全绕过自回归生成。

- **新增回归头（`cmcm_reg`）**：同样基于 CLS token，将隐藏状态映射至标量，以 L1 Loss 训练，用于情感强度回归任务。

```
# cmcm_cls / cmcm_reg 新增结构（models/multiTask/CMCM.py）
self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))   # 可学习 token

self.cls_head = nn.Sequential(          # 分类头（cmcm_cls）
    nn.Linear(hidden_size, hidden_size // 2),
    nn.GELU(),
    nn.Linear(hidden_size // 2, num_classes)   # 7 类情绪
)

self.reg_head = nn.Sequential(          # 回归头（cmcm_reg）
    nn.Linear(hidden_size, hidden_size // 2),
    nn.GELU(),
    nn.Linear(hidden_size // 2, 1)
)
```

---

## 模型架构

MSE-Adapter 以**冻结的 LLM** 为骨干，通过一个轻量适配模块（约 2.6M 可训练参数）注入音视频模态信息：

```
text [B, seq, H]   audio [B, seq, 64]   video [B, seq, 64]
       ↓                   ↓                    ↓
  text_embedding       audio_LSTM           video_LSTM
  (frozen LLM)          [B, 256]             [B, 256]
       ↓                      \              /
  (冻结不参与训练)         Text_guide_mixer
                         (文本引导的跨模态融合)
                               ↓ [B, 256]
                        mutli_scale_fusion
                         (3 尺度 MLP + Conv 集成)
                               ↓ [B, 4, H]   ← 4 个软提示 token
                   cat([fusion, text]) → LLM forward
                               ↓
                    回归头 / 分类头 / 自回归生成
```

**三种模型变体：**

| 变体 | 来源 | 输出方式 | 适用任务 |
|------|------|---------|---------|
| `cmcm` | 原论文 | LLM 自回归生成文本标签 | 情感回归 / 分类 |
| `cmcm_cls` | **本仓库新增** | CLS token → 分类头（7类） | 情绪识别 |
| `cmcm_reg` | **本仓库新增** | CLS token → 回归头（标量） | 情感强度回归 |

---

## 数据集

| 数据集 | 语言 | 任务 | 评估指标 |
|--------|------|------|---------|
| [MOSEI](https://huggingface.co/datasets/AZYoung/MOSEI_processed) | 英文 | 情感回归 | MAE ↓ |
| [SIMSV2](https://huggingface.co/datasets/AZYoung/SIMSV2_processed) | 中文 | 情感回归 | MAE ↓ |
| [MELD](https://huggingface.co/datasets/AZYoung/MELD_processed) | 英文 | 情绪分类 | Weighted-F1 ↑ |
| [CHERMA](https://huggingface.co/datasets/AZYoung/CHERMA0723_processed) | 中文 | 情绪分类 | Weighted-F1 ↑ |

---

## 环境配置

```bash
git clone https://github.com/lishuaitong16/MSE-Adapter.git
cd MSE-Adapter

conda create -n MSE-Adapter python=3.10.13
conda activate MSE-Adapter
pip install -r requirements.txt
```

> **注意**：`transformers` 版本必须与 `requirements.txt` 完全一致，版本不一致会导致评估指标异常。

---

## 快速开始

以 `MSE-Llama2-7B` 为例：

**1. 配置路径**

编辑 `MSE-Llama2-7B/run.py` 中的两个路径参数：
```python
# 数据集根目录（包含 MOSEI_processed/ SIMSV2_processed/ 等子目录）
--root_dataset_dir  /path/to/your/data

# 预训练 LLM 权重目录
--pretrain_LM       /path/to/Llama-2-7b-chat-hf
```

**2. 启动训练**

```bash
cd MSE-Llama2-7B
bash run_all.sh          # 同时启动回归（MOSEI/SIMSV2）和分类（MELD/CHERMA）
```

或单独运行：
```bash
python run.py --task_type regression --modelName cmcm_reg --gpu_ids [0]
python run.py --task_type classification --modelName cmcm_cls --gpu_ids [1]
```

**3. 监控训练**

```bash
tail -f logs/run_regression.log       # 回归进程输出
tail -f logs/run_classification.log   # 分类进程输出
tail -f logs/cmcm-mosei.log           # 单数据集结构化日志
```

**4. 可视化**

```bash
python visualize.py   # 读取 logs/cmcm-*.log，输出 logs/training_curves.png
```

---

## 项目结构

```
MSE-{Model}/
├── run.py                        # 入口：参数解析 → 循环数据集 → 训练/测试
├── run_all.sh                    # 并行启动回归与分类进程
├── visualize.py                  # 训练曲线可视化
├── config/
│   ├── config_regression.py      # MOSEI / SIMSV2 超参数
│   └── config_classification.py  # MELD / CHERMA 超参数
├── models/
│   ├── AMIO.py                   # 模型注册表
│   ├── multiTask/CMCM.py         # 核心模型
│   └── subNets/Textmodel.py      # LLM 封装（soft prompt 注入）
├── trains/
│   ├── ATIO.py                   # 训练器注册表
│   └── multiTask/CMCM.py         # 训练逻辑（AdamW + cosine warmup + AMP）
└── utils/
    ├── functions.py              # Storage、dict_to_str
    └── metricsTop.py             # 各数据集评估指标
```

---

## 致谢

- 原论文与官方实现：[AZYoung/MSE-Adapter](https://github.com/AZYoung233/MSE-Adapter)
- 代码结构参考：[SELF-MM](https://github.com/thuiar/Self-MM)
