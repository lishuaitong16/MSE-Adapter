# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境

- Conda 环境：`MSE-Adapter`（所有命令都在此环境下运行）
- GPU：8 × RTX 5090（sm_120），需要 PyTorch 2.8.0+cu128
- **transformers 版本必须与 `requirements.txt` 完全一致，禁止升降级**（作者要求，版本不一致会导致评估指标异常）
- 激活：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate MSE-Adapter`（shell 脚本中需显式激活）

---

## 项目结构

三个模型目录（`MSE-Llama2-7B` / `MSE-Qwen-1.8B` / `MSE-ChatGLM3-6B`）结构完全相同：

```
├── run.py                  # 入口：解析参数 → 循环数据集 → 调用训练/测试
├── run_all.sh              # 后台同时启动回归和分类进程
├── visualize.py            # glob logs/cmcm-*.log → 绘制训练曲线
├── config/
│   ├── config_regression.py    # MOSI/MOSEI/SIMSV2 配置
│   └── config_classification.py # MELD/CHERMA/IEMOCAP 配置
├── models/
│   ├── AMIO.py             # 模型注册表（按 modelName 路由）
│   ├── multiTask/CMCM.py   # 核心模型：LLM + 音视频 LSTM + Text_guide_mixer + mutli_scale_fusion
│   └── subNets/Textmodel.py # LLM 封装：soft prompt 注入、generate、input_processing
├── trains/
│   ├── ATIO.py             # 训练器注册表
│   └── multiTask/CMCM.py   # 训练逻辑：AdamW + cosine warmup + early stop + AMP
├── utils/
│   ├── functions.py        # Storage 类（dict-like args）、dict_to_str
│   └── metricsTop.py       # 各数据集评估指标（回归用 MAE，分类用 weight_F1）
└── results/
    ├── models/             # 最优 checkpoint（只含可训练参数，约 10MB）
    └── results/            # 测试集指标 CSV（多次运行追加）
```

---

## 模型数据流（关键架构）

```
text[B, 3, seq_len]  audio[B, seq_len, 64]  video[B, seq_len, 64]
        ↓                     ↓                     ↓
  text_embedding         audio_LSTM            video_LSTM
  [B, seq_len, H]       [B, 256]              [B, 256]
        ↓                      \               /
   (冻结 LLM)             Text_guide_mixer
                          (text-guided cross-modal fusion)
                                ↓ [B, 256]
                         mutli_scale_fusion
                          (3-scale MLP + conv integrating)
                                ↓ [B, pseudo_tokens, H]
                    cat([fusion_h, text], dim=1) → LLM forward
                                ↓
                           CrossEntropy Loss (训练) / generate (推理)
```

- **冻结**：LLM 所有参数（`requires_grad=False`）
- **可训练**：audio_LSTM、video_LSTM、Text_guide_mixer、mutli_scale_fusion（共约 2.6M 参数）
- `pseudo_tokens=4`：软 prompt token 数，即 fusion 模块输出序列长度
- `feature_dims[0]` 必须与 LLM hidden size 一致（Qwen-1.8B=2048，Llama-2-7B=4096，ChatGLM3-6B=4096）

---

## 数据集路径

数据根目录：`/data/lishuaitong/data/emotion/`

| 数据集 | 实际路径 | 任务 | KeyEval |
|--------|---------|------|---------|
| SIMSV2 | `SIMSV2_processed/ch-simsv2s.pkl` | 回归 | MAE（越小越好）|
| MOSEI  | `MOSEI_processed/unaligned_50.pkl` | 回归 | MAE |
| MELD   | `MELD_processed/` | 分类 | weight_F1（越大越好）|
| CHERMA | `CHERMA0723_processed/` | 分类 | weight_F1 |

config 里最常见的错误：目录名多/少 `_processed` 后缀。

---

## 启动训练

```bash
cd /home/lishuaitong/MSE-Adapter/MSE-Llama2-7B  # 或其他模型目录
bash run_all.sh
# 回归（simsv2, mosei）和分类（meld, cherma）分别在两张 GPU 上后台运行
```

监控：
```bash
tail -f logs/run_regression.log      # stdout/stderr，含报错
tail -f logs/run_classification.log
tail -f logs/cmcm-meld.log           # Python logging 结构化日志（每数据集单独文件）
```

可视化：
```bash
conda run -n MSE-Adapter python visualize.py
# 读取 logs/cmcm-*.log，输出 logs/training_curves.png
```

GPU 等待自动启动（等显存低于阈值时运行某模型）：
```bash
# 修改 wait_and_launch_chatglm.sh 中的 THRESHOLD（单位 MB）
nohup bash /home/lishuaitong/MSE-Adapter/wait_and_launch_chatglm.sh &
```

---

## 新模型接入清单

| 步骤 | 文件 | 修改内容 |
|------|------|---------|
| 1 | `run.py` | `--pretrain_LM` 改为本地权重路径 |
| 2 | `run.py` | `--root_dataset_dir` 确认正确 |
| 3 | `config_regression.py` `config_classification.py` | `dataPath` 子目录名对照上表 |
| 4 | `config_regression.py` `config_classification.py` | `feature_dims[0]` 改为对应 LLM hidden size |
| 5 | `run_all.sh` | `--gpu_ids` 改为目标 GPU 编号 |

---

## 日志系统

- `set_log(args)` 每次调用会向 root logger 添加新 Handler。**必须在循环里调用**（每个数据集调用一次），不能在 `__main__` 顶层调用。
- 清理旧 Handler 时必须用 `for ph in logger.handlers[:]:`（带 `[:]` 拷贝），否则 Python 列表迭代中删除元素会跳过元素。
- 每数据集生成独立文件 `logs/cmcm-{datasetName}.log`；`visualize.py` 用 `glob('logs/cmcm-*.log')` 合并读取。

---

## 已知问题与修复记录

**ChatGLM3 tokenizer 兼容性**  
`models/ChatGLM3/tokenization_chatglm.py` 中 `eos_token`/`pad_token`/`unk_token` 是只读 `@property`，transformers 的 `PreTrainedTokenizerBase.__init__` 会尝试 `setattr` 从而报 `AttributeError: can't set attribute`。  
修复：用模型权重目录中的 `tokenization_chatglm.py` 替换（该版本有无操作的 setter），**不要改 transformers 版本**。

**OOM 排查**  
ChatGLM3-6B 加载后约占 12GB 显存，加上训练 batch 需要 20GB+。若 GPU 有其他进程占用超过 ~11GB，则会 OOM。`wait_and_launch_chatglm.sh` 的 `THRESHOLD` 设为 11000（MB）可避免此问题。

---

## 输出文件

| 文件 | 内容 |
|------|------|
| `results/models/{modelName}-{dataset}-{mode}.pth` | 验证集最优 checkpoint（只含可训练参数）|
| `results/results/{dataset}-{mode}-{warmup}.csv` | 测试集指标，多次运行追加 |
| `logs/cmcm-{datasetName}.log` | 结构化训练日志，供 visualize.py 解析 |
| `logs/run_regression.log` | 回归进程完整输出，含报错 |
| `logs/run_classification.log` | 分类进程完整输出，含报错 |
