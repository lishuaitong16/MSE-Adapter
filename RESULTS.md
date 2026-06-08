# Experiment Results

All experiments use seed `1111`. For configurations with multiple runs, the latest run is reported.

Metrics: MAE and Corr for regression tasks; Acc and Weighted-F1 for classification tasks. Lower MAE is better (↓); all other metrics are higher-is-better (↑).

---

## Regression: MOSEI

| Model | Variant | Has0-Acc2 | Has0-F1 | Non0-Acc2 | Non0-F1 | Acc5 | Acc7 | MAE ↓ | Corr ↑ |
|-------|---------|-----------|---------|-----------|---------|------|------|-------|--------|
| Llama2-7B | `cmcm` | 87.16 | 86.79 | 68.79 | 68.53 | 57.03 | 54.90 | 51.94 | 77.30 |
| Llama2-7B | `cmcm_reg` | 82.89 | 83.41 | 88.30 | 88.30 | 56.88 | 54.93 | **49.18** | **80.92** |
| Qwen-1.8B | `cmcm` | 82.85 | 82.45 | 73.23 | 73.55 | 50.85 | 49.20 | 59.34 | 69.59 |
| Qwen-1.8B | `cmcm_reg` | 82.23 | 82.78 | 87.01 | 87.04 | 57.46 | 55.74 | 50.78 | 79.92 |
| ChatGLM3-6B | `cmcm` | 86.91 | 86.67 | 71.90 | 72.00 | 56.99 | 54.69 | 51.84 | 78.53 |
| ChatGLM3-6B | `cmcm_reg` | 83.82 | 84.26 | 87.75 | 87.76 | 57.69 | 55.61 | 49.63 | **80.98** |

---

## Regression: SIMSV2

| Model | Variant | Acc2 | Acc2-weak | Acc3 | Acc5 | F1 | MAE ↓ | Corr ↑ | R² |
|-------|---------|------|-----------|------|------|----|-------|--------|----|
| Llama2-7B | `cmcm` | 77.08 | 70.19 | 69.92 | 49.71 | 76.83 | 36.45 | 58.62 | 18.11 |
| Llama2-7B | `cmcm_reg` | 80.46 | 71.01 | 75.15 | 54.45 | 80.55 | **31.56** | **70.15** | 40.54 |
| Qwen-1.8B | `cmcm` | 78.72 | 71.22 | 70.89 | 53.29 | 78.59 | 33.04 | 64.74 | 29.42 |
| Qwen-1.8B | `cmcm_reg` | 80.75 | 72.05 | 73.89 | 49.90 | 80.79 | 31.51 | 70.63 | **44.01** |
| ChatGLM3-6B | `cmcm` | 79.40 | 70.81 | 71.28 | 48.55 | 78.93 | **30.54** | **70.17** | 42.23 |
| ChatGLM3-6B | `cmcm_reg` | 80.46 | 72.05 | 73.98 | 51.16 | 80.46 | 31.64 | 70.74 | 43.93 |

---

## Classification: MELD

| Model | Variant | Acc ↑ | Weighted-F1 ↑ |
|-------|---------|-------|--------------|
| Llama2-7B | `cmcm` | 64.56 | 63.54 |
| Llama2-7B | `cmcm_cls` | 66.90 | **65.58** |
| Qwen-1.8B | `cmcm` | 63.68 | 62.17 |
| Qwen-1.8B | `cmcm_cls` | 64.06 | 62.70 |
| ChatGLM3-6B | `cmcm` | 60.77 | 60.21 |
| ChatGLM3-6B | `cmcm_cls` | 64.64 | 61.82 |

---

## Classification: CHERMA

| Model | Variant | Acc ↑ | Weighted-F1 ↑ |
|-------|---------|-------|--------------|
| Llama2-7B | `cmcm` | 72.74 | 72.66 |
| Llama2-7B | `cmcm_cls` | 72.67 | 72.43 |
| Qwen-1.8B | `cmcm` | 71.73 | 71.93 |
| Qwen-1.8B | `cmcm_cls` | 72.41 | **72.65** |
| ChatGLM3-6B | `cmcm` | 73.03 | **72.85** |
| ChatGLM3-6B | `cmcm_cls` | 68.87 | 68.80 |

---

## Summary

**`cmcm_reg` vs `cmcm` on regression tasks:**

The discriminative regression head (`cmcm_reg`) consistently outperforms the generative baseline (`cmcm`) on MOSEI across all three backbones, with MAE improvements of **2.21–8.56** and Corr improvements of **2.45–10.33**. On SIMSV2, `cmcm_reg` shows similar or better MAE for Llama2-7B and Qwen-1.8B, while the generative `cmcm` achieves the lowest SIMSV2 MAE (30.54) on ChatGLM3-6B.

**`cmcm_cls` vs `cmcm` on classification tasks:**

The discriminative classification head (`cmcm_cls`) consistently improves MELD Weighted-F1 for Llama2-7B (+2.04) and Qwen-1.8B (+0.53). On CHERMA, results are more mixed — `cmcm` with ChatGLM3-6B achieves the best overall F1 (72.85). The learnable CLS token design offers a simpler and more stable training signal than autoregressive generation for emotion classification.
