

Common Beginner Questions

Q: What does BLEU 28.4 actually sound like in translation quality?

BLEU < 10  → almost useless, major errors everywhere
BLEU 10-20 → understandable but rough
BLEU 20-30 → good quality, some errors
BLEU 30-40 → very good, near human quality
BLEU > 40  → excellent, approaching human level

28.4 on English-German = very good translation with minor errors.

Q: Why doesn't the Transformer report EN-FR FLOPs?
A: The paper only reports EN-DE FLOPs for the base model — likely because EN-FR used a different dataset size and the comparison was less direct. The big model EN-FR cost is shown.

Q: Is higher BLEU always better?
A: Generally yes for comparison purposes, but BLEU has known limitations — it measures n-gram overlap, not actual meaning. A translation can score high BLEU while missing the nuance of the original. Newer metrics like COMET have since improved on BLEU.

Q: What happened after this paper?

2018: BERT (bidirectional Transformer for language understanding)
2019: GPT-2 (large generative Transformer)
2020: GPT-3 (175B parameter Transformer)
2022: ChatGPT (instruction-tuned Transformer)
2024: GPT-4, Claude 3, Gemini (massive Transformers)
All of them: built on this exact architecture.
One Line Summary

Transformer big model achieves 28.4 BLEU on EN-DE (new state of the art) and 41.0 on EN-FR (matches best ensemble) — as a single model — at 3-52× lower training cost than all competitors

---
# BLEU Score Interpretation Guide

The **Bilingual Evaluation Understudy (BLEU)** score is a widely used metric for evaluating the quality of machine-translated text. It measures how closely a machine-generated translation matches one or more human reference translations.

BLEU scores range from **0 to 100**, where **higher scores indicate translations that are more fluent, accurate, and closer to human quality**.

---

# BLEU Score Interpretation Scale

| BLEU Score | Translation Quality | Interpretation |
|------------|---------------------|----------------|
| **< 10** | **Poor / Unusable** | The translation is largely incorrect, unintelligible, or fails to preserve the original meaning. |
| **10 – 19** | **Hard to Understand** | Some words or phrases may be correct, but the overall meaning is difficult to understand and requires extensive editing. |
| **20 – 29** | **Gist is Clear** | The general meaning can be inferred, but the translation contains significant grammatical and lexical errors. |
| **30 – 39** | **Good / Understandable** | The translation is generally clear and understandable, though noticeable errors and awkward phrasing remain. |
| **40 – 49** | **High Quality** | The translation is fluent, accurate, and comparable to the performance of strong machine translation systems. |
| **50 – 59** | **Very High Quality** | The translation reads naturally and is nearly indistinguishable from a typical human translation in most cases. |
| **> 60** | **Exceptional / Dataset-Specific** | Such scores are uncommon in real-world translation tasks. They often indicate evaluation on highly similar data or potential overfitting rather than true human-level generalization. |

---

# Key Takeaways

- **Higher BLEU scores indicate closer agreement with human reference translations.**
- BLEU evaluates **n-gram overlap** between the generated translation and one or more reference translations.
- While BLEU is useful for benchmarking translation systems, it **does not directly measure semantic understanding or contextual correctness**.
- Modern evaluation methods such as **BERTScore**, **COMET**, and **BLEURT** are often used alongside BLEU to provide a more comprehensive assessment of translation quality.

---

## Summary

BLEU is one of the most widely adopted automatic evaluation metrics for machine translation. Although it provides a useful quantitative measure of translation quality, it should be interpreted alongside qualitative analysis and newer semantic evaluation metrics, as a high BLEU score does not always guarantee a better translation from a human perspective.

---
# Base Transformer Hyperparameters
 
The original Transformer model proposed in the *Attention Is All You Need* paper is defined by a set of carefully chosen hyperparameters. Each hyperparameter plays a specific role in balancing model capacity, computational efficiency, and training stability.
 
## Core Architecture Hyperparameters
 
| Hyperparameter | Value | Description & Justification |
|---|---:|---|
| **Number of Layers (N)** | 6 | Six identical layers are stacked in both the encoder and decoder. This depth provides sufficient representational power while maintaining reasonable computational cost and training time. |
| **Model Dimension (d_model)** | 512 | The dimensionality of token embeddings and the hidden representations throughout the model. It serves as the standard feature size across all sublayers and enables residual connections since every sublayer produces vectors of the same dimension. |
| **Feed-Forward Dimension (d_ff)** | 2048 | The hidden dimension of the Position-Wise Feed-Forward Network (FFN). Each token representation is expanded from 512 to 2048 dimensions, processed with a non-linear activation, and then projected back to 512. This larger intermediate space increases the model's expressive capacity. |
| **Number of Attention Heads (h)** | 8 | Multi-head attention splits the representation into eight parallel attention heads, allowing the model to learn different types of relationships (syntactic, semantic, positional, etc.) simultaneously. |
| **Key Dimension (d_k)** | 64 | Dimensionality of the query and key vectors for each attention head. Since d_k = d_model / h = 512 / 8 = 64, each head operates on a lower-dimensional representation, keeping computation efficient. |
| **Value Dimension (d_v)** | 64 | Dimensionality of the value vectors for each attention head. Keeping d_v = 64 allows all heads to operate independently while maintaining the overall output dimension after concatenation. |
 
## Regularization & Optimization Hyperparameters
 
| Hyperparameter | Value | Description & Justification |
|---|---:|---|
| **Dropout Rate (P_drop)** | 0.1 | Dropout is applied to attention weights, residual connections, embeddings, and feed-forward layers. It reduces overfitting by randomly deactivating neurons during training, improving generalization. |
| **Label Smoothing (ε_ls)** | 0.1 | Instead of assigning a probability of exactly 1 to the correct token, label smoothing distributes a small amount of probability across other classes. This prevents the model from becoming overconfident, slightly increases perplexity, but generally improves BLEU scores and overall translation quality. |
| **Warmup Steps** | 4000 | During the first 4000 training steps, the learning rate increases linearly before following the inverse square root decay schedule. This warmup phase stabilizes optimization and prevents large parameter updates early in training. |
| **Optimizer** | Adam | The Transformer uses the Adam optimizer with β₁ = 0.9, β₂ = 0.98, and ε = 10⁻⁹. Adam adapts the learning rate for each parameter individually, resulting in faster and more stable convergence during training. |
 
## Default Base Transformer Configuration
 
| Component | Value |
|---|---:|
| Encoder Layers | 6 |
| Decoder Layers | 6 |
| Model Dimension (d_model) | 512 |
| Feed-Forward Dimension (d_ff) | 2048 |
| Attention Heads | 8 |
| Key Dimension (d_k) | 64 |
| Value Dimension (d_v) | 64 |
| Dropout | 0.1 |
| Label Smoothing | 0.1 |
| Warmup Steps | 4000 |
| Optimizer | Adam (β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹) |
 
## Key Takeaways
 
- The Transformer uses **6 encoder layers** and **6 decoder layers**, providing a balance between model capacity and computational efficiency.
- A **512-dimensional hidden representation** (d_model) is maintained throughout the architecture, simplifying residual connections.
- The **2048-dimensional feed-forward network** provides additional non-linear modeling capacity through an expand–process–compress pipeline.
- **Eight attention heads** enable the model to simultaneously learn multiple types of relationships within the input sequence.
- Splitting the model dimension into **64-dimensional keys, queries, and values** ensures that multi-head attention has approximately the same computational cost as a single large attention head.
- Regularization techniques such as **dropout** and **label smoothing** improve the model's ability to generalize to unseen data.
- The **learning rate warmup schedule** and the **Adam optimizer** are critical for stable and efficient training of deep Transformer models.
## Summary
 
The original Transformer base model is built around a carefully balanced set of hyperparameters that maximize performance while maintaining computational efficiency. These architectural choices — including six-layer encoder and decoder stacks, 512-dimensional representations, eight attention heads, and 2048-dimensional feed-forward networks — have become the foundation for many modern Transformer-based architectures such as BERT, GPT, T5, and ViT.

---
---

*Previous: 6.1 Machine Translation Details*  
*Next: 6.3 English Constituency Parsing (Generalization)*25.8** |
| 0.2 | 5.47 | 25.7 |

```
εls = 0.0: better PPL (4.67!) but worse BLEU (25.3) ❌
           model is very confident but that confidence = overfit predictions
           
εls = 0.1: slightly worse PPL but BEST BLEU ✅
           confirms: hurts perplexity, improves translation quality
           
εls = 0.2: too much smoothing → model too uncertain → drops off
```

This perfectly confirms what was stated in 5.4:
**Label smoothing trades perplexity for BLEU — and that's the right trade.**

---

## Row (E) — Sinusoidal vs Learned Positional Encoding

**Question: Does the choice of positional encoding matter?**

| Positional Encoding | PPL | BLEU |
|---|---|---|
| **Sinusoidal (base)** | **4.92** | **25.8** |
| Learned embeddings | 4.92 | 25.7 |

```
Difference: 0.1 BLEU — essentially identical ✅
```

**Conclusion:** Both approaches work equally well on this task.

Sinusoidal chosen because:
```
1. Zero extra parameters ✅
2. Generalizes to longer sequences than seen in training ✅
3. Same performance ✅
```

---

## The Big Model — Final Row

```
N=6, dmodel=1024, dff=4096, h=16
Pdrop=0.3, 300K steps
→ PPL=4.33, BLEU=26.4, params=213M
```

Compare to base:
```
Base:  PPL=4.92, BLEU=25.8, params=65M,  100K steps
Big:   PPL=4.33, BLEU=26.4, params=213M, 300K steps

Improvement: +0.6 BLEU at cost of 3.3× more parameters + 3× more training
```

The big model uses **higher dropout (0.3 vs 0.1)** because:
```
Bigger model → more parameters → higher risk of overfitting
→ needs stronger regularization → higher dropout rate
```

---

## Key Lessons From the Ablation Study

| What was tested | Finding |
|---|---|
| Attention heads | h=8 is sweet spot. Too few OR too many hurts. |
| Key dimension dk | Larger dk helps — compatibility needs space |
| Number of layers N | More layers = better, with diminishing returns |
| Model dimension dmodel | Larger = better, but expensive |
| FFN dimension dff | Larger = better, moderate effect |
| Dropout | Critical — removing it costs 1.2 BLEU |
| Label smoothing | Hurts PPL, improves BLEU — worth the tradeoff |
| Positional encoding | Sinusoidal = learned, choose sinusoidal for free wins |

---

## One Line Summary

> Every design choice in the base Transformer was empirically justified — ablation shows removing or changing any component hurts performance, with h=8 heads, dmodel=512, dff=2048, Pdrop=0.1, εls=0.1 as the validated sweet spots

---