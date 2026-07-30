---

# 6.1 Machine Translation — The Details Behind the Results
> **Transformer From Scratch — Tutorial Series**  
> Section: `07_summary/key_takeaways.md`

---

## What This Section Is About

Section 6 gave us the numbers. Section 6.1 explains **exactly how those numbers were achieved** —
the inference tricks, checkpoint averaging, beam search, and how FLOPs were calculated.

These details matter because results without methodology are meaningless.

---

## English-German Results — The Headline

```
Task:             WMT 2014 English → German
Transformer big:  28.4 BLEU  (NEW STATE OF THE ART)
Margin:           +2.0 BLEU over best previous model (including ensembles)
Training time:    3.5 days on 8× P100 GPUs
```

And the impressive bonus:

```
Even the BASE model surpasses ALL previously published models and ensembles
at a FRACTION of the training cost of any competitor
```

This means the cheaper, smaller version of their model was already winning —
the big model just extended the lead further.

---

## English-French Results

```
Task:             WMT 2014 English → French
Transformer big:  41.0 BLEU
Cost:             Less than 1/4 the training cost of previous state-of-the-art
Dropout:          Pdrop = 0.1 (different from EN-DE which used 0.3)
```

**Why different dropout for EN-FR?**

```
EN-FR dataset: 36 million sentences (much larger)
EN-DE dataset: 4.5 million sentences (smaller)

Larger dataset → less risk of overfitting → less dropout needed
0.1 instead of 0.3 → less regularization → model can use its full capacity
```

---

## Inference Tricks — How Final Models Were Built

### Checkpoint Averaging 🔷

During training, the model is saved at regular intervals — these saves are called **checkpoints.**

```
Base model:  last 5 checkpoints averaged  (saved every 10 minutes)
Big model:   last 20 checkpoints averaged
```

**What is a checkpoint?**

Think of training like climbing a mountain path:

```
Step 90,000:  model weights = checkpoint A
Step 92,000:  model weights = checkpoint B
Step 94,000:  model weights = checkpoint C
Step 96,000:  model weights = checkpoint D
Step 98,000:  model weights = checkpoint E
Step 100,000: model weights = checkpoint F (final)
```

Instead of using only the final checkpoint, they **average the weights** across the last 5:

```
Final weights = (A + B + C + D + E) / 5
```

**Why does this help?**

Late in training, model weights **oscillate** around the optimal point:

```
Optimal point
      ↓
  ─── * ───
 /  ↑   ↑  \       ← weights bounce around optimal, never settle perfectly
/   A   B   \
     C D E
```

Averaging smooths out the oscillations and lands **closer to the true optimum**
than any single checkpoint alone.

> Real life analogy — **stock market averaging** 📈  
> Instead of selling at one specific day's price,
> you average over the last 5 days to get a more stable, representative value.

---

### Beam Search 🔷

```
Beam size:     4
Length penalty: α = 0.6
Max output length: input length + 50
Early termination: yes, when possible
```

**What is Greedy Decoding? (the naive approach)**

The simplest way to generate translation — at each step, pick the **single most probable word:**

```
Step 1: pick highest prob word → "मैं"    (prob 0.82)
Step 2: pick highest prob word → "खुश"   (prob 0.71)
Step 3: pick highest prob word → "था"    (prob 0.65)

Result: "मैं खुश था"
```

Problem — choosing the best word at step 1 might **prevent a better overall sentence:**

```
Step 1: "मैं" (prob 0.82) seems best
        BUT choosing "वह" (prob 0.40) at step 1
        might lead to a much better sentence overall ❌
```

Greedy decoding is locally optimal but globally suboptimal.

---

**What is Beam Search?**

Instead of tracking ONE candidate translation, track the **top k candidates simultaneously:**

```
Beam size = 4 → keep top 4 candidate translations at every step

Step 1 candidates:
  Beam 1: "मैं"   (score: 0.82)
  Beam 2: "वह"   (score: 0.40)
  Beam 3: "तुम"  (score: 0.30)
  Beam 4: "हम"   (score: 0.25)

Step 2 — expand each beam, keep top 4 overall:
  Beam 1: "मैं खुश"    (score: 0.82 × 0.71 = 0.58)
  Beam 2: "मैं था"     (score: 0.82 × 0.65 = 0.53)
  Beam 3: "वह खुश"    (score: 0.40 × 0.80 = 0.32)
  Beam 4: "वह है"     (score: 0.40 × 0.72 = 0.29)

...continue until [END] token
Final: pick the beam with highest total score
```

> Real life analogy — **chess engine thinking ahead** ♟️  
> A greedy player picks the best move right now.  
> A smart player (beam search) explores the top 4 moves,  
> thinks several steps ahead for each, then picks the best overall plan.

**Beam size tradeoff:**
```
Beam size = 1  → same as greedy (fast but worse quality)
Beam size = 4  → good balance (used here) ✅
Beam size = 100 → very slow, diminishing returns
```

---

**What is Length Penalty (α = 0.6)?** 🔷

Without a length penalty, beam search **prefers shorter translations** because:

```
Each word multiplies the probability score (numbers < 1):
"मैं खुश हूँ" = 0.82 × 0.71 × 0.65 = 0.379
"मैं खुश"    = 0.82 × 0.71         = 0.582   ← higher score but wrong! ❌
```

Shorter sequences have fewer multiplications → higher scores → model prefers them.

Length penalty divides the score by a function of length, **normalizing for sequence length:**

```
α = 0.6 → moderate penalty for shortness

Score = log_probability / length^α

This ensures:
  - Too short translations are penalized ✅
  - Too long translations are also penalized ✅  
  - Natural length translations score best ✅
```

---

**Maximum Output Length = Input Length + 50**

```
Input: "I am very happy today" (5 tokens)
Max output allowed: 5 + 50 = 55 tokens
```

This prevents the decoder from generating infinitely long outputs.
The +50 buffer handles cases where the target language naturally uses more words.

**Early Termination** — if the model generates an [END] token before hitting the limit,
it stops immediately. No need to generate all 55 tokens if the translation is complete.

---

## How FLOPs Were Calculated

The paper explains:

```
FLOPs = training_time × number_of_GPUs × sustained_FLOPs_per_GPU
```

Example for Transformer base:

```
Training time:     12 hours = 43,200 seconds
GPUs:              8× P100
P100 sustained:    ~9.5 × 10¹² FLOPs/second (single precision, sustained)

Total FLOPs = 43,200 × 8 × 9.5 × 10¹²
            = 43,200 × 7.6 × 10¹³
            ≈ 3.3 × 10¹⁸ FLOPs  ✅ (matches Table 2!)
```

This is a **hardware-agnostic** way to measure cost —
allows fair comparison between models trained on different hardware at different times.
