# Table 1: Why Self-Attention?

This section compares four different sequence-processing architectures used in deep learning and explains why the Transformer adopts **self-attention** instead of recurrent or convolutional networks.

The comparison is based on three important metrics:

- **Computational Complexity per Layer**
- **Sequential Operations**
- **Maximum Path Length**

---

# Understanding the Metrics

## 1. Computational Complexity per Layer

This measures how much computation is required to process one layer.

The notation used is:

- **n** = sequence length (number of tokens)
- **d** = model dimension (e.g., 512)
- **k** = convolution kernel size
- **r** = attention window size (restricted attention)

Lower complexity generally means faster execution and lower computational cost.

---

## 2. Sequential Operations

This measures how much of the computation must happen one step after another.

- **O(1)** means the entire sequence can be processed simultaneously.
- **O(n)** means tokens must be processed one after another.

Since GPUs are designed for parallel computation, architectures with fewer sequential operations train much faster.

---

## 3. Maximum Path Length

This measures how many computational steps information must travel between two distant tokens.

For example, consider:

> "The animal didn't cross the street because it was too tired."

To understand that **"it"** refers to **"animal"**, information must travel between those words.

A shorter path allows the model to capture long-range dependencies more easily.

- **O(1)** means any two words communicate directly.
- Larger path lengths require information to pass through multiple intermediate computations.

---

# Layer-by-Layer Comparison

## 1. Self-Attention

### Complexity

$
O(n^2 \cdot d)
$

Every token attends to every other token.

For a sequence of **n** words, there are:

$
n \times n
$

attention scores to compute.

Each score involves vectors of dimension **d**.

### Sequential Operations

$
O(1)
$

All attention scores are computed simultaneously.

The entire sequence can be processed in parallel.

### Maximum Path Length

$
O(1)
$

Every token has a direct connection to every other token through the attention mechanism.

Example:

```
Word 1  <────────────► Word 100
```

Only one attention step separates them.

---

### Advantages

- Fully parallelizable
- Captures long-range dependencies easily
- Shortest possible information path

### Disadvantage

Quadratic computation makes very long sequences expensive.

Example:

```
10 words
10 × 10 = 100 attention scores

1000 words
1000 × 1000 = 1,000,000 attention scores
```

---

## 2. Recurrent Networks (RNNs)

### Complexity

$
O(n \cdot d^2)
$

Each token is processed sequentially using matrix multiplications involving the hidden state.

---

### Sequential Operations

$
O(n)
$

Tokens cannot be processed simultaneously.

Processing order is:

```
Word1
   ↓
Word2
   ↓
Word3
   ↓
...
Word n
```

Every token depends on the previous one.

---

### Maximum Path Length

$
O(n)
$

Information must travel through every intermediate hidden state.

Example:

```
Word1
 ↓
Word2
 ↓
Word3
 ↓
...
 ↓
Word100
```

Long-distance information gradually weakens, making it difficult to model long-range dependencies.

---

### Advantages

- Naturally handles sequential data

### Disadvantages

- Cannot be parallelized
- Long information paths
- Suffers from vanishing gradients over long sequences

---

## 3. Convolutional Networks (CNNs)

### Complexity

$
O(k \cdot n \cdot d^2)
$

where

- **k** = kernel size

A convolution processes only a local neighborhood around each token.

---

### Sequential Operations

$
O(1)
$

Different positions can be processed simultaneously.

---

### Maximum Path Length

$
O(\log_k(n))
$

A single convolution only sees nearby tokens.

To connect distant words, multiple convolution layers must be stacked.

Example:

```
Layer 1
Word1 ── Word2 ── Word3

Layer 2
Word1 ───────── Word3

Layer 3
Word1 ───────────────── Word5
```

Information gradually spreads across layers.

---

### Advantages

- Parallelizable
- Efficient for local patterns

### Disadvantages

- Requires multiple layers to capture long-range relationships

---

## 4. Restricted Self-Attention

Instead of attending to every token, each token attends only to nearby neighbors.

Suppose the attention window size is **r**.

Each token only communicates with **r** surrounding tokens.

---

### Complexity

$
O(r \cdot n \cdot d)
$

This is much cheaper than full self-attention because

$
r \ll n
$

---

### Sequential Operations

$
O(1)
$

The computation remains fully parallel.

---

### Maximum Path Length

$
O\left(\frac{n}{r}\right)
$

Information travels gradually through overlapping attention windows.

Example:

```
Window 1:
1 2 3

Window 2:
3 4 5

Window 3:
5 6 7
```

A token cannot directly reach distant tokens.

Instead, information propagates through neighboring windows.

---

### Advantages

- Much lower computational cost
- Parallelizable

### Disadvantages

- Longer information paths
- Cannot model global relationships as effectively as full self-attention

---

# Comparison Table

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
| :--- | :--- | :--- | :--- |
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| **Recurrent** | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| **Convolutional** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| **Restricted Self-Attention** | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |
---

# Why Did the Transformer Choose Self-Attention?

Although self-attention has quadratic computational complexity, it provides two significant advantages:

1. **Complete parallelism**

   Every token is processed simultaneously, making training highly efficient on modern hardware.

2. **Shortest information path**

   Any token can directly attend to any other token in a single step.

These properties make it much easier to learn long-range dependencies than recurrent or convolutional architectures.

For typical NLP tasks, where sequence lengths are moderate, these benefits outweigh the increased computational cost.

---

# Summary

Self-attention trades higher computational complexity for significantly better parallelism and the shortest possible communication path between tokens. Compared with recurrent and convolutional architectures, it enables more efficient training and better modeling of long-range dependencies, making it the foundation of the Transformer architecture.