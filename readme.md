# 🚀 Transformer From Scratch

A visual, story-driven breakdown of the **"Attention Is All You Need"** paper — from RNN limitations to self-attention and Transformers, with intuitive explanations, diagrams, and minimal code implementations.

---
## Folder Structure


```
transformer-from-scratch/
│
├── README.md
├── requirements.txt
├── .gitignore
├── environment.yml
│
├── 01_story/
│   ├── rnn_problem.md
│   ├── why_sequence_models_fail.md
│   ├── introduction_to_attention.md
│
├── 02_core_concepts/
│   ├── self_attention.md
│   ├── queries_keys_values.md
│   ├── multi_head_attention.md
│   ├── positional_encoding.md
│
├── 03_transformer/
│   ├── encoder.md
│   ├── decoder.md
│   ├── full_architecture.md
│
├── 04_visuals/
│   ├── diagrams/
│   │   ├── rnn_vs_attention.png
│   │   ├── self_attention_flow.png
│   │   ├── transformer_architecture.png
│   │
│   ├── source_files/
│       ├── figma_links.md   (or draw.io / excalidraw)
│
├── 05_code/
│   ├── self_attention.py
│   ├── multi_head_attention.py
│   ├── transformer_block.py
│   ├── minimal_transformer.py
│
├── 06_examples/
│   ├── sentence_walkthrough.md
│   ├── attention_visualization.ipynb
│
├── 07_summary/
│   ├── key_takeaways.md
│   ├── cheat_sheet.md
│
└── 08_resources/
    ├── original_paper.md
    ├── references.md
```
---


## 🧠 Who This Is For

* Beginners in Machine Learning
* Students trying to understand Transformers
* Developers who prefer intuition + visuals over heavy math

---

## 📚 Learning Path

Follow this in order:

### 1️⃣ The Story (Why Transformers?)

* [The Problem with RNNs](./01_story/rnn_problem.md)
* [Why Sequence Models Struggle](./01_story/why_sequence_models_fail.md)
* [Introduction to Attention](./01_story/introduction_to_attention.md)

### 2️⃣ Core Concepts

* [Self-Attention Explained](./02_core_concepts/self_attention.md)
* [Queries, Keys, Values](./02_core_concepts/queries_keys_values.md)
* [Multi-Head Attention](./02_core_concepts/multi_head_attention.md)
* [Positional Encoding](./02_core_concepts/positional_encoding.md)

### 3️⃣ Transformer Architecture

* [Encoder](./03_transformer/encoder.md)
* [Decoder](./03_transformer/decoder.md)
* [Full Architecture](./03_transformer/full_architecture.md)

### 4️⃣ Visual Learning

* [Diagrams](./04_visuals/diagrams/)
* [Source Files (Figma / Draw.io)](./04_visuals/source_files/figma_links.md)

### 5️⃣ Code Implementation

* [Self Attention Code](./05_code/self_attention.py)
* [Multi-Head Attention](./05_code/multi_head_attention.py)
* [Transformer Block](./05_code/transformer_block.py)
* [Minimal Transformer](./05_code/minimal_transformer.py)

### 6️⃣ Examples & Walkthroughs

* [Sentence Walkthrough](./06_examples/sentence_walkthrough.md)
* [Attention Visualization Notebook](./06_examples/attention_visualization.ipynb)

### 7️⃣ Summary

* [Key Takeaways](./07_summary/key_takeaways.md)
* [Cheat Sheet](./07_summary/cheat_sheet.md)

---

## 🎯 What Makes This Different

* Story-first explanation (not paper-first)
* Clean, minimal diagrams
* Step-by-step intuition → math → code
* Beginner-friendly without dumbing things down

---

## 📊 Visual Preview

*Add your best diagram here (important for first impression)*

```
![Transformer Architecture](./04_visuals/diagrams/transformer_architecture.png)
```

---

## 💻 Setup & Usage

```bash
git clone https://github.com/your-username/transformer-from-scratch.git
cd transformer-from-scratch
pip install -r requirements.txt
```

---

## 📄 Original Paper

* ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

---

## 📌 Future Improvements

* Add interactive visualizations
* Expand examples with real datasets
* Add training loop implementation

---

## 🤝 Contributing

Feel free to open issues or submit PRs if you'd like to improve explanations or visuals.

---

## ⭐ If You Found This Useful

Consider starring the repo — it helps others discover it!
