# Understanding RNN Limitations 🧠

## 1. How RNN Actually Works

In an RNN, between the input and output, there are **hidden states**.

These hidden states follow a simple rule:

- At step **t**, the model computes a hidden state **hₜ**
- But to compute **hₜ**, it needs **hₜ₋₁** (the previous state)

👉 So each step depends on the previous one

---

## 2. Why This Is a Problem ⚠️

### 🔷 Sequential Dependency

Think of it like a **chain** ⛓️

- Step 3 waits for Step 2  
- Step 2 waits for Step 1  
- ... and so on  

You **cannot jump ahead**

---

### 🔷 No Parallelization

Modern hardware like **GPUs** are designed to do many things **at the same time** ⚡

But RNNs force:

> ❌ One step at a time  
> ❌ No parallel processing  

👉 This makes training **slow and inefficient**

---

### 🔷 Long Sequences = Bigger Problems

- More steps → longer wait times  
- More memory needed  
- Harder to learn long-range relationships  

---

## 3. Improvements That Helped (But Didn’t Fix Everything) 🛠️

### 🔷 Factorization Tricks

💡 Idea: Break big computations into smaller ones

- Large weight matrices are **split into smaller matrices**
- Same result, but:
  - Less memory  
  - Faster computation  

👉 Like simplifying a big math problem into smaller steps

---

### 🔷 Conditional Computation

💡 Idea: Use only what you need

Instead of activating the whole network:

- Only **relevant parts** are used for each input

👉 Analogy:
> A hospital 🏥  
> You don’t call every doctor — only the specialist you need

---

## 4. Key Intuition (Kid-Friendly) 🧒

### 🔷 Hidden States (hₜ and hₜ₋₁)

Think of a **relay race baton** 🏃

- Each step passes information to the next  
- You can’t run your turn until you get the baton  

👉 That baton = **memory (hidden state)**

---

## 5. The Real Bottleneck 🚧

Even with all improvements:

- RNNs became **faster**
- RNNs became **lighter**

But...

> ❌ The sequential chain is still there  
> ❌ You still have to process **one word at a time**

---

## 6. Final Punchline 🎯

> RNNs improved in efficiency, but not in structure.

The **core limitation remained unchanged**:

👉 **Sequential processing = bottleneck**

And this is exactly the problem that later led to the rise of **Transformers** 🚀