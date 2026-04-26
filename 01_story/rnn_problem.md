📁 01_story/

1. RNN, LSTM, GRU were state of the art
2. Hidden states ht and ht-1 chain problem
3. Sequential nature kills parallelization
4. Memory constraints and batching issues
5. Factorization tricks and conditional computation — helped      but didn't fix root problem


# What is RNN? 

### Q) What is RNN?
**RNN = Recurrent Neural Network**

An RNN is a type of neural network designed to process **sequential data** by remembering information from previous steps.

---

## Simple Analogy 

Think of it like reading a sentence word by word and remembering what you’ve read.

### Example Sentence:
> "The cat sat on the mat because it was tired"

### How You Read It:

- Read **"The"** → remember it  
- Read **"cat"** → remember *"The cat"*  
- Read **"sat"** → remember *"The cat sat"*  
- ... continue this process  

By the time you reach **"it"**, you already remember **"cat"**, so you understand that **"it" refers to the cat**.

---

## Key Idea 

RNN works in the same way:
- It reads input **one step at a time**
- It maintains a **memory** of previous inputs  
- This memory is called the **hidden state**

---

## In One Line 

> RNN processes sequences by passing information (memory) from one step to the next.



# Evolution of RNNs → LSTM → GRU → Transformers 

## 1. The Problem with Basic RNNs

RNNs (Recurrent Neural Networks) were designed to **remember past information** while reading sequences.

-> Sounds perfect, right?  
Well… not quite.

### Issue:
- Basic (Vanilla) RNNs **forget things over long distances**
- They struggle with **long-term memory**

### Example:
> "The cat, which was very tired after playing all day, sat on the mat because it was sleepy."

By the time the model reaches **"it"**, it might **forget "cat"** 

---

## 2. The Fix: LSTM & GRU 

To solve this memory problem, smarter versions of RNNs were created:

### 🔹 LSTM (Long Short-Term Memory)
- Designed to **remember important things for a long time**
- Uses special mechanisms called **gates** to control memory

### 🔹 GRU (Gated Recurrent Unit)
- A simpler and faster version of LSTM
- Still keeps the important memory features

---

## 3. What Were These Models Good At? 

Before Transformers, these were the **best (state-of-the-art)** models for:

### 🔷 Sequence Modeling
Understanding data that comes in order (like words in a sentence)

 Example:
> "I am going to school"  
Word order matters!

---

### 🔷 Transduction Problems
Converting one sequence into another

 Example:
> English → French  
> "Hello" → "Bonjour"

---

### 🔷 Language Modeling
Predicting the next word

 Example:
> "The cat sat on the ___" → **"mat"**

---

### 🔷 Machine Translation
Automatically translating languages

 Example:
> "I love AI" → "J'aime l'IA"

---

## 4. Important Concepts to Know 

### 🔹 Recurrent Language Models
- Built using RNN/LSTM/GRU
- Read words **one by one (left → right)**
- Carry memory forward

---

### 🔹 Encoder-Decoder Architecture

A **two-part system**:

1. **Encoder**   
   - Reads the input sentence  
   - Compresses it into a summary (vector)

2. **Decoder**   
   - Takes that summary  
   - Generates the output sentence

 Think of it like:
- Encoder = **understanding**
- Decoder = **speaking**

---

## 5. The Big Idea That Changed Everything 

Some researchers at Google had a bold thought:

> "What if we remove RNNs completely… and only use attention?"

That one idea led to the creation of the **Transformer model** 

> Sometimes, the biggest breakthroughs come from asking:
> **"What if we remove the old idea completely?"**

---


