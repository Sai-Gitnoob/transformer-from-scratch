## ReLU (Activation Function)

ReLU stands for Rectified Linear Unit:

$$\text{ReLU}(x) = \max(0, x)$$

### Simple Behavior

- Positive values → kept  
- Negative values → set to 0  

Example:

ReLU(5.3) = 5.3

ReLU(-2.1) = 0.0

### Why It Matters

Without ReLU, multiple linear layers collapse into one.  
ReLU introduces non-linearity, allowing the model to learn complex patterns.

Used inside:
- Feed-Forward Networks (FFN)
- Many deep learning architectures


---
**1\. WMT 2014 Benchmark**The **WMT 2014** refers to the Shared Tasks of the Ninth Workshop on Statistical Machine Translation held in Baltimore, USA, which serves as a foundational benchmark dataset for training and evaluating neural machine translation (NMT) architectures. \[[1](https://aclanthology.org/W14-3302.pdf)\]

*   **Official Overview Paper**: [Findings of the 2014 Workshop on Statistical Machine Translation](https://aclanthology.org/W14-3302.pdf)
    
*   **Full Conference Proceedings**: [ACL Anthology WMT 2014 Event Page](https://aclanthology.org/events/wmt-2014/)
    
*   **Workshop Website**: [ACL 2014 Ninth Workshop on Statistical Machine Translation](https://www.statmt.org/wmt14/) \[[1](https://aclanthology.org/W14-3302.pdf), [2](https://aclanthology.org/events/wmt-2014/), [3](https://www.statmt.org/wmt14/)\]


---

**2\. BPE Paper (Byte Pair Encoding)**The landmark **BPE** paper introduced Byte Pair Encoding as a subword segmentation method to let a fixed vocabulary represent rare and unseen words, effectively resolving the out-of-vocabulary (OOV) problem in translation models. \[[1](https://colendiai.com/intelligence-papers/neural-machine-translation-of-rare-words-with-subword-units)\]

*   **Title**: _Neural Machine Translation of Rare Words with Subword Units_ (Sennrich et al., 2015/2016)
    
*   **Primary Pre-print**: [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
    
*   **Alternative Repositories**: [Hugging Face Papers Entry](https://huggingface.co/papers/1508.07909) | [ResearchGate Entry](https://www.researchgate.net/publication/281427452_Neural_Machine_Translation_of_Rare_Words_with_Subword_Units) \[[1](https://arxiv.org/abs/1508.07909), [2](https://medium.com/analytics-vidhya/subword-techniques-for-neural-machine-translation-f55e4506a728), [3](https://huggingface.co/papers/1508.07909), [4](https://www.researchgate.net/publication/281427452_Neural_Machine_Translation_of_Rare_Words_with_Subword_Units), [5](https://medium.com/@zljdanceholic/llm-tokenization-bpe-bbpe-wordpiece-and-ulm-7e5568d76231)\]
    

**3\. WordPiece Paper**The **WordPiece** tokenization algorithm optimizes vocabulary construction by maximizing a language model scoring criteria (mutual information) rather than strict raw frequency counts. It is widely recognized for its fundamental usage in the BERT architecture and early Google NMT systems. \[[1](https://arxiv.org/abs/1609.08144), [2](https://cstopics.com/encyclopedia/ai-ml/natural-language-processing/text-representation/subword-tokenization-bpe-wordpiece-sentencepiece-unigram)\]

*   **Original Speech/Voice Search Implementation Paper**: _Japanese and Korean Voice Search_ (Schuster & Nakajima, 2012) — This is the initial publication where the WordPiece concept was pioneered.
    
    *   **Direct Download**: [Google Research Repository PDF](https://research.google.com/pubs/archive/37842.pdf)
        
    *   **IEEE Platform**: [IEEE Xplore Entry](https://ieeexplore.ieee.org/document/6289079) \[[1](https://research.google.com/pubs/archive/37842.pdf), [2](https://ieeexplore.ieee.org/document/6289079), [3](https://rabinadhikari.com.np/blog/2021/training-of-word-embeddings/)\]
        
*   **Popular NMT Application Paper**: _Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation_ (Wu et al., 2016) — This brought WordPiece into widespread NMT usage.
    
    *   **Direct Preprint**: [arXiv:1609.08144](https://arxiv.org/abs/1609.08144) \[[1](https://www.researchgate.net/publication/308646556_Google's_Neural_Machine_Translation_System_Bridging_the_Gap_between_Human_and_Machine_Translation), [2](https://arxiv.org/abs/1609.08144)\]
  

---

**1\. GNMT (Google's Neural Machine Translation)**This foundational paper shifted Google Translate from phrase-based translation to a deep LSTM network using attention and residual connections. It also popularized the WordPiece tokenization method in deep learning. \[[1](https://arxiv.org/abs/1609.08144), [2](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5320672)\]

*   **Title**: _Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation_ (Wu et al., 2016)
    
*   **Primary Preprint**: [arXiv:1609.08144](https://arxiv.org/abs/1609.08144)
    
*   **Alternative Formats**: [ResearchGate Entry](https://www.researchgate.net/publication/308646556_Google's_Neural_Machine_Translation_System_Bridging_the_Gap_between_Human_and_Machine_Translation)
    
*   **Official Summary**: Google Research Blog Post \[[1](https://arxiv.org/abs/1609.08144), [2](https://sh-tsang.medium.com/review-googles-neural-machine-translation-system-bridging-the-gap-between-human-and-machine-518595d87226), [3](https://www.researchgate.net/publication/308646556_Google's_Neural_Machine_Translation_System_Bridging_the_Gap_between_Human_and_Machine_Translation), [4](https://research.google/blog/a-neural-network-for-machine-translation-at-production-scale/), [5](https://dl.acm.org/doi/10.1145/3626772.3657867)\]
    

**2\. ConvS2S (Convolutional Sequence to Sequence)**Introduced by Facebook AI Research (FAIR), this architecture challenged traditional RNN dominance by relying entirely on convolutional neural networks to fully parallelize training calculations. \[[1](https://arxiv.org/abs/1705.03122), [2](https://sh-tsang.medium.com/review-convolutional-sequence-to-sequence-learning-convs2s-510a9eddce05)\]

*   **Title**: _Convolutional Sequence to Sequence Learning_ (Gehring et al., 2017)
    
*   **Primary Preprint**: [arXiv:1705.03122](https://arxiv.org/abs/1705.03122)
    
*   **Conference Proceedings**: ICML 2017 Publication Page \[[1](https://arxiv.org/abs/1705.03122), [2](https://aclanthology.org/P18-1008.pdf), [3](https://sh-tsang.medium.com/review-convolutional-sequence-to-sequence-learning-convs2s-510a9eddce05)\]
    

**3\. MoE (Mixture of Experts)**This landmark paper from Google Brain introduced conditional computation via a sparsely-gated Mixture-of-Experts layer. It scaled deep networks to over 137 billion parameters, paving the way for modern sparse LLMs. \[[1](https://arxiv.org/abs/1701.06538), [2](https://sh-tsang.medium.com/review-outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer-moe-6c67cebf9504)\]

*   **Title**: _Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer_ (Shazeer et al., 2017)
    
*   **Primary Preprint**: [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
    
*   **Official PDF**: [University of Toronto (Hinton Lab) Copy](https://www.cs.toronto.edu/~hinton/absps/Outrageously.pdf)
    
*   **Alternative Formats**: [Hugging Face Papers Entry](https://huggingface.co/papers/1701.06538) \[[1](https://sh-tsang.medium.com/review-outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer-moe-6c67cebf9504), [2](https://www.cs.toronto.edu/~hinton/absps/Outrageously.pdf), [3](https://arxiv.org/abs/1701.06538), [4](https://huggingface.co/papers/1701.06538)\]

---
# Paper Summary:

The Paper's Place in History 📜
Published:   June 12, 2017 (arXiv)
Conference:  NeurIPS 2017
Citations:   100,000+ (one of the most cited ML papers ever)
Impact:      Redefined entire field of AI

A timeline of how it changed everything:

2017 ── "Attention is All You Need" published
         └── Transformer architecture introduced

2018 ── BERT, GPT-1
         └── Pretraining paradigm begins

2019 ── GPT-2, XLNet, RoBERTa
         └── Scale starts to matter

2020 ── GPT-3, T5, Vision Transformer
         └── Few-shot learning, cross-domain dominance

2021 ── Codex, DALL-E, AlphaFold 2
         └── Code generation, image generation, protein folding

2022 ── ChatGPT, Stable Diffusion, Whisper
         └── Public AI era begins

2023 ── GPT-4, Claude 2, Gemini, LLaMA
         └── Multimodal, reasoning, open source

2024 ── Claude 3, GPT-4o, Gemini Ultra
         └── Near-human performance on many benchmarks

2026 ── You are here, learning the paper that started it all
Summary of the Entire Paper — One Page
PROBLEM:
  RNNs are sequential → can't parallelize → slow + forgets long dependencies
  CNNs need many layers to connect distant words → O(log n) path length
  
SOLUTION:
  Replace everything with self-attention
  O(1) path length — every word directly attends every other word
  O(1) sequential ops — everything happens in parallel
  
ARCHITECTURE:
  Encoder: 6 layers of (Multi-Head Self-Attention + FFN + Residual + Norm)
  Decoder: 6 layers of (Masked Self-Attention + Cross-Attention + FFN + Residual + Norm)
  dmodel=512, h=8 heads, dk=dv=64, dff=2048
  
TRAINING:
  Adam optimizer + warmup + decay learning rate
  Dropout=0.1 + Label smoothing=0.1
  4.5M EN-DE sentences, 100K steps, 12 hours on 8× P100
  
RESULTS:
  EN-DE: 28.4 BLEU (beats all ensembles, new state of the art)
  EN-FR: 41.0 BLEU (matches best ensemble at 1/4 the cost)
  
IMPACT:
  Every major AI system since 2018 is built on this architecture
One Line Summary

The Transformer proved attention alone is sufficient — faster to train, better quality, more parallelizable than anything before it — and its architecture became the foundation of every major AI breakthrough from BERT to GPT-4 to Claude