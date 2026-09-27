"""
minimal_transformer.py

A minimal implementation of the Transformer architecture
from the paper:

"Attention Is All You Need" (Vaswani et al., 2017)

This file focuses on understanding the architecture
rather than production-level optimization.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# POSITIONAL ENCODING
# ============================================================
# Transformers process all tokens simultaneously.
# Unlike RNNs, they do not naturally understand word order.
#
# Positional Encoding injects information about a token's
# position into its embedding.
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]


# ============================================================
# TRANSFORMER MODEL
# ============================================================
#
# Paper values:
#
# d_model = 512
# num_heads = 8
# num_encoder_layers = 6
# num_decoder_layers = 6
# feedforward_dim = 2048
#
# ============================================================

class MinimalTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Convert token IDs into dense vectors
        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Add position information
        self.positional_encoding = PositionalEncoding(
            d_model
        )

        # Built-in PyTorch Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Convert Transformer output
        # back into vocabulary probabilities
        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(
        self,
        src,
        tgt
    ):

        # Convert token IDs → embeddings
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)

        # Add positional information
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Run through transformer
        output = self.transformer(
            src=src,
            tgt=tgt
        )

        # Convert hidden vectors → vocab scores
        output = self.output_layer(output)

        return output


# ============================================================
# NOAM LEARNING RATE SCHEDULER
# ============================================================
#
# Original paper schedule:
#
# lr = d_model^(-0.5) *
#      min(step^(-0.5),
#          step * warmup_steps^(-1.5))
#
# This increases LR early,
# then slowly decreases it.
#
# ============================================================

class NoamScheduler:

    def __init__(
        self,
        optimizer,
        d_model,
        warmup_steps=4000
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):

        self.step_num += 1

        lr = (
            self.d_model ** (-0.5)
        ) * min(
            self.step_num ** (-0.5),
            self.step_num * (self.warmup_steps ** (-1.5))
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    vocab_size = 10000

    model = MinimalTransformer(
        vocab_size=vocab_size
    )

    # --------------------------------------------------------
    # Adam Optimizer
    #
    # Exact paper values:
    # beta1 = 0.9
    # beta2 = 0.98
    # eps   = 1e-9
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=0,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=512,
        warmup_steps=4000
    )

    # Example batch
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    src = torch.randint(
        0,
        vocab_size,
        (batch_size, src_seq_len)
    )

    tgt = torch.randint(
        0,
        vocab_size,
        (batch_size, tgt_seq_len)
    )

    output = model(
        src,
        tgt
    )

    print("Output shape:")
    print(output.shape)

    # Example scheduler update
    current_lr = scheduler.step()

    print("\nLearning Rate:")
    print(current_lr)