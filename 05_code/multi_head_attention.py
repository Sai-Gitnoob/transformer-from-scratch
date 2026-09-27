"""
multi_head_attention.py

Multi-Head Attention from scratch.

Story Idea:
Imagine a classroom where several teachers are reading
the same sentence.

Teacher 1 focuses on grammar.
Teacher 2 focuses on meaning.
Teacher 3 focuses on relationships.
Teacher 4 focuses on important words.

Each teacher produces their own interpretation.

The Transformer combines all of these perspectives
into a richer understanding of the sentence.

That is exactly what Multi-Head Attention does.
"""

import math
import torch
import torch.nn as nn


# ============================================================
# MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, num_heads=8):
        """
        Parameters
        ----------
        d_model : Embedding dimension

        num_heads : Number of attention heads

        Paper values:
        d_model = 512
        num_heads = 8
        """

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # Each head gets a portion of d_model
        self.head_dim = d_model // num_heads

        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        # Linear layers for Query, Key, Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final projection layer
        self.W_o = nn.Linear(d_model, d_model)

    # ========================================================
    # SPLIT INTO MULTIPLE HEADS
    # ========================================================

    def split_heads(self, x):
        """
        Input Shape:
        (batch_size, seq_len, d_model)

        Output Shape:
        (batch_size, num_heads, seq_len, head_dim)
        """

        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    # ========================================================
    # COMBINE HEADS BACK TOGETHER
    # ========================================================

    def combine_heads(self, x):
        """
        Input Shape:
        (batch_size, num_heads, seq_len, head_dim)

        Output Shape:
        (batch_size, seq_len, d_model)
        """

        batch_size, _, seq_len, _ = x.shape

        x = x.transpose(1, 2)

        return x.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

    # ========================================================
    # SCALED DOT-PRODUCT ATTENTION
    # ========================================================

    def scaled_dot_product_attention(
        self,
        Q,
        K,
        V
    ):
        """
        Attention(Q,K,V)

        Step 1:
        Calculate similarity scores

        Step 2:
        Scale scores

        Step 3:
        Softmax

        Step 4:
        Weighted sum of Values
        """

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention_weights,
            V
        )

        return output, attention_weights

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):
        """
        Input:
        x -> (batch_size, seq_len, d_model)

        Output:
        (batch_size, seq_len, d_model)
        """

        # ------------------------------------
        # Create Queries, Keys, Values
        # ------------------------------------

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # ------------------------------------
        # Split into heads
        # ------------------------------------

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # ------------------------------------
        # Attention per head
        # ------------------------------------

        attention_output, attention_weights = (
            self.scaled_dot_product_attention(
                Q,
                K,
                V
            )
        )

        # ------------------------------------
        # Combine heads
        # ------------------------------------

        attention_output = self.combine_heads(
            attention_output
        )

        # ------------------------------------
        # Final projection
        # ------------------------------------

        output = self.W_o(
            attention_output
        )

        return output, attention_weights


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    # Paper dimensions
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads
    )

    # Example sentence
    # Batch Size = 2
    # Sequence Length = 5
    # Embedding Dimension = 512

    x = torch.randn(
        2,
        5,
        d_model
    )

    output, attention_weights = mha(x)

    print("Input Shape:")
    print(x.shape)

    print("\nOutput Shape:")
    print(output.shape)

    print("\nAttention Weight Shape:")
    print(attention_weights.shape)

    """
    Expected:

    Input Shape:
    torch.Size([2, 5, 512])

    Output Shape:
    torch.Size([2, 5, 512])

    Attention Weight Shape:
    torch.Size([2, 8, 5, 5])

    Meaning:

    2 -> batch size

    8 -> attention heads

    5 -> token attending

    5 -> token being attended to
    """