"""
transformer_block.py

Transformer Block from scratch.

Story Idea:
Think of a Transformer Block as a team meeting.

Step 1:
Everyone looks at everyone else in the sentence
(Multi-Head Attention).

Step 2:
The original information is preserved
(Residual Connection).

Step 3:
The information is cleaned and stabilized
(Layer Normalization).

Step 4:
A small neural network thinks independently
(Feed Forward Network).

Step 5:
Another residual connection preserves information.

This complete process forms ONE Transformer Block.

Stack several blocks together and you get
the Transformer Encoder.
"""

import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention


# ============================================================
# FEED FORWARD NETWORK
# ============================================================
#
# After attention gathers information,
# a small neural network processes each token
# independently.
#
# Paper:
# 512 -> 2048 -> 512
#
# ============================================================

class FeedForwardNetwork(nn.Module):

    def __init__(
        self,
        d_model=512,
        hidden_dim=2048,
        dropout=0.1
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                d_model,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                d_model
            )
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model=512,
        num_heads=8,
        hidden_dim=2048,
        dropout=0.1
    ):
        super().__init__()

        # ----------------------------------
        # Multi-Head Attention
        # ----------------------------------

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        # ----------------------------------
        # Feed Forward Network
        # ----------------------------------

        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # ----------------------------------
        # Layer Normalization
        # ----------------------------------

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        # ----------------------------------
        # Dropout
        # ----------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        # ====================================================
        # STEP 1
        # MULTI-HEAD ATTENTION
        # ====================================================

        attention_output, attention_weights = (
            self.attention(x)
        )

        # ====================================================
        # STEP 2
        # RESIDUAL CONNECTION
        #
        # Original Input + Attention Output
        # ====================================================

        x = x + self.dropout(
            attention_output
        )

        # ====================================================
        # STEP 3
        # LAYER NORMALIZATION
        # ====================================================

        x = self.norm1(x)

        # ====================================================
        # STEP 4
        # FEED FORWARD NETWORK
        # ====================================================

        ffn_output = self.ffn(x)

        # ====================================================
        # STEP 5
        # SECOND RESIDUAL CONNECTION
        # ====================================================

        x = x + self.dropout(
            ffn_output
        )

        # ====================================================
        # STEP 6
        # SECOND NORMALIZATION
        # ====================================================

        x = self.norm2(x)

        return x, attention_weights


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    # --------------------------------------------------------
    # Paper Values
    # --------------------------------------------------------

    d_model = 512
    num_heads = 8
    hidden_dim = 2048

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        hidden_dim=hidden_dim
    )

    # --------------------------------------------------------
    # Example Input
    #
    # batch_size = 2
    # sequence_length = 5
    # embedding_dimension = 512
    # --------------------------------------------------------

    x = torch.randn(
        2,
        5,
        d_model
    )

    output, attention_weights = block(x)

    print("Input Shape:")
    print(x.shape)

    print("\nOutput Shape:")
    print(output.shape)

    print("\nAttention Shape:")
    print(attention_weights.shape)

    """
    Expected Output:

    Input Shape:
    torch.Size([2, 5, 512])

    Output Shape:
    torch.Size([2, 5, 512])

    Attention Shape:
    torch.Size([2, 8, 5, 5])

    Interpretation:

    Batch Size = 2

    8 Attention Heads

    5 Tokens attending

    5 Tokens being attended to
    """