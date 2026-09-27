"""
self_attention.py

Implements Scaled Dot-Product Attention — the formula from
02_core_concepts/2.queries_keys_values.md:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

This is the single building block that multi_head_attention.py and
transformer_block.py will reuse. Read alongside the markdown chapter if
any of the shapes below feel unfamiliar.

Notation (matches the book):
    n    = sequence length (number of tokens)
    d_k  = dimension of each query/key vector
    d_v  = dimension of each value vector
"""

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor,  shape (..., n, d_k)
        K: Key tensor,    shape (..., n, d_k)
        V: Value tensor,  shape (..., n, d_v)
        mask: Optional boolean tensor, shape broadcastable to (..., n, n).
              Positions where mask == False are blocked from attending
              (set to -inf before softmax). Used later by the decoder's
              masked self-attention (03_transformer/2.decoder.md) — the
              encoder simply calls this with mask=None.

    Returns:
        output: shape (..., n, d_v) — the attention-weighted blend of V
        attn_weights: shape (..., n, n) — the softmax weights themselves,
                      handy for printing/inspecting what the model "looked at"
    """
    d_k = Q.size(-1)

    # Step 1 & 2 — dot product every query against every key, then scale.
    # QK^T: (..., n, d_k) @ (..., d_k, n) -> (..., n, n)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 3 (optional, decoder only) — hide future/forbidden positions
    # by forcing their score to -inf before softmax, so they become
    # exactly 0 after softmax (see 03_transformer/2.decoder.md, Section 3).
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))

    # Step 4 — softmax turns scores into weights that sum to 1 along the
    # last dimension (each query's weights over all keys).
    attn_weights = F.softmax(scores, dim=-1)

    # Step 5 — weighted sum of the values.
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


if __name__ == "__main__":
    # --- A small worked demo, in the spirit of the book's examples ---
    # 4 tokens standing in for: "cat", "tired", "mat", "sat" — the same
    # supporting words used in the "it" example throughout the book.
    torch.manual_seed(0)

    n, d_k, d_v = 4, 8, 8
    tokens = ["cat", "tired", "mat", "sat"]

    # In a real model these come from the embedding + positional encoding
    # layers (02_core_concepts/4.positional_encoding.md). Here we just use
    # random vectors — the mechanism is identical either way, only the
    # actual numbers will differ from the book's illustrative example.
    Q = torch.randn(1, n, d_k)  # imagine this row as the query for "it"
    K = torch.randn(1, n, d_k)
    V = torch.randn(1, n, d_v)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("Attention weights ('it' attending to each token):")
    for i, row in enumerate(weights[0]):
        print(f"  query {i}: " + ", ".join(
            f"{tok}={w:.3f}" for tok, w in zip(tokens, row.tolist())
        ))
        print(f"    (sums to {row.sum().item():.3f} — softmax always sums to 1)")

    print("\nOutput shape:", output.shape, "→ one blended vector per query, still d_v-dimensional")