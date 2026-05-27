from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    sublayer 출력에 dense projection → dropout 적용 후 residual 연결.
    Layer Norm은 여기서 적용하지 않음(forward에서 pre-norm으로 처리).
    """
    return input + dropout(dense_layer(output))

  def forward(self, hidden_states, attention_mask):
    """
    GPT-2 Pre-LN 구조:
      1. LayerNorm → CausalSelfAttention → residual add
      2. LayerNorm → FFN (interm_dense → GELU → out_dense) → residual add
    """
    # --- Self-Attention block ---
    normed = self.attention_layer_norm(hidden_states)
    attn_out = self.self_attention(normed, attention_mask)
    hidden_states = self.add(hidden_states, attn_out, self.attention_dense, self.attention_dropout)

    # --- Feed-Forward block ---
    normed = self.out_layer_norm(hidden_states)
    ff_out = self.interm_af(self.interm_dense(normed))
    hidden_states = self.add(hidden_states, ff_out, self.out_dense, self.out_dropout)

    return hidden_states
