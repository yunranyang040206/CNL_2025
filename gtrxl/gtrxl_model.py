import jax
import jax.numpy as jnp
from flax import linen as nn

class GRUGating(nn.Module):
    embed_dim: int
    bias_init: float = 2.0  # Paper recommends b_g > 0 to initialize near identity

    @nn.compact
    def __call__(self, x, y):
        # 1. Reset Gate (r)
        r = nn.Dense(self.embed_dim)(y) + nn.Dense(self.embed_dim)(x)
        r = nn.sigmoid(r)

        # 2. Update Gate (z)
        z_dense = nn.Dense(self.embed_dim,
                           bias_init=nn.initializers.constant(-self.bias_init))
        z = z_dense(y) + nn.Dense(self.embed_dim)(x)
        z = nn.sigmoid(z)

        # 3. Candidate Activation (h_hat)
        h_hat = nn.Dense(self.embed_dim)(y) + nn.Dense(self.embed_dim)(r * x)
        h_hat = nn.tanh(h_hat)

        # 4. Gated Output
        return (1 - z) * x + z * h_hat


class GTrXLBlock(nn.Module):
    embed_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, training: bool = False):
        # --- Submodule 1: Multi-Head Attention ---
        residual = x
        x_norm = nn.LayerNorm()(x)

        y = nn.SelfAttention(num_heads=self.num_heads)(x_norm, mask=mask)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
        y = nn.relu(y)

        x = GRUGating(embed_dim=self.embed_dim)(residual, y)

        # --- Submodule 2: MLP ---
        residual = x
        x_norm = nn.LayerNorm()(x)

        y = nn.Dense(self.embed_dim * 4)(x_norm)
        y = nn.relu(y)
        y = nn.Dense(self.embed_dim)(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
        y = nn.relu(y)

        x = GRUGating(embed_dim=self.embed_dim)(residual, y)

        return x


class GTrXL(nn.Module):
    """
    Full GTrXL encoder + auxiliary heads.

    For IRL later we will mostly care about:
      - context: last-token embedding (seq history summary)
      - u: small "history bottleneck" derived from context
    """
    n_states: int
    n_actions: int
    embed_dim: int = 32
    num_heads: int = 4
    num_layers: int = 4
    seq_len: int = 10
    dropout: float = 0.1
    future_horizon: int = 5

    def setup(self):
        self.embed = nn.Embed(
            num_embeddings=self.n_states,
            features=self.embed_dim
        )
        self.pos_emb = self.param(
            'pos_embed',
            nn.initializers.normal(0.02),
            (1, self.seq_len, self.embed_dim),
        )

        self.blocks = [
            GTrXLBlock(self.embed_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ]

        # --- Auxiliary Heads for pretraining ---
        # 1. Next State Prediction
        self.head_next_state = nn.Dense(self.n_states)

        # 2. Future Occupancy
        self.head_future_occ = nn.Dense(self.n_states)

    def encode(self, x, training: bool = False):
        """
        Pure encoder forward pass.

        x: (batch, seq_len) int32 states

        Returns:
          h:         (batch, seq_len, embed_dim)
          context:   (batch, embed_dim)  - last token
        """
        h = self.embed(x)                          # (B, T, D)
        h = h + self.pos_emb                      # positional encoding
        mask = nn.make_causal_mask(x)

        for block in self.blocks:
            h = block(h, mask=mask, training=training)

        # last token = summary of window history
        context = h[:, -1, :]

        return h, context

    def __call__(self, x, training: bool = False):
        """
        During pretraining, we use:
          - logits_next_state
          - logits_future
        """
        _, context = self.encode(x, training=training)

        logits_next_state = self.head_next_state(context)
        logits_future = self.head_future_occ(context)

        return context, logits_next_state, logits_future

