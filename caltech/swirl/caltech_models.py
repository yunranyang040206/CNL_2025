"""
Shared neural-network reward model for the Caltech SWIRL experiment.

Exported:
  MLP                  — Flax module
  create_train_state   — convenience constructor
"""

import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state


class MLP(nn.Module):
    subnet_size: int
    hidden_size: int
    output_size: int   # = C = 4
    n_hidden:    int   # = K
    expand:      bool

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.n_hidden * self.output_size)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.leaky_relu(x)
        x = self.dense2(x)
        return x.reshape((self.n_hidden, self.output_size))


def create_train_state(rng, subnet_size, learning_rate, n_hidden,
                        input_size, hidden_size, output_size, expand=False):
    model  = MLP(subnet_size=subnet_size, hidden_size=hidden_size,
                  output_size=output_size, n_hidden=n_hidden, expand=expand)
    params = model.init(rng, jnp.ones((1, input_size)))['params']
    tx     = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
