"""LLaMA-style transformer model implementation (from-scratch, JAX / flax.nnx).

NOTE: this from-scratch model (`LlamaModel` / `create_sharded_model`) exists for a
minimal, self-contained version of the JAX dataset-metagrads system and is used by most
of the tests (e.g. the "small" preset; the exception is test_compare_easydel_and_hf_model,
which loads an EasyDeL model). The RL experiments in this repo do NOT use it: they train
pretrained models loaded through EasyDeL (`create_pretrained_easydel_sharded_model`,
selected by setting `easydel_pretrained_override` in the config), so for those runs the
from-scratch model below is unused.
"""

from typing import Optional
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import math
from jax.sharding import PartitionSpec as P
import easydel as ed
sharding_rules = (('batch', 'data'), ('model_dim', None), ('embed', None), ('head', None), ('mlp', None), ('head_dim', None), ('embed_model_dim', None))


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-2, dtype: jnp.dtype = jnp.bfloat16, *, rngs: nnx.Rngs): # TODO: check if this works - going from 1e-6 to 1e-2 in the hopes of stabilizing metagrads..
        self.weight = nnx.Param(jnp.ones(dim, dtype=dtype), sharding=(None,), sharding_rules=sharding_rules)
        self.eps = eps
        self.dtype = dtype

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return (
            x
            * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
            * self.weight
        )


class RotaryPositionalEncoding(nnx.Module):
    """Rotary Positional Encoding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0, dtype: jnp.dtype = jnp.bfloat16):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.base = base

    def _get_sin_cos(self):
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2).astype(self.dtype) / self.dim))
        t = jnp.arange(self.max_seq_len).astype(self.dtype)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        return jnp.sin(freqs), jnp.cos(freqs)
        

    def apply_rotary_pos_emb(
        self, x: jnp.ndarray, position_ids: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        if position_ids is None:
            seq_len = x.shape[-2]
            position_ids = jnp.arange(seq_len)

        sin_v, cos_v = self._get_sin_cos()

        cos = cos_v[position_ids]
        sin = sin_v[position_ids]

        # Split into pairs and apply rotation
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        x_rotated = jnp.stack(
            [
                x_pairs[..., 0] * cos - x_pairs[..., 1] * sin,
                x_pairs[..., 0] * sin + x_pairs[..., 1] * cos,
            ],
            axis=-1,
        )

        return x_rotated.reshape(x.shape)


class SwiGLUMLP(nnx.Module):
    """SwiGLU MLP layer."""


    def __init__(self, dim: int, hidden_dim: int, dtype: jnp.dtype = jnp.bfloat16, *, rngs: nnx.Rngs):
        init_fn = nnx.initializers.lecun_normal()
        self.gate_proj = nnx.Linear(dim, hidden_dim, use_bias=False, dtype=dtype, rngs=rngs, kernel_init=nnx.with_metadata(init_fn, sharding=('model_dim', 'mlp'), sharding_rules=sharding_rules))
        self.up_proj = nnx.Linear(dim, hidden_dim, use_bias=False, dtype=dtype, rngs=rngs, kernel_init=nnx.with_metadata(init_fn, sharding=('model_dim', 'mlp'), sharding_rules=sharding_rules))
        self.down_proj = nnx.Linear(hidden_dim, dim, use_bias=False, dtype=dtype, rngs=rngs, kernel_init=nnx.with_metadata(init_fn, sharding=('mlp', 'model_dim'), sharding_rules=sharding_rules))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(jax.nn.silu(gate) * up)


class MultiHeadAttention(nnx.Module):
    """Multi-head attention with RoPE."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048,
        dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads

        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        init_fn = nnx.initializers.lecun_normal()

        self.q_proj = nnx.Linear(
            dim, n_heads * self.head_dim, use_bias=False, dtype=dtype, rngs=rngs, kernel_init=nnx.with_metadata(init_fn, sharding=('model_dim', 'head'), sharding_rules=sharding_rules)
        )
        self.k_proj = nnx.Linear(
            dim, self.n_kv_heads * self.head_dim, use_bias=False, dtype=dtype, rngs=rngs, kernel_init=nnx.with_metadata(init_fn, sharding=('model_dim', 'head'), sharding_rules=sharding_rules)
        )
        self.v_proj = nnx.Linear(
            dim, self.n_kv_heads * self.head_dim, use_bias=False, dtype=dtype, rngs=rngs, kernel_init=nnx.with_metadata(init_fn, sharding=('model_dim', 'head'), sharding_rules=sharding_rules)
        )
        self.o_proj = nnx.Linear(
            n_heads * self.head_dim, dim, use_bias=False, dtype=dtype, rngs=rngs, kernel_init=nnx.with_metadata(init_fn, sharding=('head', 'model_dim'), sharding_rules=sharding_rules)
        )

        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len, dtype=dtype)

    def __call__(
        self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q = self.rope.apply_rotary_pos_emb(q)
        k = self.rope.apply_rotary_pos_emb(k)

        # Transpose for attention computation
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, n_heads, seq_len, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))  # (batch, n_kv_heads, seq_len, head_dim)
        v = jnp.transpose(v, (0, 2, 1, 3))  # (batch, n_kv_heads, seq_len, head_dim)

        # Grouped query attention if n_kv_heads < n_heads
        if self.n_kv_heads < self.n_heads:
            k = jnp.repeat(k, self.n_heads // self.n_kv_heads, axis=1)
            v = jnp.repeat(v, self.n_heads // self.n_kv_heads, axis=1)

        # Attention computation
        scores = jnp.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = jnp.where(mask, -jnp.inf, scores)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhij,bhjd->bhid", attn_weights, v)

        # Reshape and project
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
        return self.o_proj(out)


class TransformerBlock(nnx.Module):
    """Single transformer block."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 2048,
        dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        hidden_dim = int(dim * mlp_ratio)

        self.attention_norm = RMSNorm(dim, dtype=dtype, rngs=rngs)
        self.attention = MultiHeadAttention(
            dim, n_heads, n_kv_heads, max_seq_len, dtype=dtype, rngs=rngs
        )
        self.ffn_norm = RMSNorm(dim, dtype=dtype, rngs=rngs)
        self.mlp = SwiGLUMLP(dim, hidden_dim, dtype=dtype, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        # Pre-norm attention
        h = x + self.attention(self.attention_norm(x), mask)
        # Pre-norm MLP
        h = h + self.mlp(self.ffn_norm(h))
        return h

class LlamaModel(nnx.Module):
    """LLaMA-style transformer model."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048,
        mlp_ratio: float = 4.0,
        dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.n_layers = n_layers

        init_fn = nnx.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
        self.embedding = nnx.Embed(vocab_size, dim, dtype=dtype, rngs=rngs, embedding_init=nnx.with_metadata(init_fn, sharding=('embed', 'embed_model_dim'), sharding_rules=sharding_rules))

        @nnx.split_rngs(splits=n_layers)
        @nnx.vmap(transform_metadata={nnx.PARTITION_NAME: None})
        def make_layer(rngs: nnx.Rngs):
            return TransformerBlock(
                dim, n_heads, n_kv_heads, mlp_ratio, max_seq_len=max_seq_len, dtype=dtype, rngs=rngs
            )

        self.layers = make_layer(rngs)

        self.norm = RMSNorm(dim, dtype=dtype, rngs=rngs)
        self.lm_head = nnx.Linear(dim, vocab_size, use_bias=False, dtype=dtype, rngs=rngs, kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding=('embed_model_dim', 'embed'), sharding_rules=sharding_rules))

    def __call__(self, input_ids: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        batch_size, seq_len = input_ids.shape
        input_ids = jax.lax.with_sharding_constraint(input_ids, P('data', None)) # cant do aliasing, so we have to specify data here.

        # Create causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len))) == 0
        mask = mask[None, None, :, :]  # Add batch and head dimensions

        # Embedding - convert to appropriate dtype
        x = self.embedding(input_ids).astype(self.dtype)

        graphdef, params_list = nnx.split(self.layers)
        def layer_apply(layer_params, carry, mask):
            return nnx.merge(graphdef, layer_params)(carry, mask)
        
        layer_apply = jax.remat(layer_apply)
        
        x = nnx.scan(lambda c, layer_params: layer_apply(layer_params, c, mask), out_axes=nnx.Carry)(x, params_list)

        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

@nnx.jit(static_argnames=("max_new_tokens", "max_seq_len"))
def generate_tokens(model,
            input_ids: jnp.ndarray,
            *,
            max_new_tokens: int = 50,
            max_seq_len: int = 2048,
            temperature: float = 1.0,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> jnp.ndarray:
    """Autoregressive decode that compiles once and runs fast."""

    batch_size, init_len = input_ids.shape
    if init_len > max_seq_len:
        raise ValueError(f"context length {init_len} exceeds model context")

    max_len = max_seq_len
    pad = max_len - init_len
    tokens = jnp.pad(input_ids,
                    ((0, 0), (0, pad)),
                    constant_values=0)         
    length = jnp.asarray(init_len, jnp.int32)

    def step(carry, _):
        tokens, length, key = carry

        logits = model(tokens)
        next_logits = jax.lax.dynamic_index_in_dim(logits, length - 1, axis=1, keepdims=False) / temperature

        key, subkey = jax.random.split(key)
        next_id = jax.random.categorical(subkey, next_logits, axis=-1)

        tokens = tokens.at[:, length].set(next_id)
        return (tokens, length + 1, key), next_id 

    (tokens, _, _), _ = jax.lax.scan(
        step,
        init=(tokens, length, key),
        xs=None,
        length=max_new_tokens)

    return tokens[:, :init_len + max_new_tokens]


@nnx.jit(static_argnums=(0, 1, 2, 3, 4, 5, 6, 7))
def create_sharded_model(
    vocab_size: int,
    dim: int = 512,
    n_layers: int = 8,
    n_heads: int = 8,
    n_kv_heads: Optional[int] = None,
    max_seq_len: int = 2048,
    mlp_ratio: float = 4.0,
    dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 42,
) -> LlamaModel:
    """Create a sharded LlamaModel with automatic partitioning.
    
    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads (for grouped query attention)
        max_seq_len: Maximum sequence length
        mlp_ratio: MLP hidden dimension ratio
        dtype: Model dtype
        seed: PRNG seed

    Returns:
        Sharded LlamaModel instance
        
    Note:
        This function should be called within a mesh context:
        ```
        with mesh:
            model = create_sharded_model(...)
        ```
    """
    rngs = nnx.Rngs(jax.random.PRNGKey(seed))
    model = LlamaModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=max_seq_len,
        mlp_ratio=mlp_ratio,
        dtype=dtype,
        rngs=rngs,
    )
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


def create_pretrained_easydel_sharded_model(
    model_id: str,
    dtype: jnp.dtype = jnp.bfloat16,
) -> nnx.Module:
    """Create a sharded pretrained EasyDeL model with automatic partitioning.
    
    Args:
        model_id: The Hugging Face model id of the pretrained model
        dtype: Model dtype
    Returns:
        The EasyDeL pretrained model, wrapped in a class that returns their logits
    Note:
        This function should be called within a mesh context:
        ```
        with mesh:
            model = create_pretrained_easydel_sharded_model(...)
        ```
        
        Inputs to the resulting model should also be sent to multiple GPUs beforehand,
        as this does not happen within the model itself:
        ```
        input_ids = jax.lax.with_sharding_constraint(input_ids, P('data', None))
        logits = model(input_ids)
        ```
    """
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(model_id, dtype=dtype, param_dtype=dtype, auto_shard_model=False, trust_remote_code=True)
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = int(model.config.hidden_size / model.config.num_attention_heads)
        print(f"WARNING: model head dim is not set in the config so setting it to be hidden_size/num_attention_heads = {head_dim}")
        model.config.head_dim = head_dim
    model.dim = model.config.hidden_size
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)

    # Here we are modifying the easydel model call method to only return logits and nothing else,
    # which is what the dataset-metagradients-jax repo expects.
    _orig_cls = model.__class__
    _orig_call = _orig_cls.__call__
    LogitsOnly = type(
        f"LogitsOnly{_orig_cls.__name__}",
        (_orig_cls,),
        {
            "__call__": lambda self, *args, **kwargs: _orig_call(self, *args, **kwargs).logits
        }
    )
    model.__class__ = LogitsOnly
    return model
