from demucs import ScaledEmbedding, LayerScale, LocalState, BidirectionalLSTM, BLSTM, DConv, HybridEncoderLayer, HybridDecoderLayer, HDemucs
import flax.nnx as nnx
import pytest
import logging
from pathlib import Path

from separate import Separator
logger = logging.getLogger(__name__)
import jax.numpy as jnp
import jax

def test_leak_embedding():
    embedding = ScaledEmbedding(rngs=nnx.Rngs(0))
    x = jnp.ones((10,), dtype=jnp.int32)

    with jax.check_tracer_leaks():
        y = embedding(x)

def test_leak_bidirectional():
    rngs = nnx.Rngs(0)
    lstm = BidirectionalLSTM(num_layers=1, hidden_size=10, input_size=10, rngs=rngs)
    x = jnp.ones((1, 10, 10))

    @nnx.jit
    def forward(lstm, x):
        return lstm(x)

    with jax.check_tracer_leaks():
        y = forward(lstm, x)

def test_jit_hdemucs():
    hdemucs = HDemucs(sources=["drums", "bass", "other", "vocals"], nfft=4096, depth=6, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 2, 44100))

    # wrap in function to avoid problem with internal bidirectiona rng state being incremented
    # and treated as a side effect
    @nnx.jit
    def forward(hdemucs, x):
        return hdemucs(x)
    
    with jax.check_tracer_leaks():
        y = forward(hdemucs, x)

        
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_jit_separator(dtype):
    checkpoint_dir = Path(f"./checkpoint_{dtype.__name__}")

    x = jnp.ones((2, 44100))

    with jax.check_tracer_leaks():
        separator = Separator(checkpoint_dir, dtype=dtype)

        y = separator.separate_longform(x)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_jit_batched(dtype):
    checkpoint_dir = Path(f"./checkpoint_{dtype.__name__}")

    x = jnp.ones((2, 44100))

    with jax.check_tracer_leaks():
        separator = Separator(checkpoint_dir, dtype=dtype, batched=True)

        y = separator.separate_longform_batched(x)

        assert jnp.isfinite(y).all()

import time

@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_jit_compare(dtype):
    checkpoint_dir = Path(f"./checkpoint_{dtype.__name__}")
    x = jnp.ones((2, 44100))

    separator = Separator(checkpoint_dir, dtype=dtype)
    t0 = time.time()
    out_longform = separator.separate_longform(x)
    t1 = time.time()
    runtime_longform = t1 - t0

    separator_batched = Separator(checkpoint_dir, dtype=dtype, batched=True)
    t0b = time.time()
    out_batched = separator_batched.separate_longform_batched(x)
    t1b = time.time()
    runtime_batched = t1b - t0b

    print(f"{dtype = } longform runtime: {runtime_longform}")
    print(f"{dtype = } batched runtime: {runtime_batched}")

    assert jnp.allclose(out_longform, out_batched, atol=1e-4 if dtype==jnp.float32 else 1e-2)
