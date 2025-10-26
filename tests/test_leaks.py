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
def test_jit_batched(dtype):
    checkpoint_dir = Path(f"./checkpoint_{dtype.__name__}")

    x = jnp.ones((2, 44100))

    with jax.check_tracer_leaks():
        separator = Separator(checkpoint_dir, dtype=dtype)

        y = separator.separate_longform_batched(x)

        assert jnp.isfinite(y).all()

@pytest.mark.parametrize("dtype", [jnp.float16])
def test_jit_real(dtype):
    import soundfile as sf
    waveform, sample_rate = sf.read("testaudio_long.wav")
    waveform_n = jnp.array(waveform).transpose()
    print(f"waveform shape: {waveform_n.shape}")

    checkpoint_dir = Path(f"./checkpoint_{dtype.__name__}")
    separator = Separator(checkpoint_dir, dtype=dtype)

    out = separator.separate_longform_batched(waveform_n)
    print(out.shape)

    mask = ~jnp.isfinite(out)
    if mask.any():
        print("Nonfinite values in output:")
        idx = jnp.argwhere(mask)
        print(idx)
        print(out[mask])

    assert jnp.isfinite(out).all()

