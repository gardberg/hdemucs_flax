from demucs import ScaledEmbedding, LayerScale, LocalState, BidirectionalLSTM, BLSTM, DConv, HybridEncoderLayer, HybridDecoderLayer, HDemucs

import flax.nnx as nnx
from jax import jit
import jax.numpy as jnp

import pytest


def test_jit_embedding():
    embedding = ScaledEmbedding(rngs=nnx.Rngs(0))
    x = jnp.ones((10,), dtype=jnp.int32)
    y = embedding(x)

    jit_embedding = jit(embedding)
    y_jit = jit_embedding(x)

    assert jnp.allclose(y, y_jit)


def test_jit_layerscale():
    layerscale = LayerScale(channels=10)
    x = jnp.ones((10,))
    y = layerscale(x)

    jit_layerscale = jit(layerscale)
    y_jit = jit_layerscale(x)

    assert jnp.allclose(y, y_jit)


def test_jit_localstate():
    localstate = LocalState(channels=8, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 8, 10))
    y = localstate(x)

    jit_localstate = jit(localstate)
    y_jit = jit_localstate(x)

    assert jnp.allclose(y, y_jit)


def test_jit_bidirectional_lstm():
    lstm = BidirectionalLSTM(num_layers=1, hidden_size=10, input_size=10, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 10, 10))
    y = lstm(x)

    jit_lstm = jit(lstm)
    y_jit = jit_lstm(x)

    assert jnp.allclose(y, y_jit)


def test_jit_blstm():
    blstm = BLSTM(dim=10, layers=1, skip=True, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 10, 10))
    y = blstm(x)

    jit_blstm = jit(blstm)
    y_jit = jit_blstm(x)

    assert jnp.allclose(y, y_jit)


def test_jit_dconv():
    dconv = DConv(channels=10, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 10, 10))
    y = dconv(x)

    jit_dconv = jit(dconv)
    y_jit = jit_dconv(x)
    
    assert jnp.allclose(y, y_jit)


def test_jit_hybrid_encoder_layer_freq():
    hybrid_encoder_layer = HybridEncoderLayer(in_channels=16, out_channels=16, freq=True, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 16, 16, 10))
    y = hybrid_encoder_layer(x)

    jit_hybrid_encoder_layer = jit(hybrid_encoder_layer)
    y_jit = jit_hybrid_encoder_layer(x)

    assert jnp.allclose(y, y_jit)


def test_jit_hybrid_encoder_layer_time():
    hybrid_encoder_layer = HybridEncoderLayer(in_channels=16, out_channels=16, freq=False, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 16, 10))
    y = hybrid_encoder_layer(x)

    jit_hybrid_encoder_layer = jit(hybrid_encoder_layer)
    y_jit = jit_hybrid_encoder_layer(x)

    assert jnp.allclose(y, y_jit)


def test_jit_hybrid_decoder_layer_freq():
    hybrid_decoder_layer = HybridDecoderLayer(in_channels=16, out_channels=16, freq=True, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 16, 16, 10))
    z, y = hybrid_decoder_layer(x, skip=x)

    jit_hybrid_decoder_layer = jit(hybrid_decoder_layer)
    z_jit, y_jit = jit_hybrid_decoder_layer(x, skip=x)

    assert jnp.allclose(z, z_jit)
    assert jnp.allclose(y, y_jit)


def test_jit_hybrid_decoder_layer_time():
    hybrid_decoder_layer = HybridDecoderLayer(in_channels=16, out_channels=16, freq=False, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 16, 10))
    z, y = hybrid_decoder_layer(x, skip=x)

    jit_hybrid_decoder_layer = jit(hybrid_decoder_layer)
    z_jit, y_jit = jit_hybrid_decoder_layer(x, skip=x)

    assert jnp.allclose(z, z_jit)
    assert jnp.allclose(y, y_jit)


def test_jit_hdemucs():
    hdemucs = HDemucs(sources=["drums", "bass", "other", "vocals"], nfft=4096, depth=6, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 2, 22050))
    y = hdemucs(x)

    jit_hdemucs = jit(hdemucs)
    y_jit = jit_hdemucs(x)

    assert jnp.allclose(y, y_jit)


@pytest.mark.benchmark(group="speed")
def test_flax_jit_speed(benchmark):
    hdemucs = HDemucs(sources=["drums", "bass", "other", "vocals"], nfft=4096, depth=6, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 2, 22050))

    hdemucs_jit = jit(hdemucs)

    benchmark(hdemucs_jit, x)


@pytest.mark.skip(reason="Torchaudio HDemucs implementation is not jitable")
@pytest.mark.benchmark(group="speed")
def test_torch_jit_speed(benchmark):
    from torchaudio.models._hdemucs import HDemucs as TorchHDemucs
    import torch

    x = torch.ones((1, 2, 22050))

    state_dict_path = "models/hdemucs_high_trained.pt"

    sources = ["drums", "bass", "other", "vocals"]
    model = TorchHDemucs(sources=sources, nfft=4096, depth=6)

    state_dict = torch.load(state_dict_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    torch_model_jit = torch.jit.script(model)

    benchmark(torch_model_jit, x)
