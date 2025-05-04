from demucs import ScaledEmbedding, LayerScale, LocalState, BidirectionalLSTM, BLSTM, DConv, HybridEncoderLayer, HybridDecoderLayer, HDemucs
import torchaudio
import flax.nnx as nnx
from jax import jit
import jax.numpy as jnp

import pytest
import torch

import logging
logger = logging.getLogger(__name__)

@pytest.fixture
def rngs():
    return nnx.Rngs(0)


def test_jit_embedding(rngs):
    embedding = ScaledEmbedding(rngs=rngs)
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


def test_jit_localstate(rngs):
    localstate = LocalState(channels=8, rngs=rngs)
    x = jnp.ones((1, 8, 10))
    y = localstate(x)

    jit_localstate = jit(localstate)
    y_jit = jit_localstate(x)

    assert jnp.allclose(y, y_jit)

    
def test_jit_lstm_cell(rngs):
    lstm = nnx.LSTMCell(in_features=10, hidden_features=10, rngs=rngs)
    carry = lstm.initialize_carry((1, 10))
    x = jnp.ones((1, 10))
    new_carry, y = lstm(carry, x)
    c, h = new_carry

    jit_lstm = jit(lstm)
    new_carry_jit, y_jit = jit_lstm(carry, x)
    c_jit, h_jit = new_carry_jit

    assert jnp.allclose(c, c_jit)
    assert jnp.allclose(h, h_jit)
    assert jnp.allclose(y, y_jit)


# NOTE: Failing for flax 0.10.5, not for 0.10.2?
def test_jit_lstm():
    rngs = nnx.Rngs(0)
    lstm = nnx.RNN(nnx.LSTMCell(in_features=10, hidden_features=10, rngs=rngs))
    x = jnp.ones((1, 10))
    y = lstm(x)

    jit_lstm = jit(lstm)
    y_jit = jit_lstm(x)

    assert jnp.allclose(y, y_jit)

def test_jit_bidirectional_lstm(rngs):
    lstm = BidirectionalLSTM(num_layers=1, hidden_size=10, input_size=10, rngs=rngs)
    x = jnp.ones((1, 10, 10))
    y = lstm(x)

    jit_lstm = jit(lstm)
    y_jit = jit_lstm(x)

    assert jnp.allclose(y, y_jit)


@pytest.mark.parametrize("length", [10, 1000])
def test_jit_blstm(rngs, length):
    blstm = BLSTM(dim=10, layers=1, skip=True, rngs=rngs)
    x = jnp.ones((1, 10, length))
    y = blstm(x)

    jit_blstm = jit(blstm)
    y_jit = jit_blstm(x)

    assert jnp.allclose(y, y_jit)


@pytest.mark.skip(reason="Static feature to be removed")
@pytest.mark.parametrize("dim", [192, 384])
def test_jit_blstm_static(rngs, dim):
    static_blstm_lengths = {192: 1292, 384: 646}

    blstm = BLSTM(dim=dim, layers=2, skip=True, static_lengths=static_blstm_lengths, rngs=rngs)
    x = jnp.ones((1, dim, static_blstm_lengths[dim]))
    y = blstm(x)

    jit_blstm = jit(blstm)
    y_jit = jit_blstm(x)

    assert jnp.allclose(y, y_jit)


def test_jit_dconv(rngs):
    dconv = DConv(channels=10, rngs=rngs)
    x = jnp.ones((1, 10, 10))
    y = dconv(x)

    jit_dconv = jit(dconv)
    y_jit = jit_dconv(x)
    
    assert jnp.allclose(y, y_jit)


def test_jit_hybrid_encoder_layer_freq(rngs):
    hybrid_encoder_layer = HybridEncoderLayer(in_channels=16, out_channels=16, freq=True, rngs=rngs)
    x = jnp.ones((1, 16, 16, 10))
    y = hybrid_encoder_layer(x)

    jit_hybrid_encoder_layer = jit(hybrid_encoder_layer)
    y_jit = jit_hybrid_encoder_layer(x)

    assert jnp.allclose(y, y_jit)


def test_jit_hybrid_encoder_layer_time(rngs):
    hybrid_encoder_layer = HybridEncoderLayer(in_channels=16, out_channels=16, freq=False, rngs=rngs)
    x = jnp.ones((1, 16, 10))
    y = hybrid_encoder_layer(x)

    jit_hybrid_encoder_layer = jit(hybrid_encoder_layer)
    y_jit = jit_hybrid_encoder_layer(x)

    assert jnp.allclose(y, y_jit)


def test_jit_hybrid_decoder_layer_freq(rngs):
    hybrid_decoder_layer = HybridDecoderLayer(in_channels=16, out_channels=16, freq=True, rngs=rngs)
    x = jnp.ones((1, 16, 16, 10))
    z, y = hybrid_decoder_layer(x, skip=x)

    jit_hybrid_decoder_layer = jit(hybrid_decoder_layer)
    z_jit, y_jit = jit_hybrid_decoder_layer(x, skip=x)

    assert jnp.allclose(z, z_jit)
    assert jnp.allclose(y, y_jit)


def test_jit_hybrid_decoder_layer_time(rngs):
    hybrid_decoder_layer = HybridDecoderLayer(in_channels=16, out_channels=16, freq=False, rngs=rngs)
    x = jnp.ones((1, 16, 10))
    z, y = hybrid_decoder_layer(x, skip=x)

    jit_hybrid_decoder_layer = jit(hybrid_decoder_layer)
    z_jit, y_jit = jit_hybrid_decoder_layer(x, skip=x)

    assert jnp.allclose(z, z_jit)
    assert jnp.allclose(y, y_jit)


@pytest.mark.parametrize("length", [11025, 44100 * 180])
def test_jit_hdemucs(rngs, length):
    hdemucs = HDemucs(sources=["drums", "bass", "other", "vocals"], nfft=4096, depth=6, rngs=rngs)
    x = jnp.ones((1, 2, length))
    y = hdemucs(x)

    jit_hdemucs = jit(hdemucs)
    y_jit = jit_hdemucs(x)

    assert jnp.allclose(y, y_jit)


@pytest.mark.benchmark(group="speed")
def test_flax_jit_speed(benchmark, rngs):
    hdemucs = HDemucs(sources=["drums", "bass", "other", "vocals"], nfft=4096, depth=6, rngs=rngs)
    audio_path = "./testaudio.wav"
    waveform, sample_rate = torchaudio.load(audio_path, format="wav")

    ref = waveform.mean(0)
    waveform_n = (waveform - ref.mean()) / ref.std()

    waveform_n = waveform_n.unsqueeze(0)
    x = jnp.array(waveform_n.numpy())

    hdemucs_jit = jit(hdemucs)

    # warmup / compile
    hdemucs_jit(x)

    benchmark(hdemucs_jit, x)

    
@pytest.mark.benchmark(group="speed30s")
def test_flax_jit_speed_30s(benchmark, rngs):
    hdemucs = HDemucs(sources=["drums", "bass", "other", "vocals"], nfft=4096, depth=6, rngs=rngs)

    audio_path = "./testaudio.wav"
    waveform, sample_rate = torchaudio.load(audio_path, format="wav")

    duration = waveform.shape[1] / sample_rate
    while duration < 30:
        waveform = torch.cat([waveform, waveform], dim=1)
        duration = waveform.shape[1] / sample_rate

    waveform = waveform[:, :30 * sample_rate]
    logger.info(f"Waveform shape: {waveform.shape}")

    ref = waveform.mean(0)
    waveform_n = (waveform - ref.mean()) / ref.std()

    waveform_n = waveform_n.unsqueeze(0)
    x = jnp.array(waveform_n.numpy())

    hdemucs_jit = jit(hdemucs)

    # warmup / compile
    hdemucs_jit(x)

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
