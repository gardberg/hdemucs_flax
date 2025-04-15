import pytest
from torchaudio.models import HDemucs as TorchHDemucs
from torchaudio.models._hdemucs import _spectro, _ispectro
import torch
import jax.numpy as jnp
from flax import nnx
import logging

from demucs import ScaledEmbedding, LayerScale, LocalState, BidirectionalLSTM, BLSTM, DConv, TorchConv, HybridEncoderLayer, HybridDecoderLayer, HDemucs
from utils import copy_torch_params, get_print_hook, print_shapes_hook
from audio_utils import signal_to_spectrogram, spectrogram_to_signal, complex_spec_to_real, real_spec_to_complex
from conv import TransposedConv1d, TransposedConv2d
import torchaudio
from module import intercept_methods
from torch_utils import torch_add_print_hook

torch.manual_seed(0)

TOL = 1e-5

# stream handler by default
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logging.getLogger('jax').setLevel(logging.WARNING)

@pytest.fixture
def torch_model():

    state_dict_path = "models/hdemucs_high_trained.pt"

    sources = ["drums", "bass", "other", "vocals"]
    model = TorchHDemucs(sources=sources, nfft=4096, depth=6)

    state_dict = torch.load(state_dict_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model

    
def test_groupnorm(torch_model: TorchHDemucs):
    torch_groupnorm = torch_model.freq_encoder[0].dconv.layers[0][1]

    CHANNELS = 12
    x = torch.randn(1, CHANNELS, 1) # torch takes shape (batch_size, channels, *)
    with torch.no_grad():
        y = torch_groupnorm(x)

    # flax takes shape (batch_size, *, channels)
    nnx_groupnorm = nnx.GroupNorm(num_groups=1, num_features=CHANNELS, rngs=nnx.Rngs(0))

    nnx_groupnorm = copy_torch_params(torch_groupnorm, nnx_groupnorm)

    nnx_y = nnx_groupnorm(x.detach().numpy().transpose(0, 2, 1))
    nnx_y = nnx_y.transpose(0, 2, 1)

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"GroupNorm diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"difference: {diff}"

    
def test_conv1d(torch_model: TorchHDemucs):
    torch_conv1d = torch_model.freq_encoder[0].dconv.layers[0][0]

    x = torch.randn(1, 48, 1) # (batch_size, channels, time_steps)
    with torch.no_grad():
        y = torch_conv1d(x)
        
    # flax
    nnx_conv1d = nnx.Conv(48, 12, 3, kernel_dilation=1, padding=1, rngs=nnx.Rngs(0))

    nnx_conv1d = copy_torch_params(torch_conv1d, nnx_conv1d)

    nnx_y = nnx_conv1d(x.detach().numpy().transpose(0, 2, 1))
    nnx_y = nnx_y.transpose(0, 2, 1)

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"Conv1D diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y), f"difference: {diff}"

    
def test_torchconv(torch_model: TorchHDemucs):
    torch_conv = torch_model.freq_encoder[0].dconv.layers[0][0]

    x = torch.randn(1, 48, 1)
    with torch.no_grad():
        y = torch_conv(x)
        
    # flax
    nnx_conv = TorchConv(48, 12, 3, kernel_dilation=1, padding=1, rngs=nnx.Rngs(0))

    nnx_conv = copy_torch_params(torch_conv, nnx_conv)

    nnx_y = nnx_conv(x.detach().numpy())

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"TorchConv diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y), f"difference: {diff}"


def test_scaled_embedding(torch_model: TorchHDemucs):

    torch_scaled_embedding = torch_model.freq_emb
    x = torch.randint(0, 512, (10,))
    y = torch_scaled_embedding(x)

    # flax
    nnx_scaled_embedding = ScaledEmbedding(512, 48, scale=torch_scaled_embedding.scale, rngs=nnx.Rngs(0))

    nnx_scaled_embedding = copy_torch_params(torch_scaled_embedding, nnx_scaled_embedding)

    assert nnx_scaled_embedding.scale == torch_scaled_embedding.scale
    assert jnp.allclose(nnx_scaled_embedding.embedding.embedding.value, jnp.array(torch_scaled_embedding.embedding.weight.data))

    nnx_y = nnx_scaled_embedding(x.detach().numpy())

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"ScaledEmbedding diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y), f"y: {y.detach().numpy()}, nnx_y: {nnx_y}"


def test_layer_scale(torch_model: TorchHDemucs):

    torch_layer_scale = torch_model.freq_encoder[0].dconv.layers[0][-1]

    x = torch.randn(1, 48, 1)
    with torch.no_grad():
        y = torch_layer_scale(x)

    # flax
    nnx_layer_scale = LayerScale(48, init=0.0)
    nnx_layer_scale = copy_torch_params(torch_layer_scale, nnx_layer_scale)

    nnx_y = nnx_layer_scale(x.detach().numpy())

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"LayerScale diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y), f"y: {y.detach().numpy()}, nnx_y: {nnx_y}"


@pytest.mark.parametrize("layer_idx", [4, 5])
def test_local_state(torch_model: TorchHDemucs, layer_idx: int):

    torch_local_state = torch_model.freq_encoder[layer_idx].dconv.layers[0][4]

    B, C, T = (1, 192, 1) if layer_idx == 4 else (1, 384, 1)
    x = torch.randn(B, C, T)
    with torch.no_grad():
        y = torch_local_state(x)

    # flax
    nnx_local_state = LocalState(192 if layer_idx == 4 else 384, heads=4, ndecay=4, rngs=nnx.Rngs(0))

    nnx_local_state = copy_torch_params(torch_local_state, nnx_local_state)

    nnx_y = nnx_local_state(x.detach().numpy())

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"LocalState (layer {layer_idx}) diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"y: {y.detach().numpy()}, nnx_y: {nnx_y}, l2 norm: {diff}"


def test_bidirectional_lstm_toy():
    torch_lstm = torch.nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=False, bidirectional=True)

    x = torch.randn(2, 3, 10) # (batch_size, time_steps, channels)

    with torch.no_grad():
        y, _hidden = torch_lstm(x.permute(1, 0, 2))

    y = y.permute(1, 0, 2) # (time_steps, batch_size, 2 * channels) -> (batch_size, time_setps, 2 * channels)
        
    # flax
    nnx_lstm = BidirectionalLSTM(num_layers=torch_lstm.num_layers, hidden_size=torch_lstm.hidden_size, input_size=torch_lstm.input_size, rngs=nnx.Rngs(0))

    nnx_lstm = copy_torch_params(torch_lstm, nnx_lstm)

    nnx_y = nnx_lstm(x.detach().numpy())

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"Bidirectional LSTM (toy) diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff}"


@pytest.mark.parametrize("layer_idx", [4, 5])
def test_bidirectional_lstm(torch_model: TorchHDemucs, layer_idx: int):
    torch_lstm = torch_model.freq_encoder[layer_idx].dconv.layers[0][3].lstm

    C = torch_lstm.input_size
    x = torch.randn(2, 3, C) # (batch_size, time_steps, channels)

    # default torch lstm takes input (time_steps, batch_size, channels)

    with torch.no_grad():
        y, _hidden = torch_lstm(x.permute(1, 0, 2))

    y = y.permute(1, 0, 2) # (time_steps, batch_size, 2 * channels) -> (batch_size, time_setps, 2 * channels)

    # flax
    nnx_lstm = BidirectionalLSTM(num_layers=torch_lstm.num_layers, hidden_size=torch_lstm.hidden_size, input_size=torch_lstm.input_size, rngs=nnx.Rngs(0))
    nnx_lstm = copy_torch_params(torch_lstm, nnx_lstm)
    nnx_y = nnx_lstm(x.detach().numpy())

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"Bidirectional LSTM (layer {layer_idx}) diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff}"


@pytest.mark.parametrize("layer_idx", [4, 5])
def test_blstm(torch_model: TorchHDemucs, layer_idx: int):
    torch_blstm = torch_model.freq_encoder[layer_idx].dconv.layers[0][3]

    B, C, T = (2, 192, 3) if layer_idx == 4 else (2, 384, 3)

    x = torch.randn(B, C, T)

    with torch.no_grad():
        y = torch_blstm(x)

    # flax
    nnx_blstm = BLSTM(dim=C, layers=2, skip=True, rngs=nnx.Rngs(0))
    nnx_blstm = copy_torch_params(torch_blstm, nnx_blstm)
    nnx_y = nnx_blstm(x.detach().numpy())

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"BLSTM (layer {layer_idx}) diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff}"


@pytest.mark.parametrize("layer_idx", [0, 4]) # add 4 as test case
def test_dconv(torch_model: TorchHDemucs, layer_idx: int):
    torch_dconv = torch_model.freq_encoder[layer_idx].dconv

    # add_print_hook(torch_dconv)

    B, C, T = (2, 48, 3) if layer_idx == 0 else (2, 768, 3)
    x = torch.randn(B, C, T)

    with torch.no_grad():
        y = torch_dconv(x)

    # flax
    if layer_idx == 0:
        nnx_dconv = DConv(channels=C, compress=4, depth=2, attn=False, heads=4, ndecay=4, lstm=False, kernel_size=3, rngs=nnx.Rngs(0))
    else:
        nnx_dconv = DConv(channels=C, compress=4, depth=2, attn=True, heads=4, ndecay=4, lstm=True, kernel_size=3, rngs=nnx.Rngs(0))

    nnx_dconv = copy_torch_params(torch_dconv, nnx_dconv)
    nnx_y = nnx_dconv(x.detach().numpy())

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"DConv (layer {layer_idx}) diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff}"



@pytest.mark.parametrize(
    "layer_idx, shape",
    [
        (0, (1, 4, 2048, 1)),
        (1, (1, 48, 512, 1)),
        (4, (1, 384, 8, 1)),
        (5, (1, 768, 1, 1)),  # Last freq encoder layer
    ]
)
def test_freq_henc_layer(torch_model: TorchHDemucs, layer_idx: int, shape: tuple):
    torch_henc_layer = torch_model.freq_encoder[layer_idx]

    # add_print_hook(torch_henc_layer)

    # a bit tricky to figure out exact shapes to input here
    B, C, F, T = shape # (batch_size, channels, freqs, time)
    x = torch.randn(B, C, F, T)

    with torch.no_grad():
        y = torch_henc_layer(x)

    # flax
    param_map = {
        0: {
            "in_channels": 4,
            "out_channels": 48,
            "kernel_size": 8,
            "norm_groups": 4,
            "norm_type": "identity",
            "pad": True,
            "freq": True,
        },
        1: {
            "in_channels": 48,
            "out_channels": 96,
            "kernel_size": 8,
            "norm_groups": 4,
            "norm_type": "identity",
            "pad": True,
            "freq": True,
        },
        4: {
            "in_channels": 384,
            "out_channels": 768,
            "kernel_size": 8,
            "norm_groups": 4,
            "norm_type": "group_norm",
            "dconv_kw": {
                "lstm": True,
                "attn": True,
            },
            "pad": False,
            "freq": True,
        },
        5: {
            "in_channels": 768,
            "out_channels": 1536,
            "kernel_size": 4,
            "stride": 2,
            "norm_groups": 4,
            "norm_type": "group_norm",
            "dconv_kw": {
                "lstm": True,
                "attn": True,
            },
            "pad": True,
            "freq": False, # last layer to merge with time brach?
        }
    }

    nnx_henc_layer = HybridEncoderLayer(
        **param_map[layer_idx],
        empty=False,
        rngs=nnx.Rngs(0))

    nnx_henc_layer = copy_torch_params(torch_henc_layer, nnx_henc_layer)
    nnx_y = nnx_henc_layer(x.detach().numpy())

    assert y.shape == nnx_y.shape, f"y shape: {y.shape} must match nnx_y shape: {nnx_y.shape}"

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"Freq HEnc Layer (layer {layer_idx}) diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff}"

    
@pytest.mark.parametrize(
    "layer_idx, shape",
    [
        (0, (1, 2, 8)),  # First time encoder layer
        (1, (1, 48, 2)),  # Second time encoder layer
        (3, (1, 192, 8)),  # Fourth time encoder layer
        (4, (1, 384, 2)),  # Fifth time encoder layer (empty layer)
    ]
)
def test_time_henc_layer(torch_model: TorchHDemucs, layer_idx: int, shape: tuple):
    torch_henc_layer = torch_model.time_encoder[layer_idx]

    # add_print_hook(torch_henc_layer)

    B, C, T = shape # (batch_size, channels, time_steps)
    x = torch.randn(B, C, T)

    with torch.no_grad():
        y = torch_henc_layer(x)

    # flax
    param_map = {
        0: {
            "in_channels": 2,  # audio_channels
            "out_channels": 48,
            "kernel_size": 8,
            "stride": 4,
            "norm_groups": 4,
            "norm_type": "identity",
            "pad": True,
            "freq": False,  # time encoder
        },
        1: {
            "in_channels": 48,
            "out_channels": 96,
            "kernel_size": 8,
            "stride": 4,
            "norm_groups": 4,
            "norm_type": "identity",
            "pad": True,
            "freq": False,  # time encoder
        },
        3: {
            "in_channels": 192,
            "out_channels": 384,
            "kernel_size": 8,
            "stride": 4,
            "norm_groups": 4,
            "norm_type": "identity",
            "pad": True,
            "freq": False,  # time encoder
        },
        4: {
            "in_channels": 384,
            "out_channels": 768,
            "kernel_size": 8,
            "stride": 4,
            "norm_groups": 4,
            "norm_type": "group_norm",
            "pad": True,
            "freq": False,  # time encoder
            "empty": True,  # This is an empty layer (no dconv, no rewrite)
        }
    }

    nnx_henc_layer = HybridEncoderLayer(
        **param_map[layer_idx],
        rngs=nnx.Rngs(0))

    nnx_henc_layer = copy_torch_params(torch_henc_layer, nnx_henc_layer)
    nnx_y = nnx_henc_layer(x.detach().numpy())

    assert y.shape == nnx_y.shape, f"y shape: {y.shape} must match nnx_y shape: {nnx_y.shape}"

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"Time HEnc Layer (layer {layer_idx}) diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff}"


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, strides, padding, output_padding, dilation, input_shape",
    [
        # Basic case (original test)
        (48, 12, 8, 4, 0, 0, 1, (1, 48, 10)),
        
        # Different channel configurations
        (24, 12, 5, 2, 1, 0, 1, (2, 24, 15)),
        
        # Test with output padding
        (16, 32, 4, 2, 2, 1, 1, (1, 16, 8)),
        
        # Test with dilation
        (8, 16, 3, 1, 1, 0, 2, (3, 8, 12)),
        
        # More complex case with all parameters
        (32, 64, 6, 3, 3, 2, 2, (2, 32, 20)),
    ]
)
def test_transposed_conv1d(
    in_channels, out_channels, kernel_size, strides, 
    padding, output_padding, dilation, input_shape
):
    # Create PyTorch transposed conv
    torch_conv = torch.nn.ConvTranspose1d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size,
        stride=strides,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation
    )

    # Create input tensor with the specified shape
    x = torch.randn(*input_shape)
    logger.info(f"x shape: {x.shape}")

    # Get PyTorch output
    with torch.no_grad():
        y = torch_conv(x)
        
    # Create and initialize equivalent Flax module
    nnx_conv = TransposedConv1d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        rngs=nnx.Rngs(0)
    )

    # Copy parameters from PyTorch to Flax
    nnx_conv = copy_torch_params(torch_conv, nnx_conv)

    # Get Flax output
    nnx_y = nnx_conv(x.detach().numpy())

    # Check that shapes match
    logger.info(f"PyTorch shape: {y.shape}, Flax shape: {nnx_y.shape}")
    assert y.shape == nnx_y.shape, f"PyTorch shape: {y.shape}, Flax shape: {nnx_y.shape}"

    # Compute and log difference
    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"TransposedConv1D diff (in={in_channels}, out={out_channels}, k={kernel_size}, s={strides}): {diff}")
    
    # Assert outputs are close
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff}"


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, strides, padding, output_padding, dilation, input_shape",
    [
        # Basic test case
        (16, 8, 3, 1, 0, 0, 1, (2, 16, 12, 12)),
        
        # Test with non-square kernels and different stride values
        (16, 32, (3, 5), (2, 1), (1, 2), 0, 1, (1, 16, 10, 20)),
        
        # Test with output padding
        (24, 12, 4, 2, 2, (1, 1), 1, (2, 24, 8, 10)),
        
        # Test with dilation
        (8, 16, 3, 2, 1, 0, (2, 1), (1, 8, 10, 8)),
        
        # Complex case with all parameters and non-square everything
        (32, 64, (5, 3), (2, 3), (2, 1), (1, 2), (2, 2), (2, 32, 6, 8)),
    ]
)
def test_transposed_conv2d(
    in_channels, out_channels, kernel_size, strides, 
    padding, output_padding, dilation, input_shape
):
    # Create PyTorch transposed conv
    torch_conv = torch.nn.ConvTranspose2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size,
        stride=strides,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation
    )

    # Create input tensor with the specified shape
    x = torch.randn(*input_shape)
    
    # Get PyTorch output
    with torch.no_grad():
        y = torch_conv(x)
        
    # Create and initialize equivalent Flax module
    nnx_conv = TransposedConv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        rngs=nnx.Rngs(0)
    )

    # Copy parameters from PyTorch to Flax
    nnx_conv = copy_torch_params(torch_conv, nnx_conv)

    # Get Flax output
    nnx_y = nnx_conv(x.detach().numpy())

    # Check that shapes match
    assert y.shape == nnx_y.shape, f"PyTorch shape: {y.shape}, Flax shape: {nnx_y.shape}"

    # Compute and log difference
    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"TransposedConv2D diff (in={in_channels}, out={out_channels}, k={kernel_size}): {diff}")
    
    # Assert outputs are close
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff}"

    

# NOTE: Small numerical error here, unsure why, padding, glu? switched to ones for more deterministic
@pytest.mark.parametrize(
    "layer_idx, shape",
    [
        (0, (1, 1536, 1)), # (batch_size, channels, length)
        (1, (1, 768, 1, 1)), # (batch_size, channels, freqs, length)
        (4, (1, 96, 128, 1)), # (batch_size, channels, freqs, length)
        (5, (1, 48, 512, 1)), # (batch_size, channels, freqs, length)
    ]
)
def test_freq_hdec_layer(torch_model: TorchHDemucs, layer_idx: int, shape: tuple):
    torch_decoder_layer = torch_model.freq_decoder[layer_idx]

    torch_add_print_hook(torch_decoder_layer)

    # x = torch.randn(*shape)
    # skip = torch.randn(*shape)
    x = torch.ones(*shape)
    skip = torch.ones(*shape)
    length = x.shape[-1]

    logger.info(f"x shape: {x.shape}, skip shape: {skip.shape}, length: {length}")

    with torch.no_grad():
        z, y = torch_decoder_layer(x, skip, length)
    
    logger.info(f"y shape: {y.shape}, z shape: {z.shape}")

    # flax
    param_map = {
        0: {
            "in_channels": 1536,
            "out_channels": 768,
            "kernel_size": 4,
            "stride": 2,
            "norm_groups": 4,
            "norm_type": "group_norm",
            "freq": False,
            "last": False,
            "pad": True,
        },
        1: {
            "in_channels": 768,
            "out_channels": 384,
            "kernel_size": 8,
            "stride": 4,
            "freq": True,
            "pad": False,
            "norm_groups": 4,
        },
        4: {
            "in_channels": 96,
            "out_channels": 48,
            "kernel_size": 8,
            "stride": 4,
            "freq": True,
            "norm_groups": 4,
            "norm_type": "identity",
            "pad": True,
        },
        5: {
            "in_channels": 48,
            "out_channels": 16,
            "kernel_size": 8,
            "stride": 4,
            "freq": True,
            "norm_groups": 4,
            "norm_type": "identity",
            "pad": True,
            "last": True,
        }
    }
    
    nnx_decoder_layer = HybridDecoderLayer(
        **param_map[layer_idx],
        rngs=nnx.Rngs(0)
    )

    nnx_decoder_layer = copy_torch_params(torch_decoder_layer, nnx_decoder_layer)

    with intercept_methods(print_shapes_hook):
        nnx_z, nnx_y = nnx_decoder_layer(x.detach().numpy(), skip.detach().numpy(), length)

    logger.info(f"nnx_y shape: {nnx_y.shape}, nnx_z shape: {nnx_z.shape}")

    assert y.shape == nnx_y.shape, f"y shape: {y.shape} must match nnx_y shape: {nnx_y.shape}"

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"Freq Decoder Layer (layer {layer_idx}) diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff:.6f}"

    diff2 = jnp.linalg.norm(z.detach().numpy() - nnx_z)
    logger.info(f"Freq Decoder Layer (layer {layer_idx}) diff2: {diff2}")
    assert jnp.allclose(z.detach().numpy(), nnx_z, atol=TOL), f"l2 norm: {diff2:.6f}"


@pytest.mark.parametrize("layer_idx, shape", [
    (0, (1, 768, 1)), # (batch_size, channels, length)
    (1, (1, 384, 1)),
    (2, (1, 192, 1)),
    (3, (1, 96, 1)),
    (4, (1, 48, 1)),
])
def test_time_hdec_layer(torch_model: TorchHDemucs, layer_idx: int, shape: tuple):
    torch_decoder_layer = torch_model.time_decoder[layer_idx]

    logger.info(f"torch_decoder_layer.pad: {torch_decoder_layer.pad}")
    logger.info(f"torch_decoder_layer.last: {torch_decoder_layer.last}")
    logger.info(f"torch_decoder_layer.freq: {torch_decoder_layer.freq}")
    logger.info(f"torch_decoder_layer.empty: {torch_decoder_layer.empty}")

    torch_add_print_hook(torch_decoder_layer)

    x = torch.randn(*shape)
    skip = torch.randn(*shape) if layer_idx != 0 else None
    length = x.shape[-1]

    with torch.no_grad():
        z, y = torch_decoder_layer(x, skip, length)

    logger.info(f"y shape: {y.shape}, z shape: {z.shape}")

    # flax
    param_map = {
        0: {
            "in_channels": 768,
            "out_channels": 384,
            "kernel_size": 8,
            "stride": 4,
            "norm_groups": 4,
            "norm_type": "group_norm",
            "freq": False,
            "last": False,
            "pad": True,
            "empty": True,
        },
        1: {
            "in_channels": 384,
            "out_channels": 192,
            "kernel_size": 8,
            "stride": 4,
            "norm_type": "identity",
            "freq": False,
            "pad": True,
        },
        2: {
            "in_channels": 192,
            "out_channels": 96,
            "kernel_size": 8,
            "stride": 4,
            "norm_type": "identity",
            "freq": False,
            "pad": True,
        },
        3: {
            "in_channels": 96,
            "out_channels": 48,
            "kernel_size": 8,
            "stride": 4,
            "norm_type": "identity",
            "freq": False,
            "pad": True,
        },
        4: {
            "in_channels": 48,
            "out_channels": 8,
            "kernel_size": 8,
            "stride": 4,
            "norm_type": "identity",
            "freq": False,
            "pad": True,
            "last": True,
        }
    }

    nnx_decoder_layer = HybridDecoderLayer(
        **param_map[layer_idx],
        rngs=nnx.Rngs(0)
    )

    nnx_decoder_layer = copy_torch_params(torch_decoder_layer, nnx_decoder_layer)

    x_jnp = x.detach().numpy()
    skip_jnp = skip.detach().numpy() if skip is not None else None

    with intercept_methods(print_shapes_hook):
        nnx_z, nnx_y = nnx_decoder_layer(x_jnp, skip_jnp, length)

    logger.info(f"nnx_y shape: {nnx_y.shape}, nnx_z shape: {nnx_z.shape}")
    
    assert y.shape == nnx_y.shape, f"y shape: {y.shape} must match nnx_y shape: {nnx_y.shape}"

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"Time Decoder Layer (layer {layer_idx}) diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff:.6f}"

    diff2 = jnp.linalg.norm(z.detach().numpy() - nnx_z)
    logger.info(f"Time Decoder Layer (layer {layer_idx}) diff2: {diff2}")
    assert jnp.allclose(z.detach().numpy(), nnx_z, atol=TOL), f"l2 norm: {diff2:.6f}"


def test_hdemucs_setup(torch_model: TorchHDemucs):
    sources = ["drums", "bass", "other", "vocals"]
    flax_model = HDemucs(sources=sources, nfft=4096, depth=6, rngs=nnx.Rngs(0))

    logger.info(flax_model)

    assert len(flax_model.freq_encoder) == len(torch_model.freq_encoder)
    assert len(flax_model.time_encoder) == len(torch_model.time_encoder)
    assert len(flax_model.freq_decoder) == len(torch_model.freq_decoder)
    assert len(flax_model.time_decoder) == len(torch_model.time_decoder)


def test_hdemucs_copy(torch_model: TorchHDemucs):
    sources = ["drums", "bass", "other", "vocals"]
    flax_model = HDemucs(sources=sources, nfft=4096, depth=6, rngs=nnx.Rngs(0))

    flax_model = copy_torch_params(torch_model, flax_model)

    assert len(flax_model.freq_encoder) == len(torch_model.freq_encoder)
    assert len(flax_model.time_encoder) == len(torch_model.time_encoder)
    assert len(flax_model.freq_decoder) == len(torch_model.freq_decoder)
    assert len(flax_model.time_decoder) == len(torch_model.time_decoder)

    flax_freq_emb_weight = flax_model.freq_emb.embedding.embedding.value
    torch_freq_emb_weight = torch_model.freq_emb.embedding.weight.data
    print(f"flax_freq_emb_weight shape: {flax_freq_emb_weight.shape}, torch_freq_emb_weight shape: {torch_freq_emb_weight.shape}")
    assert flax_freq_emb_weight.shape == torch_freq_emb_weight.shape

    diff = jnp.linalg.norm(flax_freq_emb_weight - torch_freq_emb_weight.numpy())
    print(f"flax_freq_emb_weight diff: {diff}")
    assert jnp.allclose(flax_freq_emb_weight, torch_freq_emb_weight.numpy(), atol=TOL), f"l2 norm: {diff:.6f}"


"""Spectrogram related tests
NOTE: Call order is:
(torch)
_spec
    _spectro
        torch.stft
_magnitude
...
_mask
_ispec
    _ispectro
        torch.istft

(flax)
_spec
    signal_to_spectrogram
        jsig.stft
complex_spec_to_real
...
real_spec_to_complex
_ispec
    spectrogram_to_signal
        jsig.istft
"""

@pytest.mark.parametrize("shape", [
    (1, 2, 16, 100),
    (1, 2, 8, 20),
])
def test_complex_spec_to_real(torch_model: TorchHDemucs, shape: tuple):
    x = torch.randn(*shape, dtype=torch.complex64)

    with torch.no_grad():
        z_torch = torch_model._magnitude(x)

    z_jax = complex_spec_to_real(x.detach().numpy())

    assert z_torch.shape == z_jax.shape, f"z_torch shape: {z_torch.shape} must match z_jax shape: {z_jax.shape}"

    diff = jnp.linalg.norm(z_torch.detach().numpy() - z_jax)
    logger.info(f"complex_spec_to_real diff: {diff}")
    assert jnp.allclose(z_torch.detach().numpy(), z_jax, atol=TOL), f"l2 norm: {diff:.6f}"


@pytest.mark.parametrize("shape", [
    (1, 2, 2, 16, 100),
    (1, 2, 4, 8, 50),
])
def test_real_spec_to_complex(torch_model: TorchHDemucs, shape: tuple):
    x = torch.randn(*shape, dtype=torch.float32)

    with torch.no_grad():
        z_torch = torch_model._mask(x)

    z_jax = real_spec_to_complex(x.detach().numpy())

    assert z_torch.shape == z_jax.shape, f"z_torch shape: {z_torch.shape} must match z_jax shape: {z_jax.shape}"

    diff = jnp.linalg.norm(z_torch.detach().numpy() - z_jax)
    logger.info(f"real_spec_to_complex diff: {diff}")
    assert jnp.allclose(z_torch.detach().numpy(), z_jax, atol=TOL), f"l2 norm: {diff:.6f}"



# corresponds to _spectro
@pytest.mark.parametrize(
    "hop_length, shape", [
        (128, (1, 2, 44100)),
        (256, (1, 2, 22050)),
    ]
)
def test_signal_to_spectrogram(hop_length: int, shape: tuple):
    x = torch.randn(*shape)
    
    # PyTorch version
    z_torch = _spectro(x, hop_length=hop_length)
   
    # JAX version
    z_jax = signal_to_spectrogram(x.detach().numpy(), hop_length=hop_length)
   
    # Print difference statistics
    diff = z_torch.detach().numpy() - z_jax

    assert z_torch.shape == z_jax.shape, f"z_torch shape: {z_torch.shape} must match z_jax shape: {z_jax.shape}"

    diff = jnp.linalg.norm(z_torch.detach().numpy() - z_jax)
    logger.info(f"signal_to_spectrogram diff: {diff}")
    assert jnp.allclose(z_torch.detach().numpy(), z_jax, atol=TOL), f"l2 norm: {diff:.6f}"


# corresponds to _ispectro
@pytest.mark.parametrize(
    "hop_length, shape", [
        (10, (1, 2, 16, 100)),  # (batch_size, channels, freqs, time_steps)
        # (10, (1, 2, 16, 50)),
    ]
)
def test_spectrogram_to_signal(hop_length: int, shape: tuple):
    z = torch.randn(*shape, dtype=torch.complex64)

    # NOTE: Which lengths are used in forward pass?
    length = 50

    with torch.no_grad():
        x_torch = _ispectro(z, hop_length=hop_length, length=length)

    x_jax = spectrogram_to_signal(z.detach().numpy(), hop_length=hop_length, length=length)

    assert x_torch.shape == x_jax.shape, f"x_torch shape: {x_torch.shape} must match x_jax shape: {x_jax.shape}"

    diff = jnp.linalg.norm(x_torch.detach().numpy() - x_jax)
    logger.info(f"spectrogram_to_signal diff: {diff}")
    assert jnp.allclose(x_torch.detach().numpy(), x_jax, atol=TOL), f"l2 norm: {diff:.6f}"


@pytest.mark.parametrize("shape", [
    (1, 2, 44100),
    (1, 2, 22050),
])
def test_hdemucs_spec(torch_model: TorchHDemucs, shape: tuple):
    sources = ["drums", "bass", "other", "vocals"]
    flax_model = HDemucs(sources=sources, nfft=4096, depth=6, rngs=nnx.Rngs(0))

    x = torch.randn(*shape)
    
    with torch.no_grad():
        z_torch = torch_model._spec(x)

    with intercept_methods(print_shapes_hook):
        z_flax = flax_model._spec(x.detach().numpy())

    assert z_torch.shape == z_flax.shape, f"z_torch shape: {z_torch.shape} must match z_flax shape: {z_flax.shape}"

    diff = jnp.linalg.norm(z_torch.detach().numpy() - z_flax)
    logger.info(f"hdemucs_spec diff: {diff}")
    assert jnp.allclose(z_torch.detach().numpy(), z_flax, atol=TOL), f"l2 norm: {diff:.6f}"



@pytest.mark.parametrize("shape", [
    (1, 4, 2, 2048, 100), # (batch_size, extra_dim, channels, freqs, time_steps)
    (1, 4, 2, 2048, 50),
    (2, 8, 2, 2048, 50),
])
def test_hdemucs_ispec(torch_model: TorchHDemucs, shape: tuple):
    sources = ["drums", "bass", "other", "vocals"]
    flax_model = HDemucs(sources=sources, nfft=4096, depth=6, rngs=nnx.Rngs(0))

    x = torch.randn(*shape, dtype=torch.complex64)
    length = x.shape[-1]
    
    with torch.no_grad():
        z_torch = torch_model._ispec(x, length)

    # with intercept_methods(print_shapes_hook):
    z_flax = flax_model._ispec(x.detach().numpy(), length)

    assert z_torch.shape == z_flax.shape, f"z_torch shape: {z_torch.shape} must match z_flax shape: {z_flax.shape}"

    diff = jnp.linalg.norm(z_torch.detach().numpy() - z_flax)
    logger.info(f"hdemucs_ispec diff: {diff}")
    assert jnp.allclose(z_torch.detach().numpy(), z_flax, atol=TOL), f"l2 norm: {diff:.6f}"


@pytest.mark.parametrize("shape", [
    (1, 2, 22050),
    (1, 2, 44100),
    (1, 2, 180000),
])
def test_hdemucs_forward(torch_model: TorchHDemucs, shape: tuple):
    sources = ["drums", "bass", "other", "vocals"]
    flax_model = HDemucs(sources=sources, nfft=4096, depth=6, rngs=nnx.Rngs(0))

    x = torch.randn(*shape)

    # torch_add_print_hook(torch_model, log_to_file=True)

    with torch.no_grad():
        y = torch_model(x)

    flax_model = copy_torch_params(torch_model, flax_model)

    # with intercept_methods(get_print_hook(log_to_file=True)):
    y_flax = flax_model(x.detach().numpy())

    assert y.shape == y_flax.shape, f"y shape: {y.shape} must match y_flax shape: {y_flax.shape}"

    diff = jnp.linalg.norm(y.detach().numpy() - y_flax)
    logger.info(f"HDemucs forward diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), y_flax, atol=TOL), f"l2 norm: {diff:.6f}"

    # make sure each separated track matches (batch_size, tracks, channels, length)
    for i in range(4):
        diff = jnp.linalg.norm(y[:, i].detach().numpy() - y_flax[:, i])
        logger.info(f"HDemucs forward diff for track {i}: {diff}")
        assert jnp.allclose(y[:, i].detach().numpy(), y_flax[:, i], atol=TOL), f"l2 norm: {diff:.6f}"

        
def test_hdemucs_forward_real_audio(torch_model: TorchHDemucs):
    sources = ["drums", "bass", "other", "vocals"]

    audio_path = "./testaudio.wav"
    waveform, sample_rate = torchaudio.load(audio_path, format="wav")

    # normalize
    ref = waveform.mean(0)
    waveform_n = (waveform - ref.mean()) / ref.std()

    waveform_n = waveform_n.unsqueeze(0)

    # torch_add_print_hook(torch_model, log_to_file=True)

    with torch.no_grad():
        y = torch_model(waveform_n) # (1, 4, 2, length)

    # flax
    flax_model = HDemucs(sources=sources, nfft=4096, depth=6, rngs=nnx.Rngs(0))
    flax_model = copy_torch_params(torch_model, flax_model)

    # with intercept_methods(get_print_hook(log_to_file=True)):
    y_flax = flax_model(waveform_n.numpy()) # (1, 4, 2, length)

    assert y.shape == y_flax.shape, f"y shape: {y.shape} must match y_flax shape: {y_flax.shape}"

    diff = jnp.linalg.norm(y.detach().numpy() - y_flax)
    logger.info(f"HDemucs forward diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), y_flax, atol=TOL), f"l2 norm: {diff:.6f}"

    for i in range(4):
        diff = jnp.linalg.norm(y[:, i].detach().numpy() - y_flax[:, i])
        logger.info(f"HDemucs forward diff for track {i}: {diff}")
        assert jnp.allclose(y[:, i].detach().numpy(), y_flax[:, i], atol=TOL), f"l2 norm: {diff:.6f}"

