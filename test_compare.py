from optax import AddDecayedWeightsState
import pytest
from torchaudio.models import HDemucs
import torch
import jax.numpy as jnp
from flax import nnx
import logging

from demucs import ScaledEmbedding, LayerScale, LocalState, BidirectionalLSTM, BLSTM, DConv, TorchConv, HybridEncoderLayer
from utils import torch_module_to_params, copy_torch_params

torch.manual_seed(0)

TOL = 1e-5

# stream handler by default
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@pytest.fixture
def torch_model():

    state_dict_path = "../../services/source-separation/models/hdemucs_high_trained.pt"

    sources = ["drums", "bass", "other", "vocals"]
    model = HDemucs(sources=sources, nfft=4096, depth=6)

    state_dict = torch.load(state_dict_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model

    
def test_groupnorm(torch_model: HDemucs):
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

    
def test_conv1d(torch_model: HDemucs):
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

    
def test_torchconv(torch_model: HDemucs):
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


def test_scaled_embedding(torch_model: HDemucs):

    torch_scaled_embedding = torch_model.freq_emb
    x = torch.randint(0, 512, (10,))
    y = torch_scaled_embedding(x)

    # flax
    nnx_scaled_embedding = ScaledEmbedding(512, 48, scale=1.0, rngs=nnx.Rngs(0))

    nnx_scaled_embedding.embedding.embedding = torch_module_to_params(torch_scaled_embedding)

    nnx_y = nnx_scaled_embedding(x.detach().numpy())

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"ScaledEmbedding diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y), f"y: {y.detach().numpy()}, nnx_y: {nnx_y}"


def test_layer_scale(torch_model: HDemucs):

    torch_layer_scale = torch_model.freq_encoder[0].dconv.layers[0][-1]

    x = torch.randn(1, 48, 1)
    with torch.no_grad():
        y = torch_layer_scale(x)

    # flax
    nnx_layer_scale = LayerScale(48, init=0.0)
    nnx_layer_scale.scale = torch_module_to_params(torch_layer_scale)

    nnx_y = nnx_layer_scale(x.detach().numpy())

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"LayerScale diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y), f"y: {y.detach().numpy()}, nnx_y: {nnx_y}"


@pytest.mark.parametrize("layer_idx", [4, 5])
def test_local_state(torch_model: HDemucs, layer_idx: int):

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
def test_bidirectional_lstm(torch_model: HDemucs, layer_idx: int):
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
def test_blstm(torch_model: HDemucs, layer_idx: int):
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
def test_dconv(torch_model: HDemucs, layer_idx: int):
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



@pytest.mark.parametrize("layer_idx, shape", [(0, (1, 4, 2048, 1)), (4, (2, 768, 3, 3))])
def test_freq_henc_layer(torch_model: HDemucs, layer_idx: int, shape: tuple):
    torch_henc_layer = torch_model.freq_encoder[layer_idx]

    # add_print_hook(torch_henc_layer)

    # a bit tricky to figure out exact shapes to input here
    B, C, F, T = shape # (batch_size, channels, freqs, time)
    x = torch.randn(B, C, F, T)

    with torch.no_grad():
        y = torch_henc_layer(x)

    # flax
    # TODO: Make sure the module config are the same (right now getting group norm where there should be identity)
    nnx_henc_layer = HybridEncoderLayer(
        in_channels=4,
        out_channels=48,
        kernel_size=8,
        norm_groups=4,
        empty=False,
        freq=True,
        norm_type="identity",
        dconv_kw=None,
        pad=True,
        rngs=nnx.Rngs(0))

    nnx_henc_layer = copy_torch_params(torch_henc_layer, nnx_henc_layer)
    nnx_y = nnx_henc_layer(x.detach().numpy())

    assert y.shape == nnx_y.shape, f"y shape: {y.shape} must match nnx_y shape: {nnx_y.shape}"

    diff = jnp.linalg.norm(y.detach().numpy() - nnx_y)
    logger.info(f"Freq HEnc Layer (layer {layer_idx}) diff: {diff}")
    assert jnp.allclose(y.detach().numpy(), nnx_y, atol=TOL), f"l2 norm: {diff}"



def add_print_hook(module: torch.nn.Module):
    def hook_fn(module, input, output):
        if isinstance(input, tuple):  # input is always a tuple
            print(f"{name}, {module.__class__.__name__} input shape:\t {input[0].shape}, norm: {jnp.linalg.norm(input[0].detach().numpy())}")
        print(f"{name}, {module.__class__.__name__} output shape:\t {output.shape}, norm: {jnp.linalg.norm(output.detach().numpy())}")

    for name, module in module.named_modules():
        if len(list(module.children())) == 0:
            module.register_forward_hook(hook_fn)