import torch
import logging
import pytest
import jax.numpy as jnp
import torchaudio
import flax.nnx as nnx
import jax
from torchaudio.models import HDemucs as TorchHDemucs

from demucs import HDemucs
from utils import copy_torch_params, get_print_hook, print_shapes_hook

from module import intercept_methods

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

    with intercept_methods(get_print_hook(log_to_file=True)):
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


# Compare model speeds via 'pytest-benchmark' plugin
@pytest.mark.benchmark(group="speed")
def test_torch_speed(benchmark, torch_model: TorchHDemucs):
    logger.info(f"Using torch default backend: {next(torch_model.parameters()).device}")

    audio_path = "./testaudio.wav"
    waveform, sample_rate = torchaudio.load(audio_path, format="wav")

    ref = waveform.mean(0)
    waveform_n = (waveform - ref.mean()) / ref.std()

    waveform_n = waveform_n.unsqueeze(0)

    def torch_forward(x):
        with torch.no_grad():
            return torch_model(x)

    result = benchmark(torch_forward, waveform_n)
    logger.info(f"Torch forward time: {result}")


@pytest.mark.benchmark(group="speed")
def test_flax_speed(benchmark, torch_model: TorchHDemucs):
    logger.info(f"Using JAX default backend: {jax.default_backend()}")

    sources = ["drums", "bass", "other", "vocals"]

    audio_path = "./testaudio.wav"
    waveform, sample_rate = torchaudio.load(audio_path, format="wav")

    ref = waveform.mean(0)
    waveform_n = (waveform - ref.mean()) / ref.std()

    waveform_n = waveform_n.unsqueeze(0)

    flax_model = HDemucs(sources=sources, nfft=4096, depth=6, rngs=nnx.Rngs(0))
    flax_model = copy_torch_params(torch_model, flax_model)

    def flax_forward(x):
        return flax_model(x)

    result = benchmark(flax_forward, jnp.array(waveform_n.numpy()))
    logger.info(f"Flax forward time: {result}")
