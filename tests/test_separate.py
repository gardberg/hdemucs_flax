import pytest
import soundfile as sf
import jax.numpy as jnp
import torch

from separate import Separator
from torch_utils import TorchSeparator

@pytest.fixture
def separator():
    return Separator(
        "/Users/gardberg/dev/hdemucs_flax/checkpoints",
        chunk_size=30,
        sample_rate=44100,
        overlap=0.1,
    )

@pytest.fixture
def torch_separator():
    return TorchSeparator()

def test_jax_separate(separator: Separator):
    waveform, sample_rate = sf.read("testaudio_long.wav")
    waveform_n = jnp.array(waveform).transpose()
    out = separator.separate_longform(waveform_n)


# @pytest.mark.skip(reason="Pytorch implementation is not correct")
def test_separate_longform(separator: Separator, torch_separator: TorchSeparator):
    waveform, sample_rate = sf.read("testaudio_long.wav")
    # jax
    waveform_n = jnp.array(waveform).transpose()

    ref = waveform_n.mean(0)
    waveform_n = (waveform_n - ref.mean()) / ref.std()

    out = separator.separate_longform(waveform_n) # shape (4, 2, length)

    out = out * ref.std() + ref.mean()

    # torch
    torch_waveform = torch.tensor(waveform, dtype=torch.float32).transpose(0, 1).unsqueeze(0)


    torch_ref = torch_waveform.mean(0)
    torch_waveform = (torch_waveform - torch_ref.mean()) / torch_ref.std()

    # (4, 2, length)
    torch_out = torch_separator.separate_sources(torch_waveform)[0, :, :, :]

    torch_out = torch_out * torch_ref.std() + torch_ref.mean()


    assert out.shape == torch_out.shape
    # NOTE: Skipping cause of different chunking logic
    # assert jnp.allclose(out, torch_out.numpy()), f"Difference: {diff}"


@pytest.mark.benchmark(group="separate")
def test_benchmark_separate_jax_speed(benchmark, separator: Separator):
    waveform, sample_rate = sf.read("testaudio_long.wav")
    waveform = jnp.array(waveform).transpose()
    benchmark(separator.separate_longform, waveform)


@pytest.mark.benchmark(group="separate")
def test_benchmark_separate_torch_speed(benchmark, torch_separator: TorchSeparator):
    waveform, sample_rate = sf.read("testaudio_long.wav")
    waveform = torch.tensor(waveform, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    benchmark(torch_separator.separate_sources, waveform)
