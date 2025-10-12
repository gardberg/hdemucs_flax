import pytest
import jax
import jax.numpy as jnp
import logging
import tempfile
from pathlib import Path
from flax import nnx
from torchaudio.models import HDemucs as TorchHDemucs
import soundfile as sf

from demucs import HDemucs
from torch_utils import copy_torch_params
from utils import save_checkpoint, load_checkpoint, get_print_hook
from module import intercept_methods
from separate import Separator

logger = logging.getLogger(__name__)


@pytest.fixture
def torch_model():
    model = TorchHDemucs(sources=["drums", "bass", "other", "vocals"], nfft=4096, depth=6)
    model.eval()
    return model

@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_checkpoint_workflow(torch_model, dtype):
    flax_model = HDemucs(dtype=dtype, rngs=nnx.Rngs(0))
    flax_model = copy_torch_params(torch_model, flax_model, dtype=dtype)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        checkpoint_path = save_checkpoint(flax_model, checkpoint_dir)
        
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        loaded_model = load_checkpoint(checkpoint_dir, dtype=dtype)
        
        waveform, sample_rate = sf.read("testaudio_long.wav")
        waveform = jnp.array(waveform, dtype=dtype).transpose()
        waveform = waveform.reshape(1, 2, waveform.shape[-1])
        
        # with intercept_methods(get_print_hook()):
        logger.info(f"Running forward pass on waveform: {waveform.shape}")
        output = loaded_model(waveform)
        
        assert output.shape == (1, 4, 2, waveform.shape[-1])
        assert output.dtype == dtype
        
        state = nnx.state(loaded_model)
        dtypes = set()
        for param in jax.tree.leaves(state):
            if hasattr(param, 'dtype') and jnp.issubdtype(param.dtype, jnp.floating):
                dtypes.add(param.dtype)

        logger.info(f"Dtypes in state: {dtypes}")
        assert jnp.dtype(dtype) in dtypes

        # export
        logger.info(f"Creating separator with checkpoint_dir: {checkpoint_path} and dtype: {dtype}")
        separator = Separator(checkpoint_dir=checkpoint_path, dtype=dtype)

        logger.info(f"Exporting compiled separate")
        exported_path = separator.export_compiled_separate(save_path=Path(tmpdir) / "exported")

        logger.info(f"Loading compiled separate from {exported_path}")
        new_separator = Separator(dtype=dtype)
        new_separator.load_compiled_separate(exported_path)

        logger.info(f"Running separate longform on waveform: {waveform.shape}")
        waveform = waveform.reshape(2, -1)
        _output = new_separator.separate_longform(waveform)

        logger.info(f"Output shape: {_output.shape}")
