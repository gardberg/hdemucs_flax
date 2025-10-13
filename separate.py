from typing import Union
from pathlib import Path
import jax.numpy as jnp
import jax
from jax import lax
from jax import export
import flax.nnx as nnx

from demucs import HDemucs
import logging

from functools import partial

logger = logging.getLogger(__name__)

class Separator:
    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = None,
        chunk_size: int = 30,
        sample_rate: int = 44100,
        overlap: float = 0.1,
        backend: str = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        """
        Performs chunked audio source separation.

        Args:
            checkpoint_dir: Path to the checkpoint directory.
            chunk_size: Size of the chunk to separate (s).
            sample_rate: Sample rate of the audio.
            overlap: Fraction of chunk_size that is overlapped.
        """
        self.backend = jax.default_backend() if backend is None else backend
        self.model = None
        self._compiled_separate = None
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.chunk_size_samples = int(chunk_size * sample_rate)
        self.overlap_samples = int(self.chunk_size_samples * overlap)
        self.stride = self.chunk_size_samples - self.overlap_samples

        self.dtype = dtype

        self.fade_in = self._get_fade_in_array(self.chunk_size_samples, self.overlap_samples)[None, None, :]
        self.fade_out = self._get_fade_out_array(self.chunk_size_samples, self.overlap_samples)[None, None, :]

        if checkpoint_dir is not None:
            self.load_and_compile(checkpoint_dir)

    def load_and_compile(self, checkpoint_dir: Union[str, Path]):
        """
        backend: cpu or gpu
        """

        from utils import load_checkpoint

        self.model = load_checkpoint(checkpoint_dir, self.dtype)

        logger.info("Compiling and warming up...")
        @nnx.jit()
        def separate_fn(waveform: jnp.ndarray, model: HDemucs):
            return self.separate(model, waveform)

        self._compiled_separate = partial(separate_fn, model=self.model)

        # warmup
        self._compiled_separate(jnp.zeros((2, self.chunk_size_samples), dtype=self.dtype))
        logger.info("JIT compilation and warmup done")


    def export_compiled_separate(self, save_path: Union[str, Path] = None) -> Path:

        # NOTE probably doesnt work with new nnx.jit usage

        exported = export.export(self._compiled_separate)(
            jax.ShapeDtypeStruct((2, self.chunk_size_samples), self.dtype))

        serialized = exported.serialize()

        save_name = f"compiled_separate_{self.chunk_size}_{self.backend}_{self.dtype.__name__}.bin"

        if save_path is None:
            save_path = "."
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        full_path = save_path / save_name
        
        with open(full_path, "wb") as f:
            f.write(serialized)

        return full_path
   
    def load_compiled_separate(self, path: Union[str, Path]):
        # path: path to serialized exported fn to load

        with open(path, "rb") as f:
            serialized = f.read()

        deserialized = export.deserialize(serialized)
        self._compiled_separate = deserialized.call


    def separate_longform(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """
        Performs chunked audio source separation.

        Args:
            waveform: Waveform to separate. Shape: (n_channels, length)

        Returns:
            Separated waveform. Shape: (4, n_channels, length)
        """
        original_length = waveform.shape[1]
        n_channels = waveform.shape[0]
        assert n_channels == 2, f"Got invalid number of channels: {n_channels} != 2"

        # run separation directly on padded input if its short enough
        if original_length <= self.chunk_size_samples:
            padded_waveform = jnp.pad(waveform, ((0, 0), (0, self.chunk_size_samples - original_length)))
            separated_padded = self._compiled_separate(padded_waveform)
            return separated_padded[:, :, :original_length]

        # pad last chunk so we can process all chunks in the same way
        total_strides = (original_length - self.overlap_samples + self.stride - 1) // self.stride
        n_chunks = total_strides + 1 # total number of chunks
        padded_length = self.overlap_samples + total_strides * self.stride
        padding_amount = padded_length - original_length
        padded_waveform = jnp.pad(waveform, ((0, 0), (0, padding_amount))) # (2, padded_length)

        # output buffer
        output_shape = (4, n_channels, padded_length)
        init_output = jnp.zeros(output_shape, dtype=jnp.float32)

        def scan_body(carry_output, i):
            input_start_idx = i * self.stride
            # Slice the input chunk directly from the padded waveform
            current_chunk_input = lax.dynamic_slice(
                padded_waveform,
                (0, input_start_idx),
                (n_channels, self.chunk_size_samples)
            )
            
            output_start_idx = input_start_idx # Output gets placed at the same start index
            
            separated_chunk = self._compiled_separate(current_chunk_input)

            # Apply fades based on chunk index (same logic as before)
            separated_chunk = lax.cond(
                i == 0,
                lambda x: x * self.fade_out,
                lambda x: x,
                separated_chunk
            )
            separated_chunk = lax.cond(
                i == n_chunks - 1,
                lambda x: x * self.fade_in,
                lambda x: x,
                separated_chunk
            )
            separated_chunk = lax.cond(
                (i > 0) & (i < n_chunks - 1),
                lambda x: x * self.fade_in * self.fade_out,
                lambda x: x,
                separated_chunk
            )

            # Add faded chunk to output buffer (same logic as before)
            output_slice = lax.dynamic_slice(
                carry_output,
                (0, 0, output_start_idx),
                separated_chunk.shape
            )
            updated_slice = output_slice + separated_chunk
            carry_output = lax.dynamic_update_slice(
                carry_output,
                updated_slice,
                (0, 0, output_start_idx)
            )
            return carry_output, None

        # Run the scan over the chunk indices
        final_output, _ = lax.scan(scan_body, init_output, jnp.arange(n_chunks))

        return final_output[:, :, :original_length]

    def _reshape_input(self, waveform: jnp.ndarray) -> jnp.ndarray:
        return waveform[None, ...]

    def separate(self, model: HDemucs, waveform: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            waveform: Waveform to separate. Shape: (n_channels, length)

        Returns:
            Separated waveform. Shape: (4, n_channels, length)
        """

        # explicitly use model as arg to include in jit args
        waveform_n = self._reshape_input(waveform)
        output = model(waveform_n)
        output = output.astype(waveform.dtype)
        return output.reshape(4, 2, -1)

    def _get_fade_in_array(self, waveform_length: int, fade_in_len: int) -> jnp.ndarray:
        fade = jnp.linspace(0., 1., fade_in_len, dtype=self.dtype)
        ones = jnp.ones(waveform_length - fade_in_len, dtype=self.dtype)
        return jnp.concatenate((fade, ones), dtype=self.dtype)

    def _get_fade_out_array(self, waveform_length: int, fade_out_len: int) -> jnp.ndarray:
        fade = jnp.linspace(1., 0., fade_out_len, dtype=self.dtype)
        ones = jnp.ones(waveform_length - fade_out_len, dtype=self.dtype)
        return jnp.concatenate((ones, fade), dtype=self.dtype)
