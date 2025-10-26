from typing import Union
from pathlib import Path
import jax.numpy as jnp
import jax
from jax import lax
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
        compile_batches: int = 12
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
        self._compiled_batched_separate = None
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.chunk_size_samples = int(chunk_size * sample_rate)
        self.overlap_samples = int(self.chunk_size_samples * overlap)
        self.stride = self.chunk_size_samples - self.overlap_samples
        self.compile_batches = compile_batches

        self.dtype = dtype

        self.fade_in = self._get_fade_in_array(self.chunk_size_samples, self.overlap_samples)[None, None, :]
        self.fade_out = self._get_fade_out_array(self.chunk_size_samples, self.overlap_samples)[None, None, :]

        self.min_length_samples = (compile_batches - 1) * self.stride + self.overlap_samples
        self.max_length_samples = compile_batches * self.stride + self.overlap_samples - 1
        self.min_length_sec = self.min_length_samples / sample_rate
        self.max_length_sec = self.max_length_samples / sample_rate
        logger.info(f"Compiling for {compile_batches} batches: audio length range {self.min_length_sec:.1f}s - {self.max_length_sec:.1f}s")

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
        def batched_separate_fn(waveforms: jnp.ndarray, model: HDemucs):
            outputs = model(waveforms)
            return outputs.astype(waveforms.dtype)

        self._compiled_batched_separate = partial(batched_separate_fn, model=self.model)

        dummy_input = jax.random.normal(jax.random.PRNGKey(0), (self.compile_batches, 2, self.chunk_size_samples), dtype=self.dtype)
        self._compiled_batched_separate(dummy_input)

        logger.info("JIT compilation and warmup done")

    def separate_longform_batched(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """
        waveform: shape (2, length)
        """
        original_length = waveform.shape[1]
        n_channels = waveform.shape[0]
        assert n_channels == 2, f"Got invalid number of channels: {n_channels} != 2"

        max_length_samples = (self.compile_batches - 1) * self.stride + self.chunk_size_samples

        if original_length <= max_length_samples:
            # pad to pre-compiled length
            padded_length = max_length_samples
            n_chunks = self.compile_batches
            logger.info(f"Separating waveform of length {original_length / self.sample_rate:.1f}s using {n_chunks} chunks (padded to {padded_length / self.sample_rate:.1f}s)")
        else:
            # just run with re-triggered compile
            total_strides = (original_length - self.overlap_samples + self.stride - 1) // self.stride
            n_chunks = total_strides + 1
            padded_length = (n_chunks - 1) * self.stride + self.chunk_size_samples
            logger.info(f"Separating waveform of length {original_length / 44100:.0f}s using {n_chunks} chunks, each of length: {self.chunk_size}s")

        padding_amount = padded_length - original_length
        padded_waveform = jnp.pad(waveform, ((0, 0), (0, padding_amount)))

        chunks = []
        for i in range(n_chunks):
            s = i * self.stride
            chunk = padded_waveform[:, s:s + self.chunk_size_samples]
            chunks.append(chunk)
        chunks_batch = jnp.stack(chunks, axis=0)

        separated_batch = self._compiled_batched_separate(chunks_batch)

        idx = jnp.arange(n_chunks)
        fades = jnp.where(
            (idx[:, None, None, None] == 0),
            self.fade_out,
            jnp.where(
                (idx[:, None, None, None] == n_chunks - 1),
                self.fade_in,
                self.fade_in * self.fade_out,
            ),
        )
        separated_batch = separated_batch * fades

        output = jnp.zeros((4, 2, padded_length), dtype=jnp.float32)

        def add_chunk(i, carry_output):
            start = i * self.stride
            chunk = separated_batch[i]
            current_slice = lax.dynamic_slice(carry_output, (0, 0, start), chunk.shape)
            updated_slice = current_slice + chunk
            result = lax.dynamic_update_slice(carry_output, updated_slice, (0, 0, start))
            return result
        
        output = lax.fori_loop(0, n_chunks, add_chunk, output)
        result = output[:, :, :original_length]
        return result

    def _get_fade_in_array(self, waveform_length: int, fade_in_len: int) -> jnp.ndarray:
        # create linspace in float32 and then case to handle nan issues
        fade = jnp.linspace(0., 1., fade_in_len, dtype=jnp.float32).astype(self.dtype)
        ones = jnp.ones(waveform_length - fade_in_len, dtype=self.dtype)
        return jnp.concatenate((fade, ones), dtype=self.dtype)

    def _get_fade_out_array(self, waveform_length: int, fade_out_len: int) -> jnp.ndarray:
        fade = jnp.linspace(1., 0., fade_out_len, dtype=jnp.float32).astype(self.dtype)
        ones = jnp.ones(waveform_length - fade_out_len, dtype=self.dtype)
        return jnp.concatenate((ones, fade), dtype=self.dtype)
