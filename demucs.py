from flax import nnx
from jax import Array
import jax.numpy as jnp
import jax.nn as nn
from functools import partial
from typing import Optional, Dict, Any, Union, List
import math
import jax.scipy.signal as jsig

from conv import TransposedConv1d, TransposedConv2d
from audio_utils import signal_to_spectrogram, spectrogram_to_signal, complex_spec_to_real, real_spec_to_complex
from module import Module

import logging
logger = logging.getLogger(__name__)

class ScaledEmbedding(Module):
    def __init__(
        self,
        n_emb: int = 512,
        emb_dim: int = 48,
        scale: float = 10.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.embedding = nnx.Embed(n_emb, emb_dim, rngs=rngs)
        self.scale = scale

        # Don't care about 'smooth' initialization here
        # since we only support inference

    def __call__(self, x: Array) -> Array:
        """
        x: (n_emb,) indices of embeddings to look up
        return (n_emb, emb_dim)
        """
        return self.embedding(x) * self.scale


class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input: Array) -> Array:
        return input


class LayerScale(Module):
    def __init__(self, channels: int, init: float = 0):
        self.scale = nnx.Param(jnp.zeros(channels) + init)

    def __call__(self, x: Array) -> Array:
        return self.scale[:, None] * x
        
        
class LocalState(Module):
    def __init__(
        self,
        channels: int,
        heads: int = 4,
        ndecay: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        if channels % heads != 0:
            raise ValueError("Channels must be divisible by heads.")
        self.heads = heads
        self.ndecay = ndecay
        
        self.content = nnx.Conv(channels, channels, 1, rngs=rngs)
        self.query = nnx.Conv(channels, channels, 1, rngs=rngs)
        self.key = nnx.Conv(channels, channels, 1, rngs=rngs)
        
        self.query_decay = nnx.Conv(channels, heads * ndecay, 1, rngs=rngs)

        if ndecay: pass # NOTE Only inferece, so initialization doesnt matter
        
        self.proj = nnx.Conv(channels + heads * 0, channels, 1, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """LocalState forward call with input/output shape (B, C, T)"""
        B, C, T = x.shape
        
        # Convert to Flax format (B, T, C)
        x_flax = jnp.transpose(x, (0, 2, 1))
        
        # Apply convolutions
        queries_flax = self.query(x_flax)
        keys_flax = self.key(x_flax)
        content_flax = self.content(x_flax)
        
        # Reshape for attention computation
        queries = jnp.transpose(queries_flax, (0, 2, 1)).reshape(B, self.heads, -1, T)
        keys = jnp.transpose(keys_flax, (0, 2, 1)).reshape(B, self.heads, -1, T)
        content = jnp.transpose(content_flax, (0, 2, 1)).reshape(B, self.heads, -1, T)
        
        # Compute attention scores
        dots = jnp.einsum("bhct,bhcs->bhts", keys, queries) / jnp.sqrt(keys.shape[2])
        
        # Apply decay if needed
        if self.ndecay:
            # Create position delta matrix
            delta = jnp.arange(T, dtype=x.dtype)[:, None] - jnp.arange(T, dtype=x.dtype)[None, :]
            
            # Get decay parameters
            decay_q_flax = self.query_decay(x_flax)
            decay_q = jnp.transpose(decay_q_flax, (0, 2, 1)).reshape(B, self.heads, -1, T)
            decay_q = nn.sigmoid(decay_q) / 2
            
            # Apply decay based on position differences
            decays = jnp.arange(1, self.ndecay + 1, dtype=x.dtype)
            decay_kernel = -decays.reshape(-1, 1, 1) * jnp.abs(delta) / jnp.sqrt(self.ndecay)
            dots = dots + jnp.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Mask self-attention
        dots = dots.at[..., jnp.arange(T), jnp.arange(T)].set(-100)
        
        # Apply attention
        weights = nn.softmax(dots, axis=2)
        result = jnp.einsum("bhts,bhct->bhcs", weights, content).reshape(B, -1, T)
        
        # Final projection and residual connection
        result_flax = jnp.transpose(result, (0, 2, 1))
        out_flax = self.proj(result_flax) + x_flax
        
        # Return in original format
        return jnp.transpose(out_flax, (0, 2, 1))

class BidirectionalLSTM(Module):
    def __init__(self, num_layers: int, hidden_size: int, input_size: int, *, rngs: nnx.Rngs):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.layers = []

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else 2 * hidden_size
            forward_lstm = nnx.RNN(nnx.LSTMCell(in_features=layer_input_size, hidden_features=hidden_size, rngs=rngs))
            backward_lstm = nnx.RNN(nnx.LSTMCell(in_features=layer_input_size, hidden_features=hidden_size, rngs=rngs))

            bi_layer = nnx.Bidirectional(
                forward_rnn=forward_lstm,
                backward_rnn=backward_lstm,
                merge_fn=lambda f, b: jnp.concatenate([f, b], axis=-1), # concat along feature dim
                time_major=False, # Batch dim first
            )

            self.layers.append(bi_layer)

    def __call__(self, x: Array) -> Array:
        """
        x: shape (batch_size, time_steps, input_size)
        returns: shape (batch_size, time_steps, 2 * hidden_size)
        """

        for layer in self.layers:
            x = layer(x)
        return x

class BLSTM(Module):
    def __init__(self, dim: int, layers: int = 1, skip: bool = False, *, rngs: nnx.Rngs):
        self.max_steps = 200
        self.lstm = BidirectionalLSTM(num_layers=layers, hidden_size=dim, input_size=dim, rngs=rngs)
        self.linear = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.skip = skip

    def __call__(self, x: Array) -> Array:
        """
        x: shape (batch_size, channels, time_steps)
        returns: shape (batch_size, channels, time_steps)
        """

        B, C, T = x.shape
        y = x  # For skip connection

        if self.max_steps is not None and T > self.max_steps:
            x, nframes, width, stride = self._frame_input(x, T)
            framed = True
        else:
            framed = False

        x = self.lstm(x.transpose(0, 2, 1)) # (batch_size, time_steps, 2 * channels)
        x = self.linear(x).transpose(0, 2, 1) # (batch_size, time_steps, channels)
        
        if framed:
            x = self._reconstruct_frames(x, B, C, T, nframes, width, stride)

        if self.skip:
            x += y

        return x

    def _frame_input(self, x: Array, T: int):
        width = self.max_steps
        stride = width // 2
        nframes = int(jnp.ceil(T / stride))
        tgt_length = (nframes - 1) * stride + width
        x = jnp.pad(x, ((0, 0), (0, 0), (0, tgt_length - T)))

        frames = [x[:, :, i:i+width] for i in range(0, tgt_length - width + 1, stride)]
        x = jnp.stack(frames, axis=1).reshape(-1, x.shape[1], width)
        return x, nframes, width, stride

    def _reconstruct_frames(self, x: Array, B: int, C: int, T: int, nframes: int, width: int, stride: int):
        frames = x.reshape(B, nframes, C, width)
        limit = stride // 2
        out_parts = [
            frames[:, k, :, :-limit] if k == 0 else
            frames[:, k, :, limit:] if k == nframes - 1 else
            frames[:, k, :, limit:-limit]
            for k in range(nframes)
        ]
        x = jnp.concatenate(out_parts, axis=-1)
        return x[..., :T]

class TorchConv(nnx.Conv):
    def __call__(self, x: Array) -> Array:
        """
        x: shape (batch_size, channels, time_steps)
        (nnx conv takes (batch_size, time_steps, channels))
        returns: shape (batch_size, channels, time_steps)
        """
        return super().__call__(x.transpose(0, 2, 1)).transpose(0, 2, 1)

class TorchConv2d(nnx.Conv):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        strides: int = 1,
        padding: Union[int, str] = 'SAME',
        *args, **kwargs):
        """
        Convert int arguments for kernel_size, strides, padding to tuples
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size,
            strides=(strides, strides) if isinstance(strides, int) else strides,
            padding=(padding, padding) if isinstance(padding, int) else padding,
            *args, **kwargs)

    def __call__(self, x: Array) -> Array:
        """
        x: shape (batch_size, channels, freq, time)
        (nnx conv takes (batch_size, freq, time, channels))
        returns: shape (batch_size, channels, freq, time)
        """
        return super().__call__(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

class TorchGroupNorm(nnx.GroupNorm):
    def __call__(self, x: Array) -> Array:
        """
        x: shape (batch_size, channels, *) - PyTorch convention
        (nnx groupnorm expects (batch_size, *, channels) - Flax convention)
        returns: shape (batch_size, channels, *)
        
        Works for inputs with any number of dimensions >= 3
        """
        # Move channels from dim 1 to the last dimension
        ndim = x.ndim
        perm = [0] + list(range(2, ndim)) + [1]
        x_flax = x.transpose(*perm)
        
        # Apply GroupNorm
        out_flax = super().__call__(x_flax)
        
        # Move channels back from last dimension to dim 1
        perm_back = [0, ndim-1] + list(range(1, ndim-1))
        return out_flax.transpose(*perm_back)

class DConv(Module):
    def __init__(
        self,
        channels: int,
        compress: float = 4,
        depth: int = 2,
        norm_type: str = "group_norm",
        attn: bool = False,
        heads: int = 4,
        ndecay: int = 4,
        lstm: bool = False,
        kernel_size: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)

        dilate = depth > 0
        
        # norm_fn = lambda d: Identity()
        # if norm_type == "group_norm":
        #     norm_fn = lambda d: TorchGroupNorm(num_groups=1, num_features=d, epsilon=1e-5, rngs=rngs)
        norm_fn = _get_norm_fn(norm_type, num_groups=1, epsilon=1e-5, rngs=rngs)

        hidden = int(channels / compress)

        gelu = partial(nnx.gelu, approximate=False)
        glu = partial(nnx.glu, axis=1)

        self.layers = []
        for d in range(self.depth):
            dilation = pow(2, d) if dilate else 1
            padding = dilation * (kernel_size // 2)
        
            mods = [
                TorchConv(channels, hidden, kernel_size, kernel_dilation=dilation, padding=padding, rngs=rngs),
                norm_fn(hidden),
                gelu,
                TorchConv(hidden, 2 * channels, 1, rngs=rngs), # TODO: small numerical error after this layer, cause?
                norm_fn(2 * channels),
                glu,
                LayerScale(channels, init=1e-4),
            ]
            if attn: mods.insert(3, LocalState(hidden, heads=heads, ndecay=ndecay, rngs=rngs))
            if lstm: mods.insert(3, BLSTM(hidden, layers=2, skip=True, rngs=rngs))
            
            self.layers.append(mods)
        
    def __call__(self, x: Array) -> Array:
        """
        x: shape (batch_size, channels, time_steps)
        returns: shape (batch_size, channels, time_steps)
        """
        for seq in self.layers:
            residual = x
            for layer in seq:
                x = layer(x)
            x += residual
        return x

class HybridEncoderLayer(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 4,
        empty: bool = False,
        freq: bool = True,
        norm_type: str = "group_norm",
        context: int = 0,
        dconv_kw: Optional[Dict[str, Any]] = None,
        pad: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        if dconv_kw is None: dconv_kw = {}

        norm_fn = _get_norm_fn(norm_type, num_groups=norm_groups, epsilon=1e-5, rngs=rngs)

        pad_val = kernel_size // 4 if pad else 0

        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.pad = pad_val
        self.conv_class = TorchConv2d if freq else TorchConv

        if freq: # 2d
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad_val = [pad_val, 0]

        self.conv = self.conv_class(in_channels, out_channels, kernel_size=kernel_size, padding=pad_val, strides=stride, rngs=rngs) # Corresponds to torch Conv2d
        self.norm1 = norm_fn(out_channels)

        if empty:
            self.rewrite = Identity()
            self.norm2 = Identity()
            self.dconv = Identity()
        else:
            rewrite_kernel_size = 1 + 2 * context
            self.rewrite = self.conv_class(out_channels, 2 * out_channels, rewrite_kernel_size, padding=context, rngs=rngs)
            self.norm2 = norm_fn(2 * out_channels)
            self.dconv = DConv(out_channels, **dconv_kw, rngs=rngs)

    def __call__(self, x: Array, inject: Optional[Array] = None) -> Array:
        """
        x:          (batch_size, channels, time_steps) for time
                    (batch_size, channels, freq, time) for freq

        inject:     same shape as x. Tensor to add in last layer

        returns:    (batch_size, channels, freq / stride, time) for freq
                    (batch_size, channels, ceil(time / stride)) for time
        """

        # TODO: This prob needs to be rewritten to be jit compilable
        if not self.freq and x.ndim == 4:
            B, C, Fr, T = x.shape
            x = x.reshape(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = jnp.pad(x, ((0, 0), (0, 0), (0, self.stride - (le % self.stride))))

        y = self.conv(x)

        if self.empty:
            return y

        if inject is not None:
            if inject.ndim == 3 and y.ndim == 4:
                inject = inject[:, :, None]

            y = y + inject

        y: Array = nnx.gelu(self.norm1(y), approximate=False)
        if self.freq:
            B, C, Fr, T = y.shape
            y = y.transpose(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            y = y.reshape(B, Fr, C, T).transpose(0, 2, 1, 3)
        else:
            y = self.dconv(y)

        z = self.norm2(self.rewrite(y))
        z = nnx.glu(z, axis=1)

        return z

        
class HybridDecoderLayer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        last: bool = False,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        norm_type: str = "group_norm",
        context: int = 1,
        dconv_kw: Optional[Dict[str, Any]] = None,
        pad: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        if dconv_kw is None: dconv_kw = {}

        norm_fn = _get_norm_fn(norm_type, num_groups=norm_groups, epsilon=1e-5, rngs=rngs)

        if pad:
            if (kernel_size - stride) % 2 != 0:
                raise ValueError("Kernel size and stride do not align")
            pad = (kernel_size - stride) // 2
        else:
            pad = 0

        self.pad = pad

        self.last = last
        self.freq = freq
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.empty = empty

        self.conv_class = TorchConv2d if freq else TorchConv
        self.conv_class_tr = TransposedConv2d if freq else TransposedConv1d

        if freq: # 2d
            kernel_size = (kernel_size, 1)
            stride = (stride, 1)

        self.stride = stride
        self.kernel_size = kernel_size

        self.conv_tr = self.conv_class_tr(in_channels, out_channels, kernel_size=kernel_size, strides=stride, rngs=rngs)
        self.norm2 = norm_fn(out_channels)

        if empty:
            self.rewrite = Identity()
            self.norm1 = Identity()
        else:
            self.rewrite = self.conv_class(in_channels, 2 * in_channels, 1 + 2 * context, padding=context, rngs=rngs)
            self.norm1 = norm_fn(2 * in_channels)


    def __call__(self, x: Array, skip: Array = None, length: int = 0) -> Array:
        r"""Forward pass for decoding layer.

        Size depends on whether frequency or time

        Args:
            x (torch.Tensor): tensor input of shape `(B, C, F, T)` for frequency and shape
                `(B, C, T)` for time
            skip (torch.Tensor, optional): on first layer, separate frequency and time branches using param
                (default: ``None``)
            length (int): Size of tensor for output

        Returns:
            (Tensor, Tensor):
                Tensor
                    output tensor after decoder layer of shape `(B, C, F * stride, T)` for frequency domain except last
                        frequency layer shape is `(B, C, kernel_size, T)`. Shape is `(B, C, stride * T)`
                        for time domain.
                Tensor
                    contains the output just before final transposed convolution, which is used when the
                        freq. and time branch separate. Otherwise, does not matter. Shape is
                        `(B, C, F, T)` for frequency and `(B, C, T)` for time.
        """

        # x shouldnt have dim 3 here when freq right?
        # TODO: Make jit-compatible
        if self.freq and x.ndim == 3:
            B, C, T = x.shape
            x = x.reshape(B, self.in_channels, -1, T)

        if not self.empty:
            x = x + skip
            x = self.rewrite(x)
            x = self.norm1(x)
            y = nnx.glu(x, axis=1)
        else:
            y = x
            if skip is not None:
                raise ValueError("Skip must be none when empty is true.")

        z = self.conv_tr(y)
        z = self.norm2(z)
        if self.freq:
            if self.pad:
                z = z[..., self.pad : -self.pad, :]
        else:
            z = z[..., self.pad : self.pad + length]
            if z.shape[-1] != length:
                raise ValueError("Length mismatch")

        if not self.last:
            z = nnx.gelu(z, approximate=False)
        return z, y
        


def _get_norm_fn(norm_type: str, *args, **kwargs):
    if norm_type == "group_norm":
        return lambda d: TorchGroupNorm(num_features=d, *args, **kwargs)
    else: return lambda d: Identity()


class HDemucs(Module):
    def __init__(
        self,
        sources: List[str],
        audio_channels: int = 2,
        channels: int = 48,
        growth: int = 2,
        nfft: int = 4096,
        depth: int = 6,
        freq_emb: float = 0.2,
        emb_scale: int = 10,
        kernel_size: int = 8,
        time_stride: int = 2,
        stride: int = 4,
        context: int = 1,
        context_enc: int = 0,
        norm_starts: int = 4,
        norm_groups: int = 4,
        dconv_depth: int = 2,
        dconv_comp: int = 4,
        dconv_attn: int = 4,
        dconv_lstm: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        self.sources = sources
        self.audio_channels = audio_channels
        self.channels = channels
        self.nfft = nfft
        self.depth = depth

        self.kernel_size = kernel_size
        self.stride = stride
        self.context = context

        self.hop_length = self.nfft // 4
        self.freq_emb: ScaledEmbedding = None # embedding layer

        self.freq_encoder = []
        self.freq_decoder = []

        self.time_encoder = []
        self.time_decoder = []

        chin = audio_channels
        chin_z = chin * 2 # number of channels for the freq branch
        chout = channels
        chout_z = channels
        freqs = self.nfft // 2

        self.epsilon = 1e-5 # for normalization

        for index in range(self.depth):
            lstm = index >= dconv_lstm
            attn = index >= dconv_attn
            norm_type = "group_norm" if index >= norm_starts else "none"
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                if freqs != 1:
                    raise ValueError("When freq is false, freqs must be 1.")
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm_type": norm_type,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "lstm": lstm,
                    "attn": attn,
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    # "init": dconv_init,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HybridEncoderLayer(chin_z, chout_z, context=context_enc, **kw, rngs=rngs)
            if freq:
                if last_freq is True and nfft == 2048:
                    kwt["stride"] = 2
                    kwt["kernel_size"] = 4
                tenc = HybridEncoderLayer(chin, chout, context=context_enc, empty=last_freq, **kwt, rngs=rngs)
                self.time_encoder.append(tenc)

            self.freq_encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin * 2
            dec = HybridDecoderLayer(chout_z, chin_z, last=(index == 0), context=context, **kw_dec, rngs=rngs)
            if freq:
                tdec = HybridDecoderLayer(chout, chin, empty=last_freq, last=(index == 0), context=context, **kwt, rngs=rngs)
                self.time_decoder.insert(0, tdec)
            self.freq_decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(freqs, chin_z, scale=emb_scale, rngs=rngs)
                self.freq_emb_scale = freq_emb

    def __call__(self, input: Array) -> Array:
        """
        Args:
            input: (batch_size, channels, time_steps)
        Returns:
            (batch_size, num_sources, channels, time_steps)
        """
        # input goes into two branches, frequency and time
        
        # transform input for freq branch

        x = input
        length = x.shape[-1]

        z = self._spec(input)
        mag = complex_spec_to_real(z)
        x = mag

        B, C, F, T = x.shape

        # normalize input to freq branch
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        std = x.std(axis=(1, 2, 3), keepdims=True)
        x = (x - mean) / (self.epsilon + std)

        # normalize input to time branch
        xt = input
        xt_mean = xt.mean(axis=(1, 2), keepdims=True)
        xt_std = xt.std(axis=(1, 2), keepdims=True)
        xt = (xt - xt_mean) / (self.epsilon + xt_std)

        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths: List[int] = []  # saved lengths to properly remove padding, freq branch.
        lengths_t: List[int] = []  # saved lengths for time branch.

        # encode both freq and time input
        for idx, encode in enumerate(self.freq_encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.time_encoder):
                lengths_t.append(xt.shape[-1])
                tenc: HybridEncoderLayer = self.time_encoder[idx]
                xt = tenc(xt)

                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x: Array = encode(x, inject)
            
            if idx == 0 and self.freq_emb is not None:
                frs = jnp.arange(x.shape[-2], dtype=jnp.int32)
                emb = self.freq_emb(frs)

                # NOTE: this returns a view, could this be problem?
                emb = jnp.broadcast_to(emb.transpose()[None, :, :, None], x.shape)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        x = jnp.zeros_like(x)
        xt = jnp.zeros_like(x)

        # decode both freq and time
        for idx, decode in enumerate(self.freq_decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            offset = self.depth - len(self.time_decoder)
            if idx >= offset:
                tdec: HybridDecoderLayer = self.time_decoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    if pre.shape[2] != 1:
                        raise ValueError("Length mismatch")
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        if len(saved) != 0:
            raise AssertionError("saved is not empty")
        if len(lengths_t) != 0:
            raise AssertionError("lengths_t is not empty")
        if len(saved_t) != 0:
            raise AssertionError("saved_t is not empty")

        S = len(self.sources)
        x = x.reshape(B, S, -1, F, T)
        x = x * std[:, None] + mean[:, None]

        zout = real_spec_to_complex(x)
        x = self._ispec(zout, length)

        xt = xt.reshape(B, S, -1, length)
        xt = xt * xt_std[:, None] + xt_mean[:, None]
        x = xt + x
        return x
            
            
    def _spec(self, x: Array) -> Array:
        """
        Pad and convert signal to spectrogram

        Args:
            x: (batch_size, channels, time_steps)
        Returns:
            spectrogram: (batch_size, channels, freqs, time_steps)
        """
        # pad input to make sure its fft is same length
        # as input for simplicity
        length = x.shape[-1]

        le = int(math.ceil(length / self.hop_length))
        pad = self.hop_length // 2 * 3

        pad_left = pad
        pad_right = pad + le * self.hop_length - x.shape[-1]

        x = self._pad(x, pad_left, pad_right)

        z = signal_to_spectrogram(x, n_fft=self.nfft, hop_length=self.hop_length)

        z = z[..., :-1, :] # NOTE: Why? Discard padded zeros to keep consistent shape?

        # TODO: Is this JIT-compilable?
        if z.shape[-1] != le + 4:
            raise ValueError("Spectrogram's last dimension must be 4 + input size divided by stride")

        z = z[..., 2 : 2 + le] # NOTE: Why?

        return z

    def _ispec(self, z: Array, length: int) -> Array:
        """
        Pad and convert spectrogram to signal

        Args:
            z: (batch_size, extra_dim, channels, freqs, time_steps)
            length: int
        Returns:
            (batch_size, extra_dim, channels, time_steps)
        """
        # Pad second to last dim with 1 on the right
        z = jnp.pad(z, ((0, 0), (0, 0), (0, 0), (0, 1), (0, 0)))
        # Pad last dim with 2 on both sides
        z = jnp.pad(z, ((0, 0), (0, 0), (0, 0), (0, 0), (2, 2)))
        pad = self.hop_length // 2 * 3
        le = self.hop_length * int(math.ceil(length / self.hop_length)) + 2 * pad

        x = spectrogram_to_signal(z, self.hop_length, length=le)
        x = x[..., pad : pad + length]
        return x


    def _pad(self, x: Array, pad_left: int, pad_right: int) -> Array:
        length = x.shape[-1]
        reflect_pad_dims = [(0, 0), (0, 0), (pad_left, pad_right)]

        max_pad = max(pad_left, pad_right)
        if length <= max_pad:
            padding_dims = [(0, 0), (0, 0), (0, max_pad - length + 1)]
            x = jnp.pad(x, padding_dims, mode="constant", constant_values=0.0)
        return jnp.pad(x, reflect_pad_dims, mode="reflect")

    