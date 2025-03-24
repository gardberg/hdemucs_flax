from flax import nnx
from jax import Array, random
import jax.numpy as jnp
import jax.nn as nn
from functools import partial, wraps
from typing import Optional, Dict, Any, Union
class ScaledEmbedding(nnx.Module):
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

    def __call__(self, x: Array) -> Array:
        """
        x: (n_emb,) indices of embeddings to look up
        return (n_emb, emb_dim)
        """
        return self.embedding(x) * self.scale


class Identity(nnx.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input: Array) -> Array:
        return input


class LayerScale(nnx.Module):
    def __init__(self, channels: int, init: float = 0):
        self.scale = nnx.Param(jnp.zeros(channels) + init)

    def __call__(self, x: Array) -> Array:
        return self.scale[:, None] * x
        
        
class LocalState(nnx.Module):
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

class BidirectionalLSTM(nnx.Module):
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

class BLSTM(nnx.Module):
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
        x: shape (batch_size, channels, *)
        (nnx groupnorm takes (batch_size, *, channels))
        returns: shape (batch_size, channels, *)
        """
        return super().__call__(x.transpose(0, 2, 1)).transpose(0, 2, 1)

class DConv(nnx.Module):
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
                # print(f"calling {type(layer)} with x shape:\t {x.shape}, norm: {jnp.linalg.norm(x)}")
                x = layer(x)
            x += residual
        return x

class HybridEncoderLayer(nnx.Module):

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
            x = x.view(B, -1, T)

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

        y: Array = nn.gelu(self.norm1(y), approximate=False)
        if self.freq:
            B, C, Fr, T = y.shape
            y = y.transpose(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            y = y.reshape(B, Fr, C, T).transpose(0, 2, 1, 3)
        else:
            y = self.dconv(y)

        z = self.norm2(self.rewrite(y))
        z = nn.glu(z, axis=1)

        return z


def _get_norm_fn(norm_type: str, *args, **kwargs):
    if norm_type == "group_norm":
        return lambda d: TorchGroupNorm(num_features=d, *args, **kwargs)
    else: return lambda d: Identity()
