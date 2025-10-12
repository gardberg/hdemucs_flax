from typing import Tuple, Sequence, Union, Optional
from jax import Array
from flax import nnx
from flax.nnx.nn import initializers
import jax.numpy as jnp

import jax
from module import Module
import logging
logger = logging.getLogger(__name__)

"""
TransposedConv layers are only used in the time and frequency decoders.
The inputs to the TransposedConv1d init are (in_channels, out_channels, kernel_size, stride), rest default
for Conv2d its the same only that kernel_size and stride are tuples
"""

class TransposedConv1d(Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    strides: int,
    groups: int = 1,
    bias: bool = True,
    *,
    rngs: nnx.Rngs,
  ):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.strides = strides
    self.groups = groups
    self.bias = bias
    self.dilation = 1

    self.weight = nnx.Param(
      initializers.kaiming_uniform(in_axis=0, out_axis=1, batch_axis=())(
        rngs.params(), (out_channels // groups, in_channels,  kernel_size)
      )
    )

    self.bias = nnx.Param(jnp.zeros((out_channels,))) if bias else None

  def __call__(self, x: Array) -> Array:
    """Applies transposed convolution to input tensor (flax shape convention).
    x shape: (batch_size, length, in_channels)
    returns: (batch_size, out_length, out_channels)

    out_length is here determined by the shape of the convolution, see: 
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    """

    result = jax.lax.conv_transpose(
      x,
      self.weight.value.astype(x.dtype),
      strides=(self.strides,),
      padding='VALID',
      rhs_dilation=(self.dilation,),
      dimension_numbers=('NLC', 'OIL', 'NLC'),
    )

    if self.bias is not None:
      result = result + self.bias.value.astype(x.dtype).reshape(1, 1, -1) # add bias to each output channel

    return result


class TransposedConv2d(Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    bias: bool = True,
    *,
    rngs: nnx.Rngs,
  ):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    self.strides = (strides, strides) if isinstance(strides, int) else strides
    self.groups = groups
    self.bias = bias
    self.dilation = 1

    self.weight = nnx.Param(
        initializers.kaiming_uniform(in_axis=0, out_axis=1, batch_axis=())(
            rngs.params(), (out_channels // groups, in_channels, self.kernel_size[0], self.kernel_size[1])
        )
    )

    self.bias = nnx.Param(jnp.zeros((out_channels,))) if bias else None

  def __call__(self, x: Array) -> Array:
    """
    x shape: (batch_size, height, width, in_channels)
    returns: (batch_size, out_height, out_width, out_channels)
    """

    result = jax.lax.conv_transpose(
      x,
      self.weight.value.astype(x.dtype),
      strides=self.strides,
      padding='VALID',
      rhs_dilation=(self.dilation, self.dilation),
      dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )

    if self.bias is not None:
      result = result + self.bias.value.astype(x.dtype).reshape(1, 1, 1, -1)

    return result
