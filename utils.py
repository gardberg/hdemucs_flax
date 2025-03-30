import torch
from flax import nnx
import jax.numpy as jnp
from demucs import ScaledEmbedding, LayerScale, LocalState, BidirectionalLSTM, BLSTM, DConv, HybridEncoderLayer, Identity, TorchConv, TorchConv2d, HybridDecoderLayer

from torchaudio.models._hdemucs import _ScaledEmbedding, _LayerScale, _LocalState, _BLSTM, _DConv, _HEncLayer, _HDecLayer

from conv import TransposedConv1d, TransposedConv2d

import logging

logger = logging.getLogger(__name__)
logging.getLogger('jax').setLevel(logging.WARNING)

def copy_torch_params(torch_module: torch.nn.Module, nnx_module: nnx.Module):
    """
    Copies the parameters from a pytorch module and returns the corresponding nnx module.
    """
    logger.info(f"Copying {type(torch_module)} to {type(nnx_module)}")

    if isinstance(torch_module, _ScaledEmbedding):
        if not isinstance(nnx_module, ScaledEmbedding):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a ScaledEmbedding")
        if torch_module.weight.shape != nnx_module.embedding.embedding.shape:
            raise ValueError(f"Attempted to convert torch module with weight shape {torch_module.weight.shape} to nnx_module with weight shape {nnx_module.embedding.embedding.shape}")

        nnx_module.embedding.embedding = tensor_to_param(torch_module.weight)
        return nnx_module

    if isinstance(torch_module, torch.nn.Conv1d):
        if not isinstance(nnx_module, nnx.Conv):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a Conv")

        # Torch kernel shape: (out_channels, in_channels/groups, kernel_size)
        # Flax kernel shape: (kernel_size, in_channels/groups, out_channels)

        weight_p = torch_module.weight.permute(2, 1, 0)
        if weight_p.shape != nnx_module.kernel.shape:
            raise ValueError(f"Attempted to convert torch module with weight shape {weight_p.shape} to nnx_module with kernel shape {nnx_module.kernel.shape}")

        nnx_module.kernel = tensor_to_param(weight_p)
        if torch_module.bias is not None:
            if torch_module.bias.shape != nnx_module.bias.shape:
                raise ValueError(f"Attempted to convert torch module with bias shape {torch_module.bias.shape} to nnx_module with bias shape {nnx_module.bias.shape}")
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.Conv2d):
        if not isinstance(nnx_module, nnx.Conv):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a Conv")

        weight_p = torch_module.weight.permute(2, 3, 1, 0)
        if weight_p.shape != nnx_module.kernel.shape:
            raise ValueError(f"Attempted to convert torch module with weight shape {weight_p.shape} to nnx_module with kernel shape {nnx_module.kernel.shape}")

        nnx_module.kernel = tensor_to_param(weight_p)
        if torch_module.bias is not None:
            if torch_module.bias.shape != nnx_module.bias.shape:
                raise ValueError(f"Attempted to convert torch module with bias shape {torch_module.bias.shape} to nnx_module with bias shape {nnx_module.bias.shape}")
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.Identity):
        if not isinstance(nnx_module, Identity):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a Identity")
        return nnx_module
    
    if isinstance(torch_module, _LocalState):
        if not isinstance(nnx_module, LocalState):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a LocalState")

        nnx_module.content = copy_torch_params(torch_module.content, nnx_module.content)
        nnx_module.query = copy_torch_params(torch_module.query, nnx_module.query)
        nnx_module.key = copy_torch_params(torch_module.key, nnx_module.key)
        nnx_module.query_decay = copy_torch_params(torch_module.query_decay, nnx_module.query_decay)
        nnx_module.proj = copy_torch_params(torch_module.proj, nnx_module.proj)

        return nnx_module

    if isinstance(torch_module, torch.nn.Linear):
        if not isinstance(nnx_module, nnx.Linear):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a Linear")

        weight_t = torch_module.weight.transpose(0, 1)
        if weight_t.shape != nnx_module.kernel.shape:
            raise ValueError(f"Attempted to convert torch module with weight shape {weight_t.shape} to nnx_module with kernel shape {nnx_module.kernel.shape}")

        nnx_module.kernel = tensor_to_param(weight_t)
        if torch_module.bias is not None:
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.LSTM):
        if not isinstance(nnx_module, BidirectionalLSTM):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a BidirectionalLSTM")
        if not torch_module.bidirectional:
            raise ValueError("Only bidirectional LSTM is supported")

        copy_torch_lstm_to_flax(torch_module, nnx_module)

        return nnx_module

    if isinstance(torch_module, _BLSTM):
        if not isinstance(nnx_module, BLSTM):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a BLSTM")

        nnx_module.lstm = copy_torch_params(torch_module.lstm, nnx_module.lstm)
        nnx_module.linear = copy_torch_params(torch_module.linear, nnx_module.linear)

        return nnx_module

    if isinstance(torch_module, _DConv):
        if not isinstance(nnx_module, DConv):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a DConv")
        if len(torch_module.layers) != len(nnx_module.layers):
            raise ValueError(f"Attempted to convert torch module with {len(torch_module.layers)} layers to nnx_module with {len(nnx_module.layers)} layers")

        layers = []
        for torch_layer, nnx_layer in zip(torch_module.layers, nnx_module.layers):
            layers.append(copy_torch_params(torch_layer, nnx_layer))

        nnx_module.layers = layers

        return nnx_module

    if isinstance(torch_module, torch.nn.Sequential):
        if not isinstance(nnx_module, list):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a list")

        layers = []
        for torch_layer, nnx_layer in zip(torch_module, nnx_module):
            layers.append(copy_torch_params(torch_layer, nnx_layer))

        nnx_module = layers

        return nnx_module

    if isinstance(torch_module, torch.nn.GroupNorm):
        if not isinstance(nnx_module, nnx.GroupNorm):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a GroupNorm")

        if torch_module.weight.shape != nnx_module.scale.shape:
            raise ValueError(f"Attempted to convert torch module with weight shape {torch_module.weight.shape} to nnx_module with scale shape {nnx_module.scale.shape}")
        
        if torch_module.bias.shape != nnx_module.bias.shape:
            raise ValueError(f"Attempted to convert torch module with bias shape {torch_module.bias.shape} to nnx_module with bias shape {nnx_module.bias.shape}")

        nnx_module.scale = tensor_to_param(torch_module.weight)
        nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.GELU):
        # no params to copy
        return nnx_module

    if isinstance(torch_module, torch.nn.GLU):
        # no params to copy
        return nnx_module

    if isinstance(torch_module, _LayerScale):
        if not isinstance(nnx_module, LayerScale):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a LayerScale")

        if torch_module.scale.shape != nnx_module.scale.shape:
            raise ValueError(f"Attempted to convert torch module with scale shape {torch_module.scale.shape} to nnx_module with scale shape {nnx_module.scale.shape}")

        nnx_module.scale = tensor_to_param(torch_module.scale)

        return nnx_module

    if isinstance(torch_module, _HEncLayer):
        if not isinstance(nnx_module, HybridEncoderLayer):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a HybridEncoderLayer")

        nnx_module.conv = copy_torch_params(torch_module.conv, nnx_module.conv)
        nnx_module.norm1 = copy_torch_params(torch_module.norm1, nnx_module.norm1)
        nnx_module.rewrite = copy_torch_params(torch_module.rewrite, nnx_module.rewrite)
        nnx_module.norm2 = copy_torch_params(torch_module.norm2, nnx_module.norm2)
        nnx_module.dconv = copy_torch_params(torch_module.dconv, nnx_module.dconv)

        return nnx_module

    if isinstance(torch_module, torch.nn.ConvTranspose1d):
        if not isinstance(nnx_module, TransposedConv1d):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a TransposedConv1d")

        if torch_module.weight.shape != nnx_module.weight.shape:
            raise ValueError(f"Attempted to convert torch module with weight shape {torch_module.weight.shape} to nnx_module with weight shape {nnx_module.weight.shape}")

        nnx_module.weight = tensor_to_param(torch_module.weight)
        if torch_module.bias is not None:
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.ConvTranspose2d):
        if not isinstance(nnx_module, TransposedConv2d):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a TransposedConv2d")

        if torch_module.weight.shape != nnx_module.weight.shape:
            raise ValueError(f"Attempted to convert torch module with weight shape {torch_module.weight.shape} to nnx_module with weight shape {nnx_module.weight.shape}")

        nnx_module.weight = tensor_to_param(torch_module.weight)
        if torch_module.bias is not None:
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, _HDecLayer):
        if not isinstance(nnx_module, HybridDecoderLayer):
            raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(nnx_module)}, which is not a HybridDecoderLayer")

        nnx_module.conv_tr = copy_torch_params(torch_module.conv_tr, nnx_module.conv_tr)
        nnx_module.norm2 = copy_torch_params(torch_module.norm2, nnx_module.norm2)
        nnx_module.rewrite = copy_torch_params(torch_module.rewrite, nnx_module.rewrite)
        nnx_module.norm1 = copy_torch_params(torch_module.norm1, nnx_module.norm1)

        return nnx_module
        

    else:
        raise ValueError(f"Coverting {type(torch_module)} to nnx.Module not implemented")
            

def copy_torch_lstm_to_flax(torch_lstm: torch.nn.LSTM, flax_bilstm: BidirectionalLSTM):
    for layer_idx in range(torch_lstm.num_layers):
        # Get the bidirectional layer from flax module
        flax_layer: nnx.Bidirectional = flax_bilstm.layers[layer_idx]

        # Forward direction
        copy_single_direction(
            torch_lstm, flax_layer.forward_rnn.cell, layer_idx, reverse=False
        )

        # Backward direction
        copy_single_direction(
            torch_lstm, flax_layer.backward_rnn.cell, layer_idx, reverse=True
        )

def copy_single_direction(torch_lstm: torch.nn.LSTM, flax_cell: nnx.RNN, layer_idx: int, reverse=False):
    suffix = "_reverse" if reverse else ""
    
    # Extract PyTorch parameters
    weight_ih = getattr(torch_lstm, f'weight_ih_l{layer_idx}{suffix}')
    weight_hh = getattr(torch_lstm, f'weight_hh_l{layer_idx}{suffix}')
    bias_ih = getattr(torch_lstm, f'bias_ih_l{layer_idx}{suffix}')
    bias_hh = getattr(torch_lstm, f'bias_hh_l{layer_idx}{suffix}')

    # Split PyTorch weights into gates
    w_ih_chunks = weight_ih.chunk(4, dim=0)
    w_hh_chunks = weight_hh.chunk(4, dim=0)
    b_ih_chunks = bias_ih.chunk(4, dim=0)
    b_hh_chunks = bias_hh.chunk(4, dim=0)

    # Map PyTorch gates to Flax gates explicitly
    gate_names = ['i', 'f', 'g', 'o']
    for idx, gate in enumerate(gate_names):
        # Input-hidden weights (no bias in flax)
        flax_w_ih = getattr(flax_cell, f'i{gate}_' if gate == 'f' else f'i{gate}')
        flax_w_ih.kernel = tensor_to_param(w_ih_chunks[idx].transpose(0, 1))

        # Hidden-hidden weights (with bias in flax)
        flax_w_hh = getattr(flax_cell, f'h{gate}')
        flax_w_hh.kernel = tensor_to_param(w_hh_chunks[idx].transpose(0, 1))
        flax_w_hh.bias = tensor_to_param(b_ih_chunks[idx] + b_hh_chunks[idx])  # PyTorch splits biases, Flax combines them


def tensor_to_param(torch_tensor: torch.Tensor) -> nnx.Param:
    return nnx.Param(value=torch_tensor.detach().numpy())


def print_shapes_hook(next_fun, args, kwargs, context):
    """
    Interceptor that prints input and output shapes and norms.
    Similar to PyTorch's forward hook.
    """
    # Get module name and class
    module_name = context.module.name if hasattr(context.module, 'name') else context.module.__class__.__name__
    method_name = context.method_name
    
    header = f"\n{'='*80}\n{module_name}.{method_name}\n{'-'*80}"
    
    # Print input information
    if args:
        print(header)
        print("INPUTS:")
        for i, arg in enumerate(args):
            if hasattr(arg, 'shape'):
                shape_str = str(tuple(arg.shape)).ljust(20)
                norm = jnp.linalg.norm(arg).item()
                norm_str = f"{norm:.6f}".rjust(12)
                print(f"  arg{i}:    {shape_str}    norm: {norm_str}")
    
    # Call the original method to get the output
    output = next_fun(*args, **kwargs)
    
    # Print output information
    if hasattr(output, 'shape'):
        if not args:  # Print header if not already printed
            print(header)
        print("\nOUTPUT:")
        shape_str = str(tuple(output.shape)).ljust(20)
        norm = jnp.linalg.norm(output).item()
        norm_str = f"{norm:.6f}".rjust(12)
        print(f"  out0:   {shape_str}    norm: {norm_str}")
        
    elif isinstance(output, tuple) and hasattr(output[0], 'shape'):
        if not args:  # Print header if not already printed
            print(header)
        print("\nOUTPUTS:")
        for i, out in enumerate(output):
            if hasattr(out, 'shape'):
                shape_str = str(tuple(out.shape)).ljust(20)
                norm = jnp.linalg.norm(out).item()
                norm_str = f"{norm:.6f}".rjust(12)
                print(f"  out{i}:    {shape_str}    norm: {norm_str}")
    
    print(f"{'='*80}\n")
    return output

