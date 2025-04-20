import torch
import jax.numpy as jnp
import logging
from collections import defaultdict
import flax.nnx as nnx

from module import Module
from demucs import ScaledEmbedding, LayerScale, LocalState, BidirectionalLSTM, BLSTM, DConv, HybridEncoderLayer, Identity, TorchConv, TorchConv2d, HybridDecoderLayer, HDemucs, GroupNorm
from conv import TransposedConv1d, TransposedConv2d

from torchaudio.models._hdemucs import _ScaledEmbedding, _LayerScale, _LocalState, _BLSTM, _DConv, _HEncLayer, _HDecLayer
from torchaudio.models._hdemucs import HDemucs as TorchHDemucs

logger = logging.getLogger(__name__)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)


def validate_instance(target_module: Module, reference_class: type, torch_module: torch.nn.Module):
    if not isinstance(target_module, reference_class):
        raise ValueError(f"Attempted to convert torch module type {type(torch_module)} to nnx_module type {type(target_module)}, which is not a {reference_class}")

def validate_shapes(target_shape: tuple, reference_shape: tuple):
    if target_shape != reference_shape:
        raise ValueError(f"Attempted to convert torch module with shape {target_shape} to nnx_module with shape {reference_shape}")

def copy_torch_params(torch_module: torch.nn.Module, nnx_module: nnx.Module) -> nnx.Module:
    """
    Copies the parameters from a pytorch module and returns the corresponding nnx module.
    """
    # logger.info(f"Copying {type(torch_module)} to {type(nnx_module)}")

    if isinstance(torch_module, _ScaledEmbedding):
        validate_instance(nnx_module, ScaledEmbedding, torch_module)
        validate_shapes(torch_module.weight.shape, nnx_module.embedding.embedding.shape)

        nnx_module.embedding.embedding = tensor_to_param(torch_module.embedding.weight)
        nnx_module.scale = jnp.array(torch_module.scale)
        return nnx_module

    if isinstance(torch_module, torch.nn.Conv1d):
        validate_instance(nnx_module, nnx.Conv, torch_module)

        # Torch kernel shape: (out_channels, in_channels/groups, kernel_size)
        # Flax kernel shape: (kernel_size, in_channels/groups, out_channels)

        weight_p = torch_module.weight.permute(2, 1, 0)
        validate_shapes(weight_p.shape, nnx_module.kernel.shape)

        nnx_module.kernel = tensor_to_param(weight_p)
        if torch_module.bias is not None:
            validate_shapes(torch_module.bias.shape, nnx_module.bias.shape)
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.Conv2d):
        validate_instance(nnx_module, nnx.Conv, torch_module)

        weight_p = torch_module.weight.permute(2, 3, 1, 0)
        validate_shapes(weight_p.shape, nnx_module.kernel.shape)

        nnx_module.kernel = tensor_to_param(weight_p)
        if torch_module.bias is not None:
            validate_shapes(torch_module.bias.shape, nnx_module.bias.shape)
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.Identity):
        validate_instance(nnx_module, Identity, torch_module)
        return nnx_module
    
    if isinstance(torch_module, _LocalState):
        validate_instance(nnx_module, LocalState, torch_module)

        nnx_module.content = copy_torch_params(torch_module.content, nnx_module.content)
        nnx_module.query = copy_torch_params(torch_module.query, nnx_module.query)
        nnx_module.key = copy_torch_params(torch_module.key, nnx_module.key)
        nnx_module.query_decay = copy_torch_params(torch_module.query_decay, nnx_module.query_decay)
        nnx_module.proj = copy_torch_params(torch_module.proj, nnx_module.proj)

        return nnx_module

    if isinstance(torch_module, torch.nn.Linear):
        validate_instance(nnx_module, nnx.Linear, torch_module)

        weight_t = torch_module.weight.transpose(0, 1)
        validate_shapes(weight_t.shape, nnx_module.kernel.shape)

        nnx_module.kernel = tensor_to_param(weight_t)
        if torch_module.bias is not None:
            validate_shapes(torch_module.bias.shape, nnx_module.bias.shape)
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.LSTM):
        validate_instance(nnx_module, BidirectionalLSTM, torch_module)
        if not torch_module.bidirectional:
            raise ValueError("Only bidirectional LSTM is supported")

        copy_torch_lstm_to_flax(torch_module, nnx_module)

        return nnx_module

    if isinstance(torch_module, _BLSTM):
        validate_instance(nnx_module, BLSTM, torch_module)

        nnx_module.lstm = copy_torch_params(torch_module.lstm, nnx_module.lstm)
        nnx_module.linear = copy_torch_params(torch_module.linear, nnx_module.linear)

        return nnx_module

    if isinstance(torch_module, _DConv):
        validate_instance(nnx_module, DConv, torch_module)
        if len(torch_module.layers) != len(nnx_module.layers):
            raise ValueError(f"Attempted to convert torch module with {len(torch_module.layers)} layers to nnx_module with {len(nnx_module.layers)} layers")

        layers = []
        for torch_layer, nnx_layer in zip(torch_module.layers, nnx_module.layers):
            layers.append(copy_torch_params(torch_layer, nnx_layer))

        nnx_module.layers = layers

        return nnx_module

    if isinstance(torch_module, torch.nn.Sequential):
        validate_instance(nnx_module, list, torch_module)

        layers = []
        for torch_layer, nnx_layer in zip(torch_module, nnx_module):
            layers.append(copy_torch_params(torch_layer, nnx_layer))

        nnx_module = layers

        return nnx_module

    if isinstance(torch_module, torch.nn.ModuleList):
        validate_instance(nnx_module, list, torch_module)

        layers = []
        for torch_layer, nnx_layer in zip(torch_module, nnx_module):
            layers.append(copy_torch_params(torch_layer, nnx_layer))

        nnx_module = layers

        return nnx_module

    if isinstance(torch_module, torch.nn.GroupNorm):
        validate_instance(nnx_module, GroupNorm, torch_module)

        validate_shapes(torch_module.weight.shape, nnx_module.scale.shape)
        validate_shapes(torch_module.bias.shape, nnx_module.bias.shape)

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
        validate_instance(nnx_module, LayerScale, torch_module)
        validate_shapes(torch_module.scale.shape, nnx_module.scale.shape)

        nnx_module.scale = tensor_to_param(torch_module.scale)

        return nnx_module

    if isinstance(torch_module, _HEncLayer):
        validate_instance(nnx_module, HybridEncoderLayer, torch_module)

        nnx_module.conv = copy_torch_params(torch_module.conv, nnx_module.conv)
        nnx_module.norm1 = copy_torch_params(torch_module.norm1, nnx_module.norm1)
        nnx_module.rewrite = copy_torch_params(torch_module.rewrite, nnx_module.rewrite)
        nnx_module.norm2 = copy_torch_params(torch_module.norm2, nnx_module.norm2)
        nnx_module.dconv = copy_torch_params(torch_module.dconv, nnx_module.dconv)

        return nnx_module

    if isinstance(torch_module, torch.nn.ConvTranspose1d):
        validate_instance(nnx_module, TransposedConv1d, torch_module)
        weight = torch_module.weight.permute(1, 0, 2) # flax expects (out, in, kernel)
        nnx_module.weight = jnp.flip(tensor_to_param(weight), axis=-1) # flip to match PyTorch

        validate_shapes(weight.shape, nnx_module.weight.shape)

        if torch_module.bias is not None:
            validate_shapes(torch_module.bias.shape, nnx_module.bias.shape)
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.ConvTranspose2d):
        validate_instance(nnx_module, TransposedConv2d, torch_module)
        weight = torch_module.weight.permute(1, 0, 2, 3) # flax expects (out, in, kernel_height, kernel_width)
        nnx_module.weight = jnp.flip(tensor_to_param(weight), axis=(-2, -1)) # flip to match PyTorch

        validate_shapes(weight.shape, nnx_module.weight.shape)

        if torch_module.bias is not None:
            validate_shapes(torch_module.bias.shape, nnx_module.bias.shape)
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, _HDecLayer):
        validate_instance(nnx_module, HybridDecoderLayer, torch_module)

        nnx_module.conv_tr = copy_torch_params(torch_module.conv_tr, nnx_module.conv_tr)
        nnx_module.norm2 = copy_torch_params(torch_module.norm2, nnx_module.norm2)
        nnx_module.rewrite = copy_torch_params(torch_module.rewrite, nnx_module.rewrite)
        nnx_module.norm1 = copy_torch_params(torch_module.norm1, nnx_module.norm1)

        return nnx_module

    if isinstance(torch_module, TorchHDemucs):
        validate_instance(nnx_module, HDemucs, torch_module)
        nnx_module: HDemucs

        nnx_module.freq_emb = copy_torch_params(torch_module.freq_emb, nnx_module.freq_emb)
        nnx_module.time_encoder = copy_torch_params(torch_module.time_encoder, nnx_module.time_encoder)
        nnx_module.freq_encoder = copy_torch_params(torch_module.freq_encoder, nnx_module.freq_encoder)
        nnx_module.freq_decoder = copy_torch_params(torch_module.freq_decoder, nnx_module.freq_decoder)
        nnx_module.time_decoder = copy_torch_params(torch_module.time_decoder, nnx_module.time_decoder)
        
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


# torch shape hook
def torch_add_print_hook(module: torch.nn.Module, log_to_file: bool = False):
    if log_to_file:
        shape_logger = logging.getLogger('torch_shape_logger')
        shape_logger.setLevel(logging.INFO)
        if not shape_logger.handlers:
            file_handler = logging.FileHandler('./torch_shape_logs.log')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            shape_logger.addHandler(file_handler)
        print_fn = shape_logger.info
    else:
        print_fn = print

    def make_hook_fn(name):
        def hook_fn(module, input, output):
            module_str = f"{name} ({module.__class__.__name__})"
            print_fn(f"\n{'-'*10} {module_str} {'-'*10}")
            
            # Print input information
            input = input[0] if isinstance(input, tuple) and len(input) == 1 else input
            if isinstance(input, torch.Tensor):
                norm = float(torch.norm(input.detach().float()).item())
                norm_str = f"{norm:.6f}".rjust(12)
                print_fn(f"  arg0:    {str(tuple(input.shape)):<20}    norm: {norm_str}")
            elif isinstance(input, tuple):
                for i, inp in enumerate(input):
                    if isinstance(inp, torch.Tensor):
                        norm = float(torch.norm(inp.detach()).item())
                        norm_str = f"{norm:.6f}".rjust(12)
                        print_fn(f"  arg{i}:    {str(tuple(inp.shape)):<20}    norm: {norm_str}")
            
            # Print output information
            output = output[0] if isinstance(output, tuple) and len(output) == 1 else output
            if isinstance(output, torch.Tensor):
                norm = float(torch.norm(output.detach()).item())
                norm_str = f"{norm:.6f}".rjust(12)
                print_fn(f"  out0:    {str(tuple(output.shape)):<20}    norm: {norm_str}")
            elif isinstance(output, tuple):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        norm = float(torch.norm(out.detach()).item())
                        norm_str = f"{norm:.6f}".rjust(12)
                        print_fn(f"  out{i}:    {str(tuple(out.shape)):<20}    norm: {norm_str}")
        return hook_fn

    for name, child_module in module.named_modules():
        if len(list(child_module.children())) == 0:
            child_module.register_forward_hook(make_hook_fn(name))

    # Register hook for root module
    module.register_forward_hook(make_hook_fn(module.__class__.__name__))

    
def torch_add_record_intermediates_hook(root_module: torch.nn.Module, depth: int = 0):
    if not hasattr(root_module, '_recorded_intermediates'):
        root_module._recorded_intermediates = defaultdict(list)
        root_module._call_sequence = 0

    def make_hook_fn(name):
        def hook_fn(module, input, output):
            logger.debug(f"Running hook for {name}")
            input = input[0] if isinstance(input, tuple) or isinstance(input, list) and len(input) == 1 else input
            output = output[0] if isinstance(output, tuple) or isinstance(output, list) and len(output) == 1 else output
            
            root_module._call_sequence += 1
            root_module._recorded_intermediates[name].append({
                'sequence': root_module._call_sequence,
                'inputs': input,
                'outputs': output
            })
            logger.debug(f"Current recorded intermediates: {root_module._recorded_intermediates.keys()}")
        return hook_fn

    def add_hooks_recursive(current_module: torch.nn.Module, current_depth: int, prefix: str = ''):
        logger.debug(f"\nTrying to add hooks for {current_module.__class__.__name__}, depth {current_depth}")
        if current_depth > depth:
            logger.debug(f"Skipping {current_module.__class__.__name__} at depth {current_depth}")
            return
            
        name = prefix + current_module.__class__.__name__
        
        if isinstance(current_module, torch.nn.ModuleList):
            logger.debug(f"Found ModuleList {name}, registering hooks for its children")
            for i, child in enumerate(current_module):
                child_name = f"{name}[{i}]"
                logger.debug(f"Adding hook for module list child {child_name}")
                add_hooks_recursive(child, current_depth, child_name) # consider as same depth
        else:
            logger.debug(f"Adding hook for {name}")
            current_module.register_forward_hook(make_hook_fn(name))
        
        if current_depth < depth:
            if not isinstance(current_module, torch.nn.ModuleList):
                logger.debug(f"Adding hooks for children of {name}")
                for child_name, child_module in current_module.named_children():
                    child_prefix = f"{name}." if prefix else f"{child_name}."
                    add_hooks_recursive(child_module, current_depth + 1, child_prefix)

    add_hooks_recursive(root_module, 0)