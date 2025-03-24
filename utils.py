import torch
from flax import nnx

from demucs import ScaledEmbedding, LayerScale, LocalState, BidirectionalLSTM, BLSTM, DConv, HybridEncoderLayer, Identity, TorchConv, TorchConv2d

from torchaudio.models._hdemucs import _ScaledEmbedding, _LayerScale, _LocalState, _BLSTM, _DConv, _HEncLayer

def torch_module_to_params(torch_module: torch.nn.Module) -> nnx.Param:
    if isinstance(torch_module, _ScaledEmbedding):
        return tensor_to_param(torch_module.weight)
    elif isinstance(torch_module, _LayerScale):
        return tensor_to_param(torch_module.scale)
    elif isinstance(torch_module, _LocalState):
        return {
            
        }
    else:
        raise ValueError(f"Unsupported module type: {type(torch_module)}")

def copy_torch_params(torch_module: torch.nn.Module, nnx_module: nnx.Module):
    """
    Copies the parameters from a pytorch module and returns the corresponding nnx module.
    """

    if isinstance(torch_module, torch.nn.Conv1d):
        assert isinstance(nnx_module, nnx.Conv), f"nnx_module must be a Conv, got {type(nnx_module)}"

        # Torch kernel shape: (out_channels, in_channels/groups, kernel_size)
        # Flax kernel shape: (kernel_size, in_channels/groups, out_channels)

        weight_p = torch_module.weight.permute(2, 1, 0)
        assert weight_p.shape == nnx_module.kernel.shape, f"weight_p shape: {weight_p.shape} must match nnx_module.kernel shape: {nnx_module.kernel.shape}"

        nnx_module.kernel = tensor_to_param(weight_p)
        if torch_module.bias is not None:
            assert torch_module.bias.shape == nnx_module.bias.shape, f"torch_module.bias shape: {torch_module.bias.shape} must match nnx_module.bias shape: {nnx_module.bias.shape}"
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.Conv2d):
        assert isinstance(nnx_module, nnx.Conv), f"nnx_module must be a Conv, got {type(nnx_module)}"

        # Torch kernel shape: (out_channels, in_channels/groups, kernel_size, kernel_size)
        # Flax kernel shape: (kernel_size, kernel_size, in_channels/groups, out_channels)

        weight_p = torch_module.weight.permute(2, 3, 1, 0)
        assert weight_p.shape == nnx_module.kernel.shape, f"weight_p shape: {weight_p.shape} must match nnx_module.kernel shape: {nnx_module.kernel.shape}"

        nnx_module.kernel = tensor_to_param(weight_p)
        if torch_module.bias is not None:
            assert torch_module.bias.shape == nnx_module.bias.shape, f"torch_module.bias shape: {torch_module.bias.shape} must match nnx_module.bias shape: {nnx_module.bias.shape}"
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.Identity):
        assert isinstance(nnx_module, Identity), f"nnx_module must be a Identity, got {type(nnx_module)}"
        return nnx_module
    
    if isinstance(torch_module, _LocalState):
        assert isinstance(nnx_module, LocalState), f"nnx_module must be a LocalState, got {type(nnx_module)}"

        nnx_module.content = copy_torch_params(torch_module.content, nnx_module.content)
        nnx_module.query = copy_torch_params(torch_module.query, nnx_module.query)
        nnx_module.key = copy_torch_params(torch_module.key, nnx_module.key)
        nnx_module.query_decay = copy_torch_params(torch_module.query_decay, nnx_module.query_decay)
        nnx_module.proj = copy_torch_params(torch_module.proj, nnx_module.proj)

        return nnx_module

    if isinstance(torch_module, torch.nn.Linear):
        assert isinstance(nnx_module, nnx.Linear), "nnx_module must be a Linear"

        weight_t = torch_module.weight.transpose(0, 1)
        assert weight_t.shape == nnx_module.kernel.shape, f"weight_t shape: {weight_t.shape} must match nnx_module.kernel shape: {nnx_module.kernel.shape}"

        nnx_module.kernel = tensor_to_param(weight_t)
        if torch_module.bias is not None:
            nnx_module.bias = tensor_to_param(torch_module.bias)

        return nnx_module

    if isinstance(torch_module, torch.nn.LSTM):
        assert isinstance(nnx_module, BidirectionalLSTM), "nnx_module must be a BidirectionalLSTM"
        if not torch_module.bidirectional:
            raise ValueError("Only bidirectional LSTM is supported")

        copy_torch_lstm_to_flax(torch_module, nnx_module)

        return nnx_module

    if isinstance(torch_module, _BLSTM):
        assert isinstance(nnx_module, BLSTM), f"nnx_module must be a BLSTM, got {type(nnx_module)}"

        nnx_module.lstm = copy_torch_params(torch_module.lstm, nnx_module.lstm)
        nnx_module.linear = copy_torch_params(torch_module.linear, nnx_module.linear)

        return nnx_module

    if isinstance(torch_module, _DConv):
        # TODO! Tänk på att vi lägger till attention eller lstm beroende på config,
        # så måste parsea detta

        assert isinstance(nnx_module, DConv), "nnx_module must be a DConv, got {type(nnx_module)}"
        assert len(torch_module.layers) == len(nnx_module.layers), "torch_module and nnx_module must have the same number of layers"

        layers = []
        for torch_layer, nnx_layer in zip(torch_module.layers, nnx_module.layers):
            layers.append(copy_torch_params(torch_layer, nnx_layer))

        nnx_module.layers = layers

        return nnx_module

    if isinstance(torch_module, torch.nn.Sequential):
        # assert isinstance(nnx_module, nnx.Sequential), f"nnx_module must be a Sequential, got {type(nnx_module)}"
        assert isinstance(nnx_module, list), f"nnx_module must be a list, got {type(nnx_module)}"

        layers = []
        for torch_layer, nnx_layer in zip(torch_module, nnx_module):
            layers.append(copy_torch_params(torch_layer, nnx_layer))

        nnx_module = layers

        return nnx_module

    if isinstance(torch_module, torch.nn.GroupNorm):
        assert isinstance(nnx_module, nnx.GroupNorm), "nnx_module must be a GroupNorm"

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
        assert isinstance(nnx_module, LayerScale), "nnx_module must be a LayerScale"

        nnx_module.scale = tensor_to_param(torch_module.scale)

        return nnx_module

    if isinstance(torch_module, _HEncLayer):
        assert isinstance(nnx_module, HybridEncoderLayer), "nnx_module must be a HybridEncoderLayer"

        nnx_module.conv = copy_torch_params(torch_module.conv, nnx_module.conv)
        nnx_module.norm1 = copy_torch_params(torch_module.norm1, nnx_module.norm1)
        nnx_module.rewrite = copy_torch_params(torch_module.rewrite, nnx_module.rewrite)
        nnx_module.norm2 = copy_torch_params(torch_module.norm2, nnx_module.norm2)
        nnx_module.dconv = copy_torch_params(torch_module.dconv, nnx_module.dconv)

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

