import torch
import jax.numpy as jnp
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

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