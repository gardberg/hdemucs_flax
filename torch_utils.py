import torch
import jax.numpy as jnp
import logging

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