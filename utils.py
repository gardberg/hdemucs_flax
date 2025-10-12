import jax
import jax.numpy as jnp
from collections import defaultdict
from typing import Callable, List, Union
from functools import partial
import orbax.checkpoint as ocp
from pathlib import Path
from flax import nnx
from demucs import HDemucs

from module import Module

import logging

logger = logging.getLogger(__name__)
logging.getLogger('jax').setLevel(logging.WARNING)


### Hooks ###

def get_print_hook(log_to_file: bool = False) -> Callable:
    if log_to_file:
        shape_logger = logging.getLogger('shape_logger')
        shape_logger.setLevel(logging.INFO)
        if not shape_logger.handlers:
            file_handler = logging.FileHandler('./shape_logs.log')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            shape_logger.addHandler(file_handler)
        return partial(print_shapes_hook, print_fn=shape_logger.info)
    else:
        return print_shapes_hook

def print_shapes_hook(next_fun, args, kwargs, context, print_fn: Callable = print):
    """
    Interceptor that prints input and output shapes, dtypes and norms.
    Similar to PyTorch's forward hook.
    """
    # Get module name and class
    module_name = context.module.__class__.__name__
    method_name = context.method_name
    
    module_str = f"{method_name} ({module_name})"
    print_fn(f"\n{'-'*10} {module_str} {'-'*10}")
    
    # Print input information
    if args:
        for i, arg in enumerate(args):
            if hasattr(arg, 'shape'):
                norm = float(jnp.linalg.norm(jnp.astype(arg, jnp.float32)).item())
                norm_str = f"{norm:.6f}".rjust(12)
                dtype_str = str(arg.dtype)
                print_fn(f"  arg{i}:    {str(tuple(arg.shape)):<20}    dtype: {dtype_str:<10}    norm: {norm_str}")
    
    # Call the original method to get the output
    output = next_fun(*args, **kwargs)
    
    # Print output information
    if hasattr(output, 'shape'):
        norm = float(jnp.linalg.norm(jnp.astype(output, jnp.float32)).item())
        norm_str = f"{norm:.6f}".rjust(12)
        dtype_str = str(output.dtype)
        print_fn(f"  out0:    {str(tuple(output.shape)):<20}    dtype: {dtype_str:<10}    norm: {norm_str}")
        
    elif isinstance(output, tuple):
        for i, out in enumerate(output):
            if hasattr(out, 'shape'):
                norm = float(jnp.linalg.norm(jnp.astype(out, jnp.float32)).item())
                norm_str = f"{norm:.6f}".rjust(12)
                dtype_str = str(out.dtype)
                print_fn(f"  out{i}:    {str(tuple(out.shape)):<20}    dtype: {dtype_str:<10}    norm: {norm_str}")
    
    return output


def get_record_intermediates_hook(root_module: Module, modules_to_record: List[str]):
    return partial(record_intermediates_hook, root_module=root_module, modules_to_record=modules_to_record)

def record_intermediates_hook(
    next_fun,
    args,
    kwargs,
    context,
    root_module: Module,
    modules_to_record: List[str]
    ):
    
    if context.module.__class__.__name__ not in modules_to_record:
        return next_fun(*args, **kwargs)

    if not hasattr(root_module, '_recorded_intermediates'):
        root_module._recorded_intermediates = defaultdict(list)
        root_module._call_sequence = 0
        root_module._call_counter = defaultdict(int)
    
    output = next_fun(*args, **kwargs)

    root_module._call_sequence += 1
    class_name = context.module.__class__.__name__
    name = class_name + f"[{root_module._call_counter[class_name]}]"
    root_module._recorded_intermediates[name].append({
        'sequence': root_module._call_sequence,
        'inputs': args,
        'outputs': output
    })

    root_module._call_counter[class_name] += 1

    return output


### Saving & loading ###

def save_checkpoint(model: Module, checkpoint_dir: Union[str, Path]) -> Path:
    """
    Saves a model to 'path' using Orbax CheckpointManager

    Returns the path to the checkpoint file
    """
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    checkpoint_dir = checkpoint_dir.absolute()

    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            1,
            keep_checkpoints_without_metrics=False,
            create=True,
        ),
    )

    model_state = nnx.state(model)
    
    logger.info(f"Saving checkpoint to {checkpoint_dir} with dtypes: {_get_unique_dtypes(model_state)}")

    checkpoint_manager.save(
        1, args=ocp.args.Composite(state=ocp.args.PyTreeSave(model_state))
    )
    checkpoint_manager.close()

    return checkpoint_dir

def _get_unique_dtypes(state: nnx.State) -> set:
    dtypes = set()
    for param in jax.tree.leaves(state):
        if hasattr(param, 'dtype'):
            dtypes.add(str(param.dtype))

    return dtypes

def load_checkpoint(checkpoint_dir: Union[str, Path], dtype: jnp.dtype = jnp.float32) -> HDemucs:
    """
    Loads a model from 'path' using Orbax CheckpointManager
    
    Returns the loaded model
    """
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir = checkpoint_dir.absolute()

    abstract_model = nnx.eval_shape(lambda: HDemucs(rngs=nnx.Rngs(0), dtype=dtype))
    graph, abstract_state = nnx.split(abstract_model)

    with ocp.CheckpointManager(
        checkpoint_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as read_mgr:
        restored = read_mgr.restore(
            1,
            # pass in the model_state to restore the exact same State type
            args=ocp.args.Composite(state=ocp.args.PyTreeRestore(item=abstract_state))
        )

    model = nnx.merge(graph, restored['state'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_dir} with dtypes: {_get_unique_dtypes(nnx.state(model))}")

    return model
