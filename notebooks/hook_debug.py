import sys
sys.path.append("../")

from flax import nnx
import jax
import logging
# logging.basicConfig(level=logging.DEBUG)

with jax.disable_jit():
    class TestModule(nnx.Module):
        def __init__(self):
            self.linear = nnx.Linear(10, 10, rngs=nnx.Rngs(0))

        def __call__(self, x):
            print("Original __call__ executing")
            return self.linear(x)

    # Create the model
    model = TestModule()
    x = jax.random.normal(jax.random.PRNGKey(0), (10,))

    # First call - this should trigger compilation
    print("First call:")
    y1 = model(x)
    print("First call complete")

    # Get the original call method
    original_call = model.__call__

    # Create a new call method that wraps the original
    def hooked_call(self, *args, **kwargs):
        print("Hooked call executing")
        print(f"Input to {self.__class__.__name__}:", args, kwargs)
        output = original_call(self, *args, **kwargs)
        print(f"Output from {self.__class__.__name__}:", output)
        return output

    # Replace the call method
    model.__call__ = hooked_call

    # Try to force recompilation by clearing JAX's cache
    jax.clear_caches()

    # Second call - should use our hooked version
    print("\nSecond call:")
    y2 = model(x)
    print("Second call complete")

    # Let's also try to inspect the module's state
    print("\nModule state:")
    # graph, params = nnx.split(model, nnx.Param)