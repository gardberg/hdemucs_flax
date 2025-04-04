import jax.numpy as jnp
from module import Module, intercept_methods

from utils import print_shapes_hook

import logging
logger = logging.getLogger(__name__)

logging.getLogger('jax').setLevel(logging.WARNING)

def test_basic_interceptor():
    hook_called = False
    
    def test_hook(next_fun, args, kwargs, context):
        logger.info("test_hook called")
        nonlocal hook_called
        hook_called = True
        logger.info("Interceptor called!")
        return next_fun(*args, **kwargs)
    
    logger.info("Creating Foo")
    class Foo(Module):
        def __call__(self, x):
            logger.info("Foo __call__ called")
            return x * 2
    
    foo = Foo()
    
    x = jnp.ones((2, 3))
    
    with intercept_methods(test_hook):
        result_with_hook = foo(x)
    
    assert hook_called, "Hook was not called during forward pass"

def test_shape_printing_hook():
    class MultiLayerNet(Module):
        def __init__(self):
            super().__init__()
            self.layer1 = LinearLayer()
            self.layer2 = LinearLayer()
        
        def __call__(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    class LinearLayer(Module):
        def __call__(self, x):
            # Simple operation that changes shape
            return jnp.ones((x.shape[0], 5))
    
    model = MultiLayerNet()
    x = jnp.ones((2, 10))
    
    with intercept_methods(print_shapes_hook):
        output = model(x)
        
def test_multiple_interceptors():
    calls = []
    
    def interceptor1(next_fun, args, kwargs, context):
        calls.append(1)
        logger.info("Interceptor 1 called")
        return next_fun(*args, **kwargs)
    
    def interceptor2(next_fun, args, kwargs, context):
        calls.append(2)
        logger.info("Interceptor 2 called")
        return next_fun(*args, **kwargs)
    
    class Foo(Module):
        def __call__(self, x):
            return x
    
    foo = Foo()
    x = jnp.ones((2, 2))
    
    with intercept_methods(interceptor1):
        with intercept_methods(interceptor2):
            result = foo(x)
    
    # Verify both interceptors were called in the correct order
    assert calls == [1, 2], f"Interceptors called in wrong order: {calls}"


def test_intercept_submodules():
    class Submodule(Module):
        def __call__(self, x):
            logger.info("Submodule __call__ called")
            return x
            
    class ParentModule(Module):
        def __init__(self):
            super().__init__()
            self.submodule = Submodule()
            
        def __call__(self, x):
            logger.info("ParentModule __call__ called")
            return self.submodule(x)

    intercepted_modules = []
    def interceptor(next_fun, args, kwargs, context):
        logger.info(f"Interceptor called for {context.module.__class__.__name__}")
        # Record which module was intercepted
        module_name = context.module.__class__.__name__
        intercepted_modules.append(module_name)
        return next_fun(*args, **kwargs)
            
    parent = ParentModule()
    x = jnp.ones((2, 2))
    
    with intercept_methods(interceptor):
        parent(x)
    
    # Verify both parent and submodule were intercepted
    assert len(intercepted_modules) == 2, f"Expected 2 intercepted calls, got {len(intercepted_modules)}"
    assert "ParentModule" in intercepted_modules, "ParentModule was not intercepted"
    assert "Submodule" in intercepted_modules, "Submodule was not intercepted"
    # Verify parent was intercepted before submodule
    assert intercepted_modules[0] == "ParentModule", "ParentModule should be intercepted first"
    assert intercepted_modules[1] == "Submodule", "Submodule should be intercepted second"