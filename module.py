from flax import nnx
import contextlib
import threading
from typing import Any, Iterator, Callable, TypeVar, cast
import dataclasses
import functools
import types

import logging
logger = logging.getLogger(__name__)

logging.getLogger('jax').setLevel(logging.WARNING)

# custom module to allow for interceptors
class Module(nnx.Module):
    """An extension of nnx.Module that supports method interception.
    
    This subclass intercepts the __call__ method at class initialization
    to allow for adding hooks via the intercept_methods context manager.
    """
    
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Wrap the __call__ method if it exists in this subclass
        if '__call__' in cls.__dict__:
            original_call = cls.__call__
            
            @functools.wraps(original_call)
            def wrapped_call(self, *args, **kwargs):
                if not _global_interceptor_stack or len(_global_interceptor_stack) == 0:
                    return original_call(self, *args, **kwargs)
                
                return run_interceptors(original_call, self, *args, **kwargs)
                
            cls.__call__ = wrapped_call
    
    def __call__(self, *args, **kwargs):
        # This is the base implementation that will be inherited if not overridden
        return super().__call__(*args, **kwargs)


# Adapted from flax.linen.module
@dataclasses.dataclass(frozen=True)
class InterceptorContext:
  """Read only state showing the calling context for method interceptors.

  Attributes:
    module: The Module instance whose method is being called.
    method_name: The name of the method being called on the module.
    orig_method: The original method defined on the module. Calling it will
      short circuit all other interceptors.
  """

  module: 'Module'
  method_name: str
  orig_method: Callable[..., Any]


class ThreadLocalStack(threading.local):
  """Thread-local stack."""

  def __init__(self):
    self._storage = []

  def push(self, elem: Any) -> None:
    self._storage.append(elem)

  def pop(self) -> Any:
    return self._storage.pop()

  def __iter__(self) -> Iterator[Any]:
    return iter(reversed(self._storage))

  def __len__(self) -> int:
    return len(self._storage)

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}({self._storage})'


Args = tuple[Any]
Kwargs = dict[str, Any]
NextGetter = Callable[..., Any]
Interceptor = Callable[[NextGetter, Args, Kwargs, InterceptorContext], Any]
_global_interceptor_stack = ThreadLocalStack()

# TODO: Add a flag to only intercept the root __call__
# TODO: Handle modules that are not subclasses of 'Module', e.g. that subclass nnx.Conv or nnx.GroupNorm
@contextlib.contextmanager
def intercept_methods(interceptor: Interceptor):
  """Context manager that registers an interceptor for module method '__call__'.

  This context manager will run the interceptor for all __call__ methods ran
  inside the context by any subclasses to 'Module'. This includes any submodules
  of the module, i.e. any other Module.__call__ methods that are called by the root
  Module.__call__ method.
  
  The interceptor can for example modify arguments, results, or skip calling the original method.
  
  Args:
    interceptor: A callable that takes (next_method, args, kwargs, context)
                and returns the result of the intercepted method.
  """
  _global_interceptor_stack.push(interceptor)
  try:
    yield
  finally:
    assert _global_interceptor_stack.pop() is interceptor


def run_interceptors(
  orig_method: Callable[..., Any],
  module: 'Module',
  *args,
  **kwargs,
) -> Any:
  """Runs method interceptors."""
  method_name = _get_fn_name(orig_method)
  # Create a bound method that will correctly receive 'self' as first argument
  fun = types.MethodType(orig_method, module)
  context = InterceptorContext(module, method_name, fun)

  def wrap_interceptor(interceptor, fun):
    """Wraps `fun` with `interceptor`."""

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
      return interceptor(fun, args, kwargs, context)

    return wrapped

  # Wraps interceptors around the original method. The innermost interceptor is
  # the last one added and directly wrapped around the original bound method.
  for interceptor in _global_interceptor_stack:
    fun = wrap_interceptor(interceptor, fun)
  return fun(*args, **kwargs)


def _get_fn_name(fn):
  if isinstance(fn, functools.partial):
    return _get_fn_name(fn.func)
  return getattr(fn, '__name__', 'unnamed_function')
