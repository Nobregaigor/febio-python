from functools import lru_cache, wraps
from .bases import FebBaseObject

def feb_instance_cache(func):
    """ Cache decorator that uses instance attributes to generate cache keys. """
    @wraps(func)
    def wrapper(self: FebBaseObject, *args, **kwargs):
        # Assuming self.tree or self.root can be converted to a hashable type for caching.
        # If these are complex types, you might need a stable identifier or use id(self.tree).
        cache_key = (id(self.tree), id(self.root)) + args + tuple(kwargs.items())
        if cache_key not in wrapper.cache:
            wrapper.cache[cache_key] = func(self, *args, **kwargs)
        return wrapper.cache[cache_key]
    wrapper.cache = {}
    return wrapper
