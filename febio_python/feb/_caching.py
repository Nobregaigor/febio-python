from functools import wraps
from .bases import FebBaseObject
import hashlib


def feb_instance_cache(func):
    """ Cache decorator that uses a hash of the string representation of instance data to generate cache keys. """
    @wraps(func)
    def wrapper(self: 'FebBaseObject', *args, **kwargs):
        # Obtain the string representation of the object, possibly using __repr__ or a custom method
        object_repr = self.__repr__()

        # Generate a hash of this representation
        object_hash = hashlib.sha256(object_repr.encode('utf-8')).hexdigest()

        # Use the hash, function arguments, and keyword arguments to form a cache key
        cache_key = (object_hash,) + args + tuple(sorted(kwargs.items()))

        if cache_key not in wrapper.cache:
            wrapper.cache[cache_key] = func(self, *args, **kwargs)
        return wrapper.cache[cache_key]
    wrapper.cache = {}
    return wrapper
