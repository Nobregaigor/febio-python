
from enum import IntEnum

class DATA_TYPES(IntEnum):
    SCALAR = 0 # float
    VECTOR = 1 # vector of size 3
    MATRIX = 2 # vector of size 6 (due to element symmetry)

class DATA_LENGTHS(IntEnum):
    SCALAR = 1
    VECTOR = 3
    MATRIX = 6
    
    