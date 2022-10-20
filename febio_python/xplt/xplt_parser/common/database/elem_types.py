from enum import IntEnum

class ELEM_TYPES(IntEnum):
    HEX = 0
    PENTA = 1
    PENTA15 = 13
    TET = 2
    QUAD = 3
    TRI = 4
    TRUSS = 5
    HEX20 = 6
    TET10 = 7
    TET15 = 8
    HEX27 = 9
    
    
    @classmethod
    def in_values(cls, key):
        return key in cls._value2member_map_

    