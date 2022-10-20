from enum import IntEnum

class NODES_PER_ELEM(IntEnum):
    HEX = 8
    PENTA = 6
    PENTA15 = 15
    TET = 4
    QUAD = 4
    TRI = 3
    TRUSS = 2
    HEX20 = 20
    TET10 = 10
    TET15 = 15
    HEX27 = 27

    @classmethod
    def in_values(cls, key):
        return key in cls._value2member_map_