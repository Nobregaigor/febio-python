from enum import Enum, IntEnum


class FEB_ROOT(Enum):
    ROOT = 'febio_spec'


class FEB_LEAD_TAGS(Enum):
    MODULE = "Module"
    CONTROL = "Control"
    MATERIAL = "Material"
    GLOBALS = "Globals"
    GEOMETRY = "Geometry"
    BOUNDARY = "Boundary"
    LOADS = "Loads"
    DISCRETESET = "Discreteset"
    LOADDATA = "LoadData"
    OUTPUT = "Output"
    MESHDATA = "MeshData"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ELEM_TYPES(Enum):
    TETRA = "tet4"
    HEXA = "hex8"


class SURFACE_EL_TYPE(IntEnum):
    tri3 = 3
    quad4 = 4
