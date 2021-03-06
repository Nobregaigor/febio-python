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
    DISCRETE = "Discrete"
    LOADDATA = "LoadData"
    OUTPUT = "Output"
    MESHDATA = "MeshData"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ELEM_TYPES(Enum):
    TRIANGLE = "tri3"
    TETRAHEDRON = "tet4"
    HEXAHEDRON = "hex8"
    QUADRATIC_HEXAHEDRON = "hex20"

class N_PTS_IN_ELEMENT(IntEnum):
    TRIANGLE = 3
    TETRAHEDRON = 4
    HEXAHEDRON = 8
    QUADRATIC_HEXAHEDRON = 20

class SURFACE_EL_TYPE(IntEnum):
    tri3 = 3
    quad4 = 4

