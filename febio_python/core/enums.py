from enum import Enum, IntEnum


class FEB_ROOT(Enum):
    ROOT = 'febio_spec'


class FEB_2_5_LEAD_TAGS(Enum):
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

class FEB_3_0_LEAD_TAGS(Enum):
    MODULE = "Module"
    CONTROL = "Control"
    MATERIAL = "Material"
    GLOBALS = "Globals"
    MESH = "Mesh"
    MESHDOMAINS = "MeshDomains"
    MESHDATA = "MeshData"
    BOUNDARY = "Boundary"
    LOADS = "Loads"
    DISCRETE = "Discrete"
    LOADDATA = "LoadData"
    OUTPUT = "Output"

class FEB_MAJOR_TAGS(Enum):
    NODES = "Nodes"
    NODESET = "NodeSet"
    NODEDATA = "NodeData" # Need to check this
    ELEMENTS = "Elements"
    ELEMENTSET = "ElementSet" # Need to check this
    ELEMENTDATA = "ElementData"
    SURFACE = "Surface"
    SURFACESET = "SurfaceSet" # Need to check this
    SURFACE_DATA = "SurfaceData"
    LOADCURVE = "loadcurve"

    MATERIAL = "material"
    NODALLOAD = "nodal_load"
    SURFACELOAD = "surface_load"

class ELEM_TYPES(Enum):
    TRIANGLE = "tri3"
    TETRAHEDRON = "tet4"
    HEXAHEDRON = "hex8"
    QUADRATIC_HEXAHEDRON = "hex20"
    QUADRATIC_TRIANGLE = "tri6"


class N_PTS_IN_ELEMENT(IntEnum):
    TRIANGLE = 3
    TETRAHEDRON = 4
    HEXAHEDRON = 8
    QUADRATIC_HEXAHEDRON = 20
    QUADRATIC_TRIANGLE = 6


class SURFACE_EL_TYPE(IntEnum):
    tri3 = 3
    quad4 = 4
    tri6 = 6
