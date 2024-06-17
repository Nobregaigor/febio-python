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


class FEB_4_0_LEAD_TAGS(Enum):
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
    NODEDATA = "NodeData"  # Need to check this
    ELEMENTS = "Elements"
    ELEMENTSET = "ElementSet"  # Need to check this
    ELEMENTDATA = "ElementData"
    SURFACE = "Surface"
    SURFACESET = "SurfaceSet"  # Need to check this
    SURFACE_DATA = "SurfaceData"
    LOADCURVE = "loadcurve"

    MATERIAL = "material"
    NODALLOAD = "nodal_load"
    SURFACELOAD = "surface_load"

    SHELLDOMAIN = "ShellDomain"


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
    TRIANGLE = 3
    quad4 = 4
    QUAD = 4
    tri6 = 6
    QUADRATIC_TRIANGLE = 6
    quad8 = 8
    QUADRATIC_QUAD = 8
    

# ==================================================================================
# XPLT_TAGS
# ==================================================================================


class XPLT_TAGS(IntEnum):
    FEBIO = int("0x00464542", base=16)

    VERSION_2_5 = int("0x0005", base=16)
    VERSION_3_0 = int("0x0008", base=16)

    ROOT = int("0x01000000", base=16)
    HEADER = int("0x01010000", base=16)

    HDR_VERSION = int("0x01010001", base=16)
    HDR_NODES = int("0x01010002", base=16)
    HDR_MAX_FACET_NODES = int("0x01010003", base=16)
    HDR_COMPRESSION = int("0x01010004", base=16)

    DICTIONARY = int("0x01020000", base=16)
    DIC_ITEM = int("0x01020001", base=16)
    DIC_ITEM_TYPE = int("0x01020002", base=16)
    DIC_ITEM_FMT = int("0x01020003", base=16)
    DIC_ITEM_NAME = int("0x01020004", base=16)
    DIC_GLOBAL = int("0x01021000", base=16)
    DIC_MATERIAL = int("0x01022000", base=16)
    DIC_NODAL = int("0x01023000", base=16)
    DIC_DOMAIN = int("0x01024000", base=16)
    DIC_SURFACE = int("0x01025000", base=16)

    MATERIALS = int("0x01030000", base=16)
    MATERIAL = int("0x01030001", base=16)
    MAT_ID = int("0x01030002", base=16)
    MAT_NAME = int("0x01030003", base=16)

    GEOMETRY = int("0x01040000", base=16)  # spec 2.5
    MESH = int("0x01040000", base=16)      # spec 3.0

    # -- Node section
    NODE_SECTION = int("0x01041000", base=16)
    NODE_HEADER = int("0x01041100", base=16)
    NODE_COORDS_2_5 = int("0x01041001", base=16)   # spec 2.5
    NODE_COORDS_3_0 = int("0x01041200", base=16)   # spec 3.0
    NODE_N_NODES = int("0x01041101", base=16)
    NODE_DIM = int("0x01041102", base=16)
    NODE_NAME = int("0x01041103", base=16)

    # -- domain section
    DOMAIN_SECTION = int("0x01042000", base=16)
    DOMAIN = int("0x01042100", base=16)
    DOMAIN_HEADER = int("0x01042101", base=16)
    DOM_ELEM_TYPE = int("0x01042102", base=16)
    DOM_MAT_ID = int("0x01042103", base=16)     # spec 2.4
    DOM_PART_ID = int("0x01042103", base=16)    # spec 3.0
    DOM_N_ELEMS = int("0x01032104", base=16)
    DOM_NAME = int("0x01032105", base=16)
    DOM_ELEM_LIST = int("0x01042200", base=16)
    ELEMENT = int("0x01042201", base=16)

    # -- surface section
    SURFACE_SECTION = int("0x01043000", base=16)
    SURFACE = int("0x01043100", base=16)
    SURFACE_HEADER = int("0x01043101", base=16)
    SURFACE_ID = int("0x01043102", base=16)
    SURFACE_N_FACETS = int("0x01043103", base=16)
    SURFACE_NAME = int("0x01043104", base=16)
    MAX_FACET_NODES = int("0x01043105", base=16)
    FACET_LIST = int("0x01043200", base=16)
    FACET = int("0x01043201", base=16)

    # -- nodeset section
    NODESET_SECTION = int("0x01044000", base=16)
    NODESET = int("0x01044100", base=16)
    NODESET_HEADER = int("0x01044101", base=16)
    NODESET_ID = int("0x01044102", base=16)
    NODESET_NAME = int("0x01044103", base=16)
    NODESET_N_NODES = int("0x01044104", base=16)
    NODESET_LIST = int("0x01044200", base=16)

    # -- parts section
    PART_SECTION = int("0x01045000", base=16)
    PART = int("0x01045100", base=16)
    PART_ID = int("0x01045101", base=16)
    PART_NAME = int("0x01045102", base=16)

    # -- states section
    STATE = int("0x02000000", base=16)
    STATE_HEADER = int("0x02010000", base=16)
    STATE_HEADER_ID = int("0x02010001", base=16)
    STATE_HEADER_TIME = int("0x02010002", base=16)

    STATE_DATA = int("0x02020000", base=16)
    STATE_VARIABLE = int("0x02020001", base=16)
    STATE_VAR_ID = int("0x02020002", base=16)
    STATE_VAR_DATA = int("0x02020003", base=16)

    GLOBAL_DATA = int("0x02020100", base=16)
    MATERIAL_DATA = int("0x02020200", base=16)
    NODE_DATA = int("0x02020300", base=16)
    ELEMENT_DATA = int("0x02020400", base=16)
    SURFACE_DATA = int("0x02020500", base=16)


class XPLT_DATA_TYPES(IntEnum):
    SCALAR = 0  # float
    VECTOR = 1  # vector of size 3
    MATRIX = 2  # vector of size 6 (due to element symmetry)


class XPLT_DATA_DIMS(IntEnum):
    SCALAR = 1
    VECTOR = 3
    MATRIX = 6
