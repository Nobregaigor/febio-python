from enum import Enum, IntEnum


class FEBioElementType(Enum):
    # Linear elements
    TRIANGLE = "tri3"
    QUAD = "quad4"
    TETRA = "tet4"
    WEDGE = "penta6"
    HEXAHEDRON = "hex8"
    TRUSS = "truss"  # Assuming linear element
    # Quadratic elements
    QUADRATIC_TRIANGLE = "tri6"
    QUADRATIC_QUAD = "quad8"
    QUADRATIC_TETRA = "tet10"
    QUADRATIC_WEDGE = "penta15"
    QUADRATIC_HEXAHEDRON = "hex20"
    # Higher order elements
    BIQUADRATIC_QUAD = "quad9"
    TRIQUADRATIC_HEXAHEDRON = "hex27"
    HIGHER_ORDER_TETRA = "tet15"


class FEBioElementValue(IntEnum):
    HEXAHEDRON = 0  # HEX
    WEDGE = 1  # PENTA
    TETRA = 2  # TET
    QUAD = 3  # QUAD
    TRIANGLE = 4  # TRI
    TRUSS = 5  # Assuming linear element
    QUADRATIC_HEXAHEDRON = 6  # HEX20
    QUADRATIC_TETRA = 7  # TET10
    QUADRATIC_WEDGE = 13  # PENTA15
    TRIQUADRATIC_HEXAHEDRON = 9  # HEX27
    QUADRATIC_QUAD = 11  # QUAD8
    HIGHER_ORDER_TETRA = 8  # TET15


class NumberOfNodesPerElement(IntEnum):
    HEXAHEDRON = 8
    WEDGE = 6
    TETRA = 4
    QUAD = 4
    TRIANGLE = 3
    TRUSS = 2
    QUADRATIC_HEXAHEDRON = 20
    QUADRATIC_TETRA = 10
    QUADRATIC_WEDGE = 15
    TRIQUADRATIC_HEXAHEDRON = 27
    QUADRATIC_QUAD = 8
    HIGHER_ORDER_TETRA = 15


class NumberOfNodesPerSurfaceElement(IntEnum):
    QUAD = 4
    TRIANGLE = 3
    QUADRATIC_QUAD = 8
    QUADRATIC_TRIANGLE = 6


class FebioElementTypeToVTKElementType(Enum):
    # Linear elements
    tri3 = "TRIANGLE"
    quad4 = "QUAD"
    tet4 = "TETRA"
    penta6 = "WEDGE"
    hex8 = "HEXAHEDRON"
    # Quadratic elements
    tri6 = "QUADRATIC_TRIANGLE"
    quad8 = "QUADRATIC_QUAD"
    tet10 = "QUADRATIC_TETRA"
    penta15 = "QUADRATIC_WEDGE"
    hex20 = "QUADRATIC_HEXAHEDRON"
    # Higher order elements
    quad9 = "BIQUADRATIC_QUAD"
    hex27 = "TRIQUADRATIC_HEXAHEDRON"
    tet15 = "HIGHER_ORDER_TETRA"


class SURFACE_ELEMENT_TYPES(IntEnum):
    # Linear elements
    TRIANGLE = FEBioElementValue.TRIANGLE
    QUAD = FEBioElementValue.QUAD
    # Quadratic elements
    QUADRATIC_QUAD = FEBioElementValue.QUADRATIC_QUAD
    # Higher order elements
    BIQUADRATIC_QUAD = FEBioElementValue.QUADRATIC_QUAD
    # VTK-supported surface types (unsupported by FEBio, value will be -1)
    QUADRATIC_TRIANGLE = -1
    BIQUADRATIC_TRIANGLE = -1
    POLYGON = -1


