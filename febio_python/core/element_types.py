from enum import Enum

class FEBioElementType(Enum):
    # Linear elements
    TRIANGLE = "tri3"
    QUAD = "quad4"
    TETRA = "tet4"
    WEDGE = "penta6"
    HEXAHEDRON = "hex8"
    # Quadratic elements
    QUADRATIC_TRIANGLE = "tri6"
    QUADRATIC_QUAD = "quad8"
    QUADRATIC_TETRA = "tet10"
    QUADRATIC_WEDGE = "penta15"
    QUADRATIC_HEXAHEDRON = "hex20"
    # Higher order elements
    BIQUADRATIC_QUAD = "quad9"
    TRIQUADRATIC_HEXAHEDRON = "hex27"


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

