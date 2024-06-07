from enum import Enum

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
