# from collections import namedtuple
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from typing import Dict, Union, List, Optional


# Geometry
# ------------------------------
@dataclass
class Nodes:
    name: str   # Name of the nodes (e.g. 'NodesPart1')
    coordinates: ndarray[float]     # 2-d array of floats, where each row is the coordinates of a node (e.g. [[0, 0, 0], [1, 0, 0]])
    ids: ndarray[int] = None  # If not provided, it will be inferred from the length of the coordinates list


@dataclass
class Elements:
    name: str   # Name of the element set
    type: str   # Element type (e.g. 'tri3', 'quad4', 'tet4', 'hex8')
    connectivity: ndarray[int]  # 2-d array of integers, where each row is the connectivity of an element (e.g. [[1, 2, 3], [2, 3, 4]])
    ids: ndarray[int] = None  # If not provided, it will be inferred from the length of the connectivity list
    mat: int = None  # Required for spec 2.5, optional for later versions
    part: int = None  # Optional for spec 3.0, required for earlier versions


@dataclass
class Surfaces(Elements):
    pass


@dataclass
class NodeSet:
    name: str       # Name of the node set
    ids: ndarray[int]   # Node IDs as a 1-d array of integers


@dataclass
class SurfaceSet:
    name: str       # Name of the surface set
    ids: ndarray[int]   # Surface IDs as a 1-d array of integers


@dataclass
class ElementSet:
    name: str    # Name of the element set
    ids: ndarray[int]       # Element IDs as a 1-d array of integers


@dataclass
class DiscreteSet:
    name: str  # Name of the discrete set
    src: Union[ndarray[int], str]  # Source node set or surface set
    dst: Union[ndarray[int], str]  # Destination node set or surface set
    dmat: int  # Discrete material ID


# Materials
# ------------------------------
@dataclass
class Material:
    id: int     # Material ID
    type: str   # Material type (e.g. 'rigid body', 'neo-Hookean', 'uncoupled solid mixture')
    name: str   # Material name (will be used to attach a domain or element set to the material)
    parameters: Dict[str, Union[int, float, str]]   # Material parameters (e.g. {'E': 1e6, 'v': 0.3})
    attributes: Dict[str, Union[int, float, str]] = None    # Optional TAG attributes (e.g. {'density': 1e3})
    load_curve: Dict[str, int] = None   # Load curve ID for each material parameter (e.g. {'E': 1, 'v': 2})


@dataclass
class DiscreteMaterial:
    id: int     # Material ID
    name: str   # Material name (will be used to attach a domain or element set to the material)
    type: str = "linear spring"
    parameters: Dict[str, Union[int, float, str]] = None  # Material parameters (e.g. {'E': 1e6, 'v': 0.3})
    attributes: Dict[str, Union[int, float, str]] = None    # Optional TAG attributes (e.g. {'density': 1e3})
    load_curve: Dict[str, int] = None   # Load curve ID for each material parameter (e.g. {'E': 1, 'v': 2})
    

@dataclass
class DiscreteNonlinearSpringMaterial(DiscreteMaterial):
    type: str = "nonlinear spring"
    scale: float = 1.0
    measure: str = "elongation"
    force_type: str = "point"
    points: ndarray[float] = None
    compression: float = None
    tension: float = None
    interpolate: str = "linear"  # used when force_type is 'point'
    extend: str = "constant"  # used when force_type is 'point'
    
    def __post_init__(self):
        if self.points is None:
            if self.compression is None or self.tension is None:
                raise ValueError("Either points or compression and tension must be provided")
            # Create points from compression and tension
            # NOTE: This is a simple linear interpolation centered at 0.0
            self.points = np.vstack([
                [-1, -self.compression],
                [0, 0],
                [1, self.tension]
            ])
        else:
            if self.compression is not None or self.tension is not None:
                raise ValueError("Either points or compression and tension must be provided, not both. "
                                 "Please, set 'points' to None if you want to use compression and tension, "
                                 "or set 'compression' and 'tension' to None if you want to use 'points'.")

# Loads
# ------------------------------
@dataclass
class NodalLoad:
    node_set: str   # Name of the node set to which the load is applied
    load_curve: int   # Load curve ID
    scale: Union[float, str, tuple, ndarray]    # Load scale factor
    name: str = None   # Optional name for the load
    dof: str = None    # Degree of freedom - Will be used for spec <4
    type: str = "nodal_load"  # Load type (e.g. 'nodal_force', 'nodal_traction' only for spec >=4
    shell_bottom: bool = False  # Only for spec >=4, when type=nodal_force
    relative: bool = False  # Only for spec >=4, when type=nodal_force

    def __post_init__(self):
        self.type = str(self.type).lower()
        if self.type != "nodal_load" and self.type != "nodal_force":
            raise ValueError(f"Invalid type '{self.type}' for {self.__class__.__name__}"
                             "Valid types are 'nodal_load' and 'nodal_force'")
        if self.type == "nodal_force" and not isinstance(self.scale, (tuple, ndarray)):
            raise ValueError(f"Invalid scale {self.scale} for {self.__class__.__name__}"
                             "Scale must be a tuple or ndarray when type='nodal_force'")
        if self.type == "nodal_load" and self.dof is None:
            raise ValueError("dof cannot be None when type='nodal_load'")


@dataclass
class SurfaceLoad:
    surface: str    # Name of the surface to which the load is applied
    load_curve: int   # Load curve ID
    scale: Union[float, str, tuple, ndarray]    # Load scale factor
    type: str = "pressure"
    name: str = None   # Optional name for the load
    linear: bool = False     # Linear pressure load
    symmetric_stiffness: bool = False   # Symmetric stiffness matrix
    attributes: Dict[str, Union[int, float, str]] = None    # Pressure load attributes (e.g. {'lc': 1})
    traction_vector: ndarray = None  # Traction vector for surface traction load


@dataclass
class PressureLoad(SurfaceLoad):
    type: str = "pressure"  # Default type is 'pressure' -- mainly an alias for SurfaceLoad


@dataclass
class SurfaceTractionLoad(SurfaceLoad):
    type: str = "traction"


@dataclass
class LoadCurve:
    id: int     # Load curve ID
    interpolate_type: str       # Interpolation type (e.g. 'linear', 'smooth', 'step')
    extend: str = "CONSTANT"    # Extend type (e.g. 'CONSTANT', 'EXTRAPOLATE')
    data: ndarray = None      # Load curve data as a 2-d array (e.g. [[0, 0], [1, 1]])
    x: ndarray = None   # Optional x values for the load curve data
    y: ndarray = None   # Optional y values for the load curve data
    
    def __post_init__(self):
        if self.data is None:
            if self.x is None or self.y is None:
                raise ValueError("Either 'data' or 'x' and 'y' must be provided")
            if len(self.x) != len(self.y):
                raise ValueError("Length of 'x' and 'y' must be the same"
                                 f"Length of 'x': {len(self.x)}, Length of 'y': {len(self.y)}")
            x = self.x.reshape(-1, 1) if len(self.x.shape) == 1 else self.x
            y = self.y.reshape(-1, 1) if len(self.y.shape) == 1 else self.y
            self.data = np.hstack([x, y])
        else:
            if self.x is not None or self.y is not None:
                raise ValueError("Either 'data' or 'x' and 'y' must be provided, not both."
                                 "Please, set 'data' to None if you want to use 'x' and 'y', "
                                 "or set 'x' and 'y' to None if you want to use 'data'.")


# Boundary conditions
# ------------------------------
@dataclass
class BoundaryCondition:
    dof: str        # Degree of freedom (e.g. 'x', 'y', 'z')
    type: str       # Boundary condition type (e.g. 'fix', 'prescribe', 'rigid body', 'rigid body velocity')
    node_set: str = None    # Name of the node set to which the boundary condition is applied
    surf_set: str = None    # Name of the surface set to which the boundary condition is applied
    elem_set: str = None    # Name of the element set to which the boundary condition is applied
    material: Union[int, str] = None  # Material ID or name
    name: str = None  # Optional name for the boundary condition
    attributes: Dict[str, Union[int, float, str]] = None    # Optional attributes for the boundary condition
    tags: Dict[str, Union[int, float, str]] = None      # Optional TAG attributes


@dataclass
class FixCondition(BoundaryCondition):
    type: str = "fix"       # Default type is 'fix'

    def __post_init__(self):
        if self.node_set is None:
            raise ValueError(f"node_set cannot be None for {self.__class__.__name__}")


@dataclass
class ZeroDisplacementCondition(FixCondition):
    type: str = "zero displacement"     # Default type is 'zero displacement'

    def __post_init__(self):
        if self.node_set is None:
            raise ValueError(f"node_set cannot be None for {self.__class__.__name__}")


@dataclass
class ZeroShellDisplacementCondition(FixCondition):
    type: str = "zero shell displacement"     # Default type is 'zero shell displacement'

    def __post_init__(self):
        if self.node_set is None:
            raise ValueError(f"node_set cannot be None for {self.__class__.__name__}")

@dataclass
class PrescribedDisplacementCondition(BoundaryCondition):
    type: str = "prescribed displacement"
    relative: bool = False  # Used for prescribed displacement conditions
    value: Union[float, str, tuple, ndarray] = None # Load scale factor
    load_curve: int = None  # Load curve ID
    
    def __post_init__(self):
        if self.node_set is None:
            raise ValueError(f"node_set cannot be None for {self.__class__.__name__}")
        if self.value is None:
            raise ValueError(f"value cannot be None for {self.__class__.__name__}")
        if not isinstance(self.value, np.ndarray) and self.load_curve is None:
            raise ValueError(f"load_curve cannot be None for {self.__class__.__name__} "
                             "when value is not an ndarray")

@dataclass  # NOTE: THIS IS ONLY USED FOR SPEC <4.0, switch to RigidBodyConstraint for spec >=4.0
class RigidBodyCondition(BoundaryCondition):
    material: Union[int, str]   # Material ID or name
    dof: str        # Degree of freedom (e.g. 'x', 'y', 'z')
    name: str = None    # Optional name for the boundary condition
    type: str = "rigid body"    # Default type is 'rigid body'


@dataclass
class RigidBodyConstraint:
    dof: str   # Degree of freedom (e.g. 'x', 'y', 'z')
    body: str  # Material name of the rigid body
    type: str  # Type of rigid body constraint (e.g. 'rigid_fixed')
    name: Optional[str] = None  # Name of the rigid body constraint
    
    def __post_init__(self):
        if self.name is None:
            self.name = f"{self.body}_{self.type}"


@dataclass
class RigidBodyFixedConstraint(RigidBodyConstraint):
    type: str = "rigid_fixed"


# Mesh data
# ------------------------------
@dataclass
class NodalData:
    node_set: str   # Name of the node set to which the data is applied
    name: str   # Name of the data
    data: ndarray[float]    # Data values
    ids: ndarray[int] = None    # Node IDs, refer to the nodes in the node set (optional)
    data_type: str = None   # Data type (e.g. 'scalar', 'vector', 'tensor')


@dataclass
class SurfaceData:
    surf_set: str   # Name of the surface set to which the data is applied
    name: str   # Name of the data
    data: ndarray[float]    # Data values
    ids: ndarray[int] = None    # Surface IDs, refer to the surfaces in the surface set (optional)
    data_type: str = None   # Data type (e.g. 'scalar', 'vector', 'tensor')


@dataclass
class ElementData:
    elem_set: str   # Name of the element set to which the data is applied
    name: str  # Name of the data
    data: ndarray[float]    # Data values
    ids: ndarray[int] = None    # Element IDs, refer to the elements in the element set (optional)
    data_type: str = None   # Data type (e.g. 'scalar', 'vector', 'tensor')
    var: str = None    # Data variable (e.g. 'shell thickness', 'fiber density') Used for spec <4.0
    sub_element_tags: list[str] = None  # if data is multi-dimensional, e.g. mat-axis, we need to specify the sub-element tags (e.g. ['a', 'd'])

    def __post_init__(self):
        if self.ids is None:
            self.ids = np.arange(self.data.shape[0])
        
# Mesh Domains
# ------------------------------
@dataclass
class GenericDomain:
    name: str   # Domain name (must match at least one of the Elements of the model)
    mat: Union[int, str]    # Material ID or name
    id: int = None   # Domain ID
    tag_name: str = None    # Optional tag name


@dataclass
class SolidDomain(GenericDomain):
    tag_name: str = "SolidDomain"


@dataclass
class ShellDomain(GenericDomain):
    type: str = "elastic-shell"  # used for spec >=4
    shell_normal_nodal: float = 1.0     # normal to the shell, used for spec >=4
    shell_thickness: float = 0.01    # shell thickness, used for spec >=4
    tag_name: str = "ShellDomain"


# ================================
# Xplt data
# ================================

# Mesh parts
# ------------------------------
@dataclass
class XpltMeshPart:
    id: int     # Part ID
    name: str   # Part name


@dataclass
class XpltMesh:
    nodes: List[Nodes]
    elements: List[Elements]
    surfaces: List[SurfaceSet]
    nodesets: List[NodeSet]
    parts: List[XpltMeshPart]


# States
# ------------------------------
@dataclass
class StatesDict:
    types: List[str]    # List of state types as defined in XPLT_DATA_TYPES, e.g. scalar, vector, matrix
    formats: List[str]  # List of state formats, e.g. node, element, surface
    names: List[str]    # List of state names, e.g. displacement, velocity, acceleration


@dataclass
class StateVariable:
    name: str   # Name of the state variable
    dim: int    # Dimension of the state variable  (e.g. 1 for scalar, 3 for vector, 6 for symmetric matrix)
    dom: str    # Domain to which the state variable is attached
    data: ndarray[float]    # State variable data


@dataclass
class StateData:
    name: str       # Name of the state data
    dom: str        # Domain to which the state data is attached
    data: ndarray[float]        # State data


@dataclass
class States:
    nodes: List[StateData]
    elements: List[StateData]
    surfaces: List[StateData]
    timesteps: ndarray[float]
