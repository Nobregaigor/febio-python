# from collections import namedtuple
from dataclasses import dataclass
from numpy import ndarray
from typing import Dict, Union, Tuple, List


# Geometry
# ------------------------------
@dataclass
class Nodes:
    """Represents a collection of nodes in a finite element mesh.

    Attributes:
        name (str): Name of the nodes (e.g., 'NodesPart1').
        coordinates (ndarray[float]): A 2D array of floats, where each row contains the coordinates of a node
            (e.g., [[0, 0, 0], [1, 0, 0]]).
        ids (ndarray[int], optional): A 1D array of integers representing node IDs. If not provided,
            it will be inferred from the length of the coordinates list.
    """
    name: str   # Name of the nodes (e.g. 'NodesPart1')
    coordinates: ndarray[float]     # 2-d array of floats, where each row is the coordinates of a node (e.g. [[0, 0, 0], [1, 0, 0]])
    ids: ndarray[int] = None  # If not provided, it will be inferred from the length of the coordinates list


@dataclass
class Elements:
    """Represents a collection of elements in a finite element mesh.

    Attributes:
        name (str): Name of the element set.
        type (str): Type of the elements (e.g., 'tri3', 'quad4', 'tet4', 'hex8').
        connectivity (ndarray[int]): A 2D array of integers, where each row contains the connectivity of an element
            (e.g., [[1, 2, 3], [2, 3, 4]]).
        ids (ndarray[int], optional): A 1D array of integers representing element IDs. If not provided,
            it will be inferred from the length of the connectivity list.
        mat (int, optional): Material ID for the elements. Required for spec 2.5, optional for later versions.
        part (int, optional): Part ID for the elements. Optional for spec 3.0, required for earlier versions.
    """
    name: str   # Name of the element set
    type: str   # Element type (e.g. 'tri3', 'quad4', 'tet4', 'hex8')
    connectivity: ndarray[int]  # 2-d array of integers, where each row is the connectivity of an element (e.g. [[1, 2, 3], [2, 3, 4]])
    ids: ndarray[int] = None  # If not provided, it will be inferred from the length of the connectivity list
    mat: int = None  # Required for spec 2.5, optional for later versions
    part: int = None  # Optional for spec 3.0, required for earlier versions


@dataclass
class NodeSet:
    """Represents a set of nodes in a finite element mesh.

    Attributes:
        name (str): Name of the node set.
        ids (ndarray[int]): A 1D array of integers representing node IDs.
    """
    name: str       # Name of the node set
    ids: ndarray[int]   # Node IDs as a 1-d array of integers


@dataclass
class SurfaceSet:
    """Represents a set of surfaces in a finite element mesh.

    Attributes:
        name (str): Name of the surface set.
        ids (ndarray[int]): A 1D array of integers representing surface IDs.
    """
    name: str       # Name of the surface set
    ids: ndarray[int]   # Surface IDs as a 1-d array of integers


@dataclass
class ElementSet:
    """Represents a set of elements in a finite element mesh.

    Attributes:
        name (str): Name of the element set.
        ids (ndarray[int]): A 1D array of integers representing element IDs.
    """
    name: str    # Name of the element set
    ids: ndarray[int]       # Element IDs as a 1-d array of integers


# Materials
# ------------------------------
@dataclass
class Material:
    """Represents a material in a finite element model.

    Attributes:
        id (int): Material ID.
        type (str): Type of the material (e.g., 'rigid body', 'neo-Hookean', 'uncoupled solid mixture').
        name (str): Name of the material, used to attach a domain or element set to the material.
        parameters (Dict[str, Union[int, float, str]]): Material parameters (e.g., {'E': 1e6, 'v': 0.3}).
        attributes (Dict[str, Union[int, float, str]], optional): Optional TAG attributes (e.g., {'density': 1e3}).
    """
    id: int     # Material ID
    type: str   # Material type (e.g. 'rigid body', 'neo-Hookean', 'uncoupled solid mixture')
    name: str   # Material name (will be used to attach a domain or element set to the material)
    parameters: Dict[str, Union[int, float, str]]   # Material parameters (e.g. {'E': 1e6, 'v': 0.3})
    attributes: Dict[str, Union[int, float, str]] = None    # Optional TAG attributes (e.g. {'density': 1e3})


# Loads
# ------------------------------
@dataclass
class NodalLoad:
    """Represents a nodal load in a finite element model.

    Attributes:
        node_set (str): Name of the node set to which the load is applied.
        scale (Union[float, str, tuple]): Load scale factor. For spec <4, it must be a float or a string; for spec >4,
            it must be a tuple of length 3.
        load_curve (int): Load curve ID.
        dof (str, optional): Degree of freedom, used for spec <4.
        type (str): Load type (e.g., 'nodal_force', 'nodal_traction'), only for spec >=4. Defaults to "nodal_force".
        shell_bottom (bool): Only for spec >=4. Defaults to False.
        relative (bool): Only for spec >=4. Defaults to False.
    """
    node_set: str   # Name of the node set to which the load is applied
    scale: Union[float, str, tuple]    # Load scale factor (for spec <4, it must be a float or a string; for spec >4, it must be a tuple of length 3)
    load_curve: int   # Load curve ID
    dof: str = None    # Degree of freedom - Will be used for spec <4
    type: str = "nodal_force"  # Load type (e.g. 'nodal_force', 'nodal_traction'), only for spec >=4
    shell_bottom: bool = False  # Only for spec >=4
    relative: bool = False  # Only for spec >=4


@dataclass
class PressureLoad:
    """Represents a pressure load in a finite element model.

    Attributes:
        surface (str): Name of the surface to which the load is applied.
        attributes (Dict[str, Union[int, float, str]]): Pressure load attributes (e.g., {'lc': 1}).
        multiplier (float): Load multiplier.
    """
    surface: str    # Name of the surface to which the load is applied
    attributes: Dict[str, Union[int, float, str]]    # Pressure load attributes (e.g. {'lc': 1})
    multiplier: float


@dataclass
class LoadCurve:
    """Represents a load curve in a finite element model.

    Attributes:
        id (int): Load curve ID.
        interpolate_type (str): Interpolation type (e.g., 'linear', 'smooth', 'step').
        data (ndarray[Tuple[float, float]]): Load curve data as a 2D array of tuples (e.g., [(0.0, 0.0), (1.0, 1.0)]).
    """
    id: int     # Load curve ID
    interpolate_type: str       # Interpolation type (e.g. 'linear', 'smooth', 'step')
    data: ndarray[Tuple[float, float]]      # Load curve data as a 2-d array of tuples (e.g. [(0.0, 0.0), (1.0, 1.0)])


# Boundary conditions
# ------------------------------
@dataclass
class BoundaryCondition:
    """Represents a boundary condition in a finite element model.

    Attributes:
        dof (str): Degree of freedom (e.g., 'x', 'y', 'z').
        type (str): Boundary condition type (e.g., 'fix', 'prescribe', 'rigid body', 'rigid body velocity').
        node_set (str, optional): Name of the node set to which the boundary condition is applied.
        surf_set (str, optional): Name of the surface set to which the boundary condition is applied.
        elem_set (str, optional): Name of the element set to which the boundary condition is applied.
        material (Union[int, str], optional): Material ID or name.
        name (str, optional): Optional name for the boundary condition.
        attributes (Dict[str, Union[int, float, str]], optional): Optional attributes for the boundary condition.
        tags (Dict[str, Union[int, float, str]], optional): Optional TAG attributes.
    """
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
    """Represents a fixed boundary condition in a finite element model.

    Attributes:
        type (str): Boundary condition type, defaults to 'fix'.
        see BoundaryCondition for other attributes.
    """
    type: str = "fix"       # Default type is 'fix'

    def __post_init__(self):
        if self.node_set is None:
            raise ValueError(f"node_set cannot be None for {self.__class__.__name__}")


@dataclass
class ZeroDisplacementCondition(BoundaryCondition):
    """Represents a zero displacement boundary condition in a finite element model.

    Attributes:
        type (str): Boundary condition type, defaults to 'zero displacement'.
        see BoundaryCondition for other attributes.
    """
    type: str = "zero displacement"     # Default type is 'zero displacement'

    def __post_init__(self):
        if self.node_set is None:
            raise ValueError(f"node_set cannot be None for {self.__class__.__name__}")


@dataclass
class RigidBodyCondition(BoundaryCondition):
    """Represents a rigid body boundary condition in a finite element model.

    Attributes:
        material (Union[int, str]): Material ID or name.
        dof (str): Degree of freedom (e.g., 'x', 'y', 'z').
        see BoundaryCondition for other attributes.
    """
    material: Union[int, str]   # Material ID or name
    dof: str        # Degree of freedom (e.g. 'x', 'y', 'z')


# Mesh data
# ------------------------------
@dataclass
class NodalData:
    """Represents nodal data in a finite element model.

    Attributes:
        node_set (str): Name of the node set to which the data is applied.
        name (str): Name of the data.
        data (ndarray[float]): Data values.
        ids (ndarray[int], optional): Node IDs, refer to the nodes in the node set.
        data_type (str, optional): Data type (e.g., 'scalar', 'vector', 'tensor').
    """
    node_set: str   # Name of the node set to which the data is applied
    name: str   # Name of the data
    data: ndarray[float]    # Data values
    ids: ndarray[int] = None    # Node IDs, refer to the nodes in the node set (optional)
    data_type: str = None   # Data type (e.g. 'scalar', 'vector', 'tensor')


@dataclass
class SurfaceData:
    """Represents surface data in a finite element model.

    Attributes:
        surf_set (str): Name of the surface set to which the data is applied.
        name (str): Name of the data.
        data (ndarray[float]): Data values.
        ids (ndarray[int], optional): Surface IDs, refer to the surfaces in the surface set.
        data_type (str, optional): Data type (e.g., 'scalar', 'vector', 'tensor').
    """
    surf_set: str   # Name of the surface set to which the data is applied
    name: str   # Name of the data
    data: ndarray[float]    # Data values
    ids: ndarray[int] = None    # Surface IDs, refer to the surfaces in the surface set (optional)
    data_type: str = None   # Data type (e.g. 'scalar', 'vector', 'tensor')


@dataclass
class ElementData:
    """Represents element data in a finite element model.

    Attributes:
        elem_set (str): Name of the element set to which the data is applied.
        name (str): Name of the data.
        data (ndarray[float]): Data values.
        ids (ndarray[int], optional): Element IDs, refer to the elements in the element set.
        data_type (str, optional): Data type (e.g., 'scalar', 'vector', 'tensor').
        var (str, optional): Data variable (e.g., 'shell thickness', 'fiber density'). Used for spec <4.0.
    """
    elem_set: str   # Name of the element set to which the data is applied
    name: str  # Name of the data
    data: ndarray[float]    # Data values
    ids: ndarray[int] = None    # Element IDs, refer to the elements in the element set (optional)
    data_type: str = None   # Data type (e.g. 'scalar', 'vector', 'tensor')
    var: str = None    # Data variable (e.g. 'shell thickness', 'fiber density') Used for spec <4.0


# Mesh Domains
# ------------------------------
@dataclass
class GenericDomain:
    """Represents a generic domain in a finite element model.

    Attributes:
        id (int): Domain ID.
        name (str): Domain name (must match at least one of the elements of the model).
        mat (Union[int, str]): Material ID or name.
    """
    id: int    # Domain ID
    name: str   # Domain name (must match at least one of the Elements of the model)
    mat: Union[int, str]    # Material ID or name


@dataclass
class ShellDomain(GenericDomain):
    """Represents a shell domain in a finite element model.

    Attributes:
        type (str): Domain type, defaults to 'elastic-shell'. Used for spec >=4.
        shell_normal_nodal (float): Normal to the shell, defaults to 1.0. Used for spec >=4.
        shell_thickness (float): Shell thickness, defaults to 0.0. Used for spec >=4.
    """
    type: str = "elastic-shell"  # used for spec >=4
    shell_normal_nodal: float = 1.0     # normal to the shell, used for spec >=4
    shell_thickness: float = 0.0    # shell thickness, used for spec >=4


# ================================
# Xplt data
# ================================

# Mesh parts
# ------------------------------
@dataclass
class XpltMeshPart:
    """Represents a part of a mesh in an XPLT file.

    Attributes:
        id (int): Part ID.
        name (str): Part name.
    """
    id: int     # Part ID
    name: str   # Part name


@dataclass
class XpltMesh:
    """Represents a mesh in an XPLT file.

    Attributes:
        nodes (List[Nodes]): List of Nodes objects representing the nodes in the mesh.
        elements (List[Elements]): List of Elements objects representing the elements in the mesh.
        surfaces (List[SurfaceSet]): List of SurfaceSet objects representing the surface sets in the mesh.
        nodesets (List[NodeSet]): List of NodeSet objects representing the node sets in the mesh.
        parts (List[XpltMeshPart]): List of XpltMeshPart objects representing the parts in the mesh.
    """
    nodes: List[Nodes]
    elements: List[Elements]
    surfaces: List[SurfaceSet]
    nodesets: List[NodeSet]
    parts: List[XpltMeshPart]


# States
# ------------------------------
@dataclass
class StatesDict:
    """Represents a dictionary of states in a finite element model.

    Attributes:
        types (List[str]): List of state types as defined in XPLT_DATA_TYPES (e.g., scalar, vector, matrix).
        formats (List[str]): List of state formats (e.g., node, element, surface).
        names (List[str]): List of state names (e.g., displacement, velocity, acceleration).
    """
    types: List[str]    # List of state types as defined in XPLT_DATA_TYPES, e.g. scalar, vector, matrix
    formats: List[str]  # List of state formats, e.g. node, element, surface
    names: List[str]    # List of state names, e.g. displacement, velocity, acceleration


@dataclass
class StateVariable:
    """Represents a state variable in a finite element model.

    Attributes:
        name (str): Name of the state variable.
        dim (int): Dimension of the state variable (e.g., 1 for scalar, 3 for vector, 6 for symmetric matrix).
        dom (str): Domain to which the state variable is attached.
        data (ndarray[float]): State variable data.
    """
    name: str   # Name of the state variable
    dim: int    # Dimension of the state variable  (e.g. 1 for scalar, 3 for vector, 6 for symmetric matrix)
    dom: str    # Domain to which the state variable is attached
    data: ndarray[float]    # State variable data


@dataclass
class StateData:
    """Represents state data in a finite element model.

    Attributes:
        name (str): Name of the state data.
        dom (str): Domain to which the state data is attached.
        data (ndarray[float]): State data.
    """
    name: str       # Name of the state data
    dom: str        # Domain to which the state data is attached
    data: ndarray[float]        # State data


@dataclass
class States:
    """Represents the state data at different timesteps in a finite element model.

    Attributes:
        nodes (List[StateData]): List of StateData objects representing state data for nodes.
        elements (List[StateData]): List of StateData objects representing state data for elements.
        surfaces (List[StateData]): List of StateData objects representing state data for surfaces.
        timesteps (ndarray[float]): Array of timesteps at which the state data is recorded.
    """
    nodes: List[StateData]
    elements: List[StateData]
    surfaces: List[StateData]
    timesteps: ndarray[float]


# StateVariable = namedtuple("StateVariable", ["name", "dim", "dom", "data"])
# StateData = namedtuple("StateData", ["name", "dom", "data"])
# States = namedtuple("States", ["nodes", "elements", "surfaces", "timesteps"])
