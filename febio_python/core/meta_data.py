from collections import namedtuple

# Geometry
# ------------------------------
Nodes = namedtuple('Nodes', ['name', 'coordinates', 'ids'])
Nodes.__new__.__defaults__ = (None,)  # This sets a default for the last field only
Elements = namedtuple('Elements', ['name', 'mat', 'part', 'type', 'connectivity', 'ids'])
Elements.__new__.__defaults__ = (None,)  # This sets a default for the last field only

NodeSet = namedtuple('NodeSet', ['name', 'ids'])
SurfaceSet = namedtuple('SurfaceSet', ['name', 'ids'])
ElementSet = namedtuple('ElementSet', ['name', 'ids'])

# Materials
# ------------------------------
Material = namedtuple('Material', ['id', 'type', 'parameters', 'name', 'attributes'])
Material.__new__.__defaults__ = (None,)  # This sets a default for the last field only

# Loads
# ------------------------------
NodalLoad = namedtuple('NodalLoad', ['dof', 'node_set', 'scale', 'load_curve', 'shell_bottom', 'type', 'relative'])
NodalLoad.__new__.__defaults__ = (None, None)
# Pressure load must be checked (not sure if it is correct)
PressureLoad = namedtuple('PressureLoad', ['surface', 'attributes', 'multiplier'])

LoadCurve = namedtuple('LoadCurve', ['id', 'interpolate_type', 'data'])

# Boundary conditions
# ------------------------------
BoundaryCondition = namedtuple('BoundaryCondition', ['type', 'attributes', 'tags'])  # generic boundary condition
BoundaryCondition.__new__.__defaults__ = (None,)  # This sets a default for the last field only
FixCondition = namedtuple('FixCondition', ['dof', 'node_set', 'name'])
FixCondition.__new__.__defaults__ = (None,)  # This sets a default for the last field only
RigidBodyCondition = namedtuple('RigidBodyCondition', ['material', 'dof'])

# Mesh data
# ------------------------------
NodalData = namedtuple('NodalData', ['node_set', 'name', 'data', 'ids', 'data_type'])
NodalData.__new__.__defaults__ = (None,)
SurfaceData = namedtuple('SurfaceData', ['surf_set', 'name', 'data', 'ids'])
ElementData = namedtuple('ElementData', ['elem_set', 'data', 'ids', 'name', 'var', 'type'])
ElementData.__new__.__defaults__ = (None, None, None,)

# Mesh Domains
GenericDomain = namedtuple('GenericDomain', ['id', 'name', 'mat'])
ShellDomain = namedtuple('ShellDomain', ['id', 'name', 'mat', 'type', 'shell_normal_nodal', 'shell_thickness'])
ShellDomain.__new__.__defaults__ = ("type", 1, 0,)  # This sets a default for the last field only

# xplt mesh data
XpltMeshPart = namedtuple('MeshPart', ['id', 'name'])
XpltMesh = namedtuple("XpltMesh", ["nodes", "elements", "surfaces", "nodesets", "parts"])

# States
# ------------------------------
StatesDict = namedtuple("StatesDict", ["types", "formats", "names"])
StateVariable = namedtuple("StateVariable", ["name", "dim", "dom", "data"])
StateData = namedtuple("StateData", ["name", "dom", "data"])
States = namedtuple("States", ["nodes", "elements", "surfaces", "timesteps"])
