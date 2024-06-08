from collections import namedtuple

# Geometry
# ------------------------------
Nodes = namedtuple('Nodes', ['name', 'coordinates', 'ids'])
Nodes.__new__.__defaults__ = (None,)  # This sets a default for the last field only
Elements = namedtuple('Elements', ['name', 'mat', 'type', 'connectivity', 'ids'])
Elements.__new__.__defaults__ = (None,)  # This sets a default for the last field only

NodeSet = namedtuple('NodeSet', ['name', 'ids'])
SurfaceSet = namedtuple('SurfaceSet', ['name', 'ids'])
ElementSet = namedtuple('ElementSet', ['name', 'ids'])

# Materials
# ------------------------------
Material = namedtuple('Material', ['id', 'type', 'parameters', 'name', 'attributes'])

# Loads
# ------------------------------
NodalLoad = namedtuple('NodalLoad', ['bc', 'node_set', 'scale', 'load_curve'])
# Pressure load must be checked (not sure if it is correct)
PressureLoad = namedtuple('PressureLoad', ['surface', 'attributes', 'multiplier'])

LoadCurve = namedtuple('LoadCurve', ['id', 'type', 'data'])


# Boundary conditions
# ------------------------------
BoundaryCondition = namedtuple('BoundaryCondition', ['type', 'attributes']) # generic boundary condition
FixCondition = namedtuple('FixCondition', ['bc', 'node_set'])
# FixedAxis = namedtuple('FixedAxis', ['bc'])
RigidBodyCondition = namedtuple('RigidBodyCondition', ['material', 'fixed_axes'])


# Mesh data
# ------------------------------
NodalData = namedtuple('NodalData', ['node_set', 'name', 'data', 'ids'])
SurfaceData = namedtuple('SurfaceData', ['surf_set', 'name', 'data', 'ids'])
ElementData = namedtuple('ElementData', ['elem_set', 'name', 'data', 'ids'])

