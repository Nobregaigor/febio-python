from collections import namedtuple

# Materials
# ------------------------------
Material = namedtuple('Material', ['id', 'name', 'parameters', 'attributes'])

# Loads
# ------------------------------
NodalLoad = namedtuple('NodalLoad', ['bc', 'node_set', 'scale', 'load_curve'])
# Pressure load must be checked (not sure if it is correct)
PressureLoad = namedtuple('PressureLoad', ['surface', 'attributes', 'multiplier'])


# Boundary conditions
# ------------------------------
BoundaryCondition = namedtuple('BoundaryCondition', ['type', 'attributes']) # generic boundary condition
FixCondition = namedtuple('FixCondition', ['bc', 'node_set'])
FixedAxis = namedtuple('FixedAxis', ['bc'])
RigidBodyCondition = namedtuple('RigidBodyCondition', ['material', 'fixed_axes'])


# Mesh data
# ------------------------------
NodalData = namedtuple('NodalData', ['node_set', 'name', 'data'])
SurfaceData = namedtuple('SurfaceData', ['surf_set', 'name', 'data'])
ElementData = namedtuple('ElementData', ['elem_set', 'name', 'data'])

