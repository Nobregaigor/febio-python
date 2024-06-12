import pyvista as pv
import numpy as np
from copy import deepcopy

from febio_python import FEBioContainer
from febio_python.feb._feb_25 import Feb25
from febio_python.feb._feb_30 import Feb30

# from febio_python.feb import Feb
from febio_python.core import (
    Nodes,
    Elements,
    FixCondition,
    RigidBodyCondition,
    States,
    StateData
)
from typing import Union, List

from febio_python.core.element_types import FebioElementTypeToVTKElementType
# from copy import deepcopy
from collections import OrderedDict

def febio_to_pyvista(data: Union[FEBioContainer, Feb25, Feb30], apply_load_curves=True) -> List[pv.UnstructuredGrid]:
    """
    Converts FEBio simulation data into a PyVista MultiBlock structure for advanced visualization and analysis. 
    This function orchestrates a series of operations to transfer all pertinent data from FEBio into a structured 
    MultiBlock format that PyVista can utilize effectively.

    Steps involved in the conversion include:
    1. Validation of Input: Converts the input data into a FEBioContainer object if not already one.
    2. Creation of MultiBlock: Initializes a MultiBlock from the FEBio container's mesh data including nodes and elements.
    3. Addition of Sets: Integrates nodal, element, and surface sets, mapping these to the corresponding elements and nodes within the PyVista grids.
    4. Integration of Data: Nodal, element, and surface data are added, ensuring all mesh-related information is transferred.
    5. Inclusion of Material Properties: Material properties defined in the FEBio model are mapped to the respective elements in PyVista.
    6. Application of Loads: Both nodal and pressure loads specified in the FEBio model are applied to the respective nodes and elements.
    7. Implementation of Boundary Conditions: Boundary conditions, both fixed and rigid body, are applied to the nodes as specified.

    Please check the individual helper functions for more details on each step.

    Parameters:
        data (Union[FEBioContainer, Feb]): Data container from an FEBio simulation.

    Returns:
        pv.MultiBlock: A fully populated PyVista MultiBlock object representing the entire FEBio model.
    """
    
    # Make sure we have a FEBioContainer object
    container: FEBioContainer = ensure_febio_container(data)
    
    # Create a multiblock object from the FEBioContainer (nodes, elements, etc.)
    grid: pv.UnstructuredGrid = create_unstructured_grid_from_febio_container(container)
    
    # Add nodal sets, element sets, and surface sets
    grid = add_nodalsets(container, grid)
    grid = add_elementsets(container, grid)
    grid = add_surfacesets(container, grid)
    
    # Add mesh data -> point data, cell data
    grid = add_nodaldata(container, grid)
    grid = add_elementdata(container, grid)
    grid = add_surface_data(container, grid)
    
    # Add materials -> cell data (parameters), field data (parameter names, type, material name)
    grid = add_material(container, grid)
    
    # Add loads -> point data (resultant nodal load), cell data (resultant pressure load)
    grid = add_nodalload(container, grid)
    grid = add_pressure_load(container, grid)
    
    # Add boundary conditions -> point data (fixed boundary conditions), cell data (rigid body boundary conditions
    grid = add_boundary_conditions(container, grid)
    
    # If states data is available, we should create a list of grids for each state
    grid_or_list_of_grids = add_states_to_grid(container, grid, apply_load_curves=apply_load_curves)
    
    if not isinstance(grid_or_list_of_grids, list):
        return [grid_or_list_of_grids]
    
    return grid_or_list_of_grids

# =============================================================================
# Validation functions
# =============================================================================

def ensure_febio_container(data: Union[FEBioContainer, Feb25, Feb30]) -> FEBioContainer:
    """Ensure the input data is a FEBioContainer object."""
    if isinstance(data, (Feb25, Feb30)):
        return FEBioContainer(feb=data)
    elif isinstance(data, FEBioContainer):
        return data
    else:
        raise ValueError("Input data must be a Feb or FEBioContainer object")

# =============================================================================
# Create mesh (multiblock) from FEBioContainer
# =============================================================================

def create_unstructured_grid_from_febio_container(container: FEBioContainer) -> pv.UnstructuredGrid:
    """
    Converts an FEBioContainer object containing mesh data into a PyVista UnstructuredGrid object.
    This function handles the conversion of node coordinates and element connectivity from the FEBio format (1-based indexing)
    to the PyVista format (0-based indexing). For each node set in the container, it creates a corresponding unstructured grid in the UnstructuredGrid.

    Parameters:
        container (FEBioContainer): The FEBio container with mesh data.

    Returns:
        pv.UnstructuredGrid: A UnstructuredGrid object containing the mesh data.
    """
    nodes: List[Nodes] = container.nodes
    volumes: List[Elements] = container.volumes
    surfaces: List[Elements] = container.surfaces
    
    elements = deepcopy(volumes) + deepcopy(surfaces) # deep copy to avoid modifying the original data
    
    # create a MultiBlock object
    # multiblock = pv.MultiBlock()
    
    # First, stack all the node coordinates; this will be the points of the mesh
    coordinates = np.vstack([node.coordinates for node in nodes])
    
    # Next, create a cells_dict.
    # This is a dictionary that maps the element type to the connectivity
    cells_dict = {}
    for elem in elements:
        el_type: str = elem.type
        connectivity: np.ndarray = elem.connectivity # FEBio uses 1-based indexing
        try:
            elem_type = FebioElementTypeToVTKElementType[el_type].value
            elem_type = pv.CellType[elem_type]
        except KeyError:
            raise ValueError(f"Element type {el_type} is not supported. "
                                "Please add it to the FebioElementTypeToVTKElementType enum.")
        
        # if the element type already exists in the cells_dict, append the connectivity
        if el_type in cells_dict:
            cells_dict[elem_type] = np.vstack([cells_dict[elem_type], connectivity])
        else:
            cells_dict[elem_type] = connectivity      
    # print(cells_dict)
    grid = pv.UnstructuredGrid(cells_dict, coordinates)
    
    return grid

# =============================================================================
# Helper functions related to mesh data
# =============================================================================

def add_nodalsets(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """
    Adds nodal sets from the FEBioContainer to the PyVista UnstructuredGrid.
    Nodal sets define specific groups of nodes. This function maps these groups to the corresponding nodes in the PyVista grids.
    Node sets are converted to binary arrays (masks) where each element represents whether a node is part of the set,
    and these masks are stored in the 'point_data' of the grid. The key for each nodal set is the name of the set.
    The reason we use binary arrays is to allow for easy visualization and analysis of nodal sets in PyVista; it also
    allows us to keep data consistent even after we "extract" parts of the mesh or apply filters.

    Parameters:
        container (FEBioContainer): The container containing nodal sets.
        grid (pv.UnstructuredGrid): The UnstructuredGrid to which the nodal sets will be added.

    Returns:
        pv.UnstructuredGrid: The updated UnstructuredGrid with nodal sets added.
    """
    for node_set in container.nodesets:
        name = node_set.name
        ids = node_set.ids

        mask = np.zeros(grid.n_points, dtype=bool)
        mask[ids] = True
        grid.point_data[name] = mask
        # print(f"Added nodal set {name} to the grid.")
    return grid

def add_elementsets(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """
    Adds element sets from the FEBioContainer to the PyVista UnstructuredGrid.
    Element sets define specific groups of elements. This function maps these groups to the corresponding elements in the PyVista grids.
    Element sets are converted to binary arrays (masks) where each element represents whether an element is part of the set,
    and these masks are stored in the 'cell_data' of the grid. The key for each element set is the name of the set.
    The reason we use binary arrays is to allow for easy visualization and analysis of element sets in PyVista; it also
    allows us to keep data consistent even after we "extract" parts of the mesh or apply filters.

    Parameters:
        container (FEBioContainer): The container containing element sets.
        UnstructuredGrid (pv.UnstructuredGrid): The UnstructuredGrid to which the element sets will be added.

    Returns:
        pv.UnstructuredGrid: The updated UnstructuredGrid with element sets added.
    """

    for elem_set in container.elementsets:
        name = elem_set.name
        ids = elem_set.ids
        # Add the element set to the field data
        mask = np.zeros(grid.n_cells, dtype=bool)
        mask[ids] = True
        grid.cell_data[name] = mask

    return grid

def add_surfacesets(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    if len(container.surfacesets) > 0:
        print("WARNING: Surface sets are not yet supported.")
    return grid

# Data
# -----------------------------------------------------------------------------

def add_nodaldata(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """
    Adds nodal data from the FEBioContainer to the PyVista UnstructuredGrid.
    Nodal data is stored in the 'point_data' of the grid, where each data field is associated with the corresponding nodes.
    NaNs are used to fill the gaps in the data arrays to ensure consistent dimensions across the grid.

    Parameters:
        container (FEBioContainer): The container containing nodal data.
        UnstructuredGrid (pv.UnstructuredGrid): The UnstructuredGrid where nodal data will be added.

    Returns:
        pv.UnstructuredGrid: The updated UnstructuredGrid with nodal data added.
    """
    nodesets = container.nodesets
    nodal_data = container.nodal_data
    # Add nodal data
    if len(nodesets) > 0:
        for nd in nodal_data:
            # Get the nodal data and reshape if necessary
            data = nd.data.reshape(-1, 1) if len(nd.data.shape) == 1 else nd.data
            node_set = nd.node_set
            name = nd.name

            # Find the nodeset and corresponding grid using a more efficient method
            related_nodeset = next((ns for ns in nodesets if ns.name == node_set), None)
            if related_nodeset is None:
                raise ValueError(f"Node set {node_set} not found.")

            # Create a full data array with NaNs and assign the actual data
            full_data = np.full((grid.n_points, data.shape[1]), np.nan)
            full_data[related_nodeset.ids] = data  # Adjusting for zero indexing
            grid.point_data[name] = full_data

    return grid

def add_elementdata(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """
    Adds element data from the FEBioContainer to the PyVista UnstructuredGrid.
    Element data is stored in the 'cell_data' of the appropriate grid.
    NaNs are used to fill the gaps in the data arrays to ensure consistent dimensions across the grid.

    Parameters:
        container (FEBioContainer): The container containing element data.
        UnstructuredGrid (pv.UnstructuredGrid): The UnstructuredGrid where element data will be added.

    Returns:
        pv.UnstructuredGrid: The updated UnstructuredGrid with element data added.
        
    Notes:
        This function assumes that the element data provided in the FEBioContainer is appropriately formatted and that element
        sets match the indices used in the data. If element sets are not properly aligned or if data is missing, NaNs are used
        to fill the gaps in the data arrays ensuring consistent dimensions across the grid.
    """
    element_data = container.element_data
    # Add element data
    for el_data in element_data:
        # get the element data
        data: np.ndarray = el_data.data
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        # get the element set
        elem_set = el_data.elem_set
        elem_ids = el_data.ids
        # get the name of the data
        name = el_data.name
        var = el_data.var
        # Find the proper grid
        full_data = np.full((grid.n_cells, data.shape[1]), np.nan)
        full_data[elem_ids] = data
        if name is not None:
            grid.cell_data[name] = full_data
        elif var is not None:
            grid.cell_data[var] = full_data
        else:
            grid.cell_data[f"element_data_{elem_set}"] = full_data
    return grid

def add_surface_data(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    if len(container.surface_data) > 0:
        print("WARNING: Surface data is not yet supported.")
    return grid

# =============================================================================
# Material helper functions
# =============================================================================

def add_material(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """
    Adds material properties from the FEBioContainer to the PyVista UnstructuredGrid. Material properties such as Young's modulus,
    Poisson's ratio, or any other parameters defined in FEBio are associated with specific elements based on their material IDs.

    - `Material Parameters`: These are transferred to PyVista as arrays in `cell_data` under "mat_parameters:{mat_id}",
      where each row corresponds to an element and each column to a material parameter. The order of parameters is consistent
      across `cell_data` and `field_data`.
    - `Material Type and Name`: These are stored in `field_data` under "mat_type:{mat_id}" and "mat_name:{mat_id}", respectively,
      providing a reference to the type and name of the material used.
    - `Parameters Order`: The names of the parameters (e.g., 'Young's modulus', 'Poisson's ratio') are stored in `field_data`
      under "mat_parameters:{mat_id}" to maintain an understanding of the data structure in `cell_data`.

    Parameters:
        container (FEBioContainer): The container containing material data.
        UnstructuredGrid (pv.UnstructuredGrid): The UnstructuredGrid where material properties will be added.

    Returns:
        pv.UnstructuredGrid: The updated UnstructuredGrid with material properties added.

    Example:
        If a material in FEBio with mat_id=1 has a Young's modulus of 210 GPa and a Poisson's ratio of 0.3, after
        running this function, the parameters can be accessed in PyVista as follows:
        - Access material parameters:
          grid.cell_data['mat_parameters:1']  # Array of shape [n_elements, 2]
          where the first column is Young's modulus and the second is Poisson's ratio.
        - Access material type and name:
          grid.field_data['mat_type:1']  # Returns ['Elastic']
          grid.field_data['mat_name:1']  # Returns ['GenericElasticMaterial']
        - Access the order of parameters:
          grid.field_data['mat_parameters:1']  # Returns ['Young's modulus', 'Poisson's ratio']
    """
    elements = container.elements
    materials = container.materials
    
    for mat in materials:
        mat_name = mat.name
        mat_type = mat.type
        mat_id = mat.id
        parameters = OrderedDict(mat.parameters)

        # Find the corresponding element for the material
        target_element = next((elem for elem in elements if (str(elem.mat) == str(mat_id) or (elem.name) == mat_name)), None)
        if not target_element:
            raise ValueError(f"Could not find the proper element for material {mat_name} "
                             "Currently, the material ID must match the element material ID or the element name."
                             "e.g.:\n"
                             '<material id="1" name="MESH1" type="isotropic elastic">\n'
                             '...\n'
                             '<Elements type="tri3" name="MESH1">\n'
                             'or\n'
                             '<Elements type="tri3" name="MESH1", mat="1">\n'
                             )
        
        num_params = len(parameters)
        params_names = list(parameters.keys())
        params_values = list(parameters.values())

        # Initialize parameter array with NaNs
        params_array = np.full((grid.n_cells, num_params), np.nan)
        
        # Assign values to the parameter array
        for i, value in enumerate(params_values):
            # Directly assign scalar values
            if isinstance(value, (int, float)):
                params_array[:, i] = value
            elif isinstance(value, str):
                # If the value corresponds to existing cell data, use that data
                if value in grid.cell_data.keys():
                    params_array[:, i] = grid.cell_data[value]
                else:
                    raise ValueError(f"Value {value} is not a valid cell data for material {mat_name}")
            else:
                raise ValueError(f"Unsupported material parameter format for {value} in material {mat_name}")

        # Store material properties in the grid
        grid.cell_data[f"mat_parameters:{mat_id}"] = params_array
        grid.field_data[f"mat_parameters:{mat_id}"] = np.array(params_names, dtype=object)
        grid.field_data[f"mat_type:{mat_id}"] = [mat_type]
        grid.field_data[f"mat_name:{mat_id}"] = [mat_name]

    return grid

# =============================================================================
# Load helper functions
# =============================================================================

def add_nodalload(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """
    Adds nodal force loads from the FEBioContainer to the PyVista UnstructuredGrid. This function interprets force loads applied to specific
    nodes as described in the FEBio model. It processes these loads, assigning them to the correct nodes based on the node sets
    specified in the loads, and stores a resultant vector for each node in point_data under the key "nodal_load". The resultant
    load vector for each node is calculated by summing all applicable force vectors along the x, y, and z axes.

    Parameters:
        container (FEBioContainer): The container containing nodal force load data.
        UnstructuredGrid (pv.UnstructuredGrid): The UnstructuredGrid where nodal loads will be added.

    Returns:
        pv.UnstructuredGrid: The updated UnstructuredGrid with nodal force loads aggregated and added.

    Example:
        Consider nodal loads specified in FEBio for certain nodes in various directions:
        - 100 N in the X direction for nodes in 'NodeSet1'
        - 150 N in the Y direction for nodes in 'NodeSet2'
        After processing by this function, these loads are combined where they overlap and result in a summed force vector for each node.
        The resultant force vectors can be accessed in PyVista as:
        multiblock['MeshBlockName'].point_data['nodal_load']  # Array of shape [n_points, 3]
        Each row in the array represents the total force vector for each node, with columns corresponding to forces in the X, Y, and Z directions.
    """
    nodesets = container.nodesets
    nodal_loads = container.nodal_loads

    for nodal_load in nodal_loads:
        bc = nodal_load.dof.lower()  # 'x', 'y', or 'z' axis
        node_set = nodal_load.node_set
        scale = nodal_load.scale  # scale can be numeric or a reference to another data field

        related_nodeset = next((ns for ns in nodesets if ns.name == node_set), None)
        if related_nodeset is None:
            raise ValueError(f"Node set {node_set} not found.")

        if "nodal_load" not in grid.point_data:
            grid.point_data["nodal_load"] = np.zeros((grid.n_points, 3))
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_index = axis_map[bc]
        load_indices = related_nodeset.ids  # Adjust indices for zero-based indexing

        # Handle scale being a reference to other data fields or a numeric scale
        if isinstance(scale, str) and '*' in scale:
            parts = scale.split('*')
            scale_factor = float(parts[0]) if parts[0].replace('-', '', 1).isdigit() else float(parts[1])
            data_field = parts[1] if parts[0].replace('-', '', 1).isdigit() else parts[0]

            if data_field not in grid.point_data:
                raise ValueError(f"Referenced data field '{data_field}' not found in grid point data.")

            # Extract only the relevant scale data for the specified indices
            scale_data = grid.point_data[data_field][load_indices] * scale_factor
        elif isinstance(scale, str):
            if scale not in grid.point_data:
                raise ValueError(f"Referenced data field '{scale}' not found in grid point data.")
            scale_data = grid.point_data[scale][load_indices]
        else:
            scale_data = np.full(len(load_indices), float(scale))  # Create a full array of the scale value

        # Update the nodal load data
        grid.point_data["nodal_load"][load_indices, axis_index] += scale_data

    return grid

def add_pressure_load(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    if len(container.pressure_loads) > 0:
        print("WARNING: Pressure loads are not yet supported.")
    return grid

# =============================================================================
# Boundary condition helper functions
# =============================================================================

def add_boundary_conditions(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """
    Adds boundary conditions from the FEBioContainer to the PyVista UnstructuredGrid. This function handles two main types of boundary 
    conditions: fixed conditions (FixCondition) and rigid body conditions (RigidBodyCondition):

    - `Fixed Conditions`: These apply constraints on node displacements ('x', 'y', 'z') or shell rotations ('sx', 'sy', 'sz') for 
      specific node sets. They are recorded as binary arrays in `point_data`, where each element represents whether a node is fixed 
      along a certain axis:
        - 'fix': Binary array of shape [n_points, 3], indicating fixed displacements in X, Y, and Z directions.
        - 'fix_shell': Binary array of shape [n_points, 3], indicating fixed rotations in X, Y, and Z directions.
      Both arrays consolidate all applicable constraints per node, summing constraints where multiple conditions affect the same node.

    - `Rigid Body Conditions`: These restrict the movement or rotation of all nodes associated with a specific material, denoted by:
        - 'rigid_body': Binary array of shape [n_points, 3], indicating fixed positions in X, Y, and Z directions for nodes associated with a material.
        - 'rigid_body_rot': Binary array of shape [n_points, 3], indicating fixed rotations in X, Y, and Z directions for nodes associated with a material.
      These conditions are labeled with specific material IDs, enhancing traceability and management in complex models.

    Parameters:
        container (FEBioContainer): The container containing boundary conditions.
        UnstructuredGrid (pv.UnstructuredGrid): The UnstructuredGrid where boundary conditions will be added.

    Returns:
        pv.UnstructuredGrid: The updated UnstructuredGrid with boundary conditions processed and added.

    Example:
        After processing, to access the constraints:
        - Displacement constraints for a specific mesh block:
          grid.point_data['fix']  # Outputs a binary array where 1 indicates a fixed displacement.
        - Shell rotation constraints for the same block:
          grid.point_data['fix_shell']  # Outputs a binary array where 1 indicates a fixed shell rotation.
        - For rigid body constraints related to a specific material ID:
          grid.point_data['rigid_body']  # Fixed position constraints.
          grid.point_data['rigid_body_rot']  # Fixed rotational constraints.
    """

    for bc in container.boundary_conditions:
        if isinstance(bc, FixCondition):
            node_set = bc.node_set
            if node_set not in grid.point_data:
                raise ValueError(f"Node set {node_set} not found.")
            
            # grid, indices = grid[node_set]
            indices = np.where(grid.point_data[node_set] == 1)[0] # Get the indices of the nodes in the node set
            fixed_axes = np.zeros((grid.n_points, 3))  # For 'x', 'y', 'z'
            fixed_shells = np.zeros((grid.n_points, 3))  # For 'sx', 'sy', 'sz'
            
            # Apply constraints to axes
            for axis in ['x', 'y', 'z']:
                if axis in bc.dof:
                    fixed_axes[indices, 'xyz'.index(axis)] = 1
            for axis in ['sx', 'sy', 'sz']:
                if axis in bc.dof:
                    fixed_shells[indices, 'xyz'.index(axis[-1])] = 1
            
            # Update or initialize 'fix' and 'fix_shell' arrays in grid's point_data
            if "fix" in grid.point_data:
                grid.point_data["fix"] = grid.point_data["fix"].astype(int) | fixed_axes.astype(int)
            else:
                grid.point_data["fix"] = fixed_axes
            
            if "fix_shell" in grid.point_data:
                grid.point_data["fix_shell"] = grid.point_data["fix_shell"].astype(int) | fixed_shells.astype(int)
            else:
                grid.point_data["fix_shell"] = fixed_shells
        
        elif isinstance(bc, RigidBodyCondition):
            # material = bc.material
            # if grid.material == material:
            for axis in ['x', 'y', 'z', 'Rx', 'Ry', 'Rz']:
                key = "rigid_body" if 'R' not in axis else "rigid_body_rot"
                rigid_body_axes = np.zeros((grid.n_points, 3))
                rigid_body_axes[:, 'xyz'.index(axis[-1])] = 1
                
                if key in grid.point_data:
                    grid.point_data[key] = grid.point_data[key].astype(int) | rigid_body_axes.astype(int)
                else:
                    grid.point_data[key] = rigid_body_axes

    return grid

# =============================================================================
# DOMAINS
# =============================================================================

# def break_grid_into_domains(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.MultiBlock:
        
#     # get domains:
#     domains = container.mesh_domains # List[Union[GenericDomain, ShellDomain]]
    
#     elems_by_name = {elem.name: elem for elem in container.elements}
#     nodes_by_name = {node.name: node for node in container.nodes}
#     materials_by_name = {mat.name: mat for mat in container.materials}
    
#     node_names = [node.name for node in container.nodes]
#     elem_names = [elem.name for elem in container.elements]
    
#     # selected_domains and
#     selected_domains = []
#     selected_materials = []
#     for dom in domains:
#         part = dom.name        
#         selected_materials.append(dom.mat)
#         if part in elem_names:
#             selected_domains.append(("elements", part))
#         elif part in node_names:
#             selected_domains.append(("nodes", part))
#         else:
#             print(f"Domain {part} not found in the mesh.")
    
#     # Extract the selected domains
#     mb = pv.MultiBlock()
#     for domain_type, domain_name in selected_domains:
#         mat_name = selected_materials[selected_domains.index((domain_type, domain_name))]
#         mat_id = materials_by_name[mat_name].id
#         if domain_type == "elements":
#             indices = elems_by_name[domain_name].ids
#             if len(indices) == 0:
#                 raise ValueError(f"Element domain {domain_name} not found.")
#             grid_part = grid.extract_cells(indices)
#             mb[domain_name] = grid_part
#         elif domain_type == "nodes":
#             indices = nodes_by_name[domain_name].ids
#             if len(indices) == 0:
#                 raise ValueError(f"Node domain {domain_name} not found.")
#             grid_part = grid.extract_points(indices)
        
#         # Copy correspondind material data (from field_data) to the grid_part
#         mat_type_key = f"mat_type:{mat_id}"
#         mat_name_key = f"mat_name:{mat_id}"
#         grid_part.field_data[mat_type_key] = grid.field_data[mat_type_key]
#         grid_part.field_data[mat_name_key] = grid.field_data[mat_name_key]
#         mb[domain_name] = grid_part
    
#     return mb


# =============================================================================
# STATES
# =============================================================================

def _load_curvers_to_interpolators(container: FEBioContainer) -> dict:
    from scipy import interpolate
    interpolators = dict()
    loadcurves = container.load_curves
    for lc in loadcurves:
        lc_id = lc.id
        lc_type = lc.interpolate_type
        lc_data = lc.data
        # if lc_type == "linear":
        #     this_interpolator = interpolate.interp1d(lc_data[:, 0], lc_data[:, 1], kind='linear', fill_value="extrapolate")
        # elif lc_type == "smooth":
        #     this_interpolator = interpolate.interp1d(lc_data[:, 0], lc_data[:, 1], kind='cubic', fill_value="extrapolate")
        # else:
        #     # default to linear
        this_interpolator = interpolate.interp1d(lc_data[:, 0], lc_data[:, 1], kind='linear', fill_value="extrapolate")
        interpolators[lc_id] = this_interpolator
    return interpolators

def add_states_to_grid(container: FEBioContainer, grid:pv.UnstructuredGrid, apply_load_curves=True) -> List[pv.UnstructuredGrid]:
    
    # First, check if .xplt is provided
    if container.xplt is None:
        return grid # No states to add
    
    # Otherwise, we can extract the states
    states: States = container.states
    
    node_states: List[StateData] = states.nodes
    element_states: List[StateData] = states.elements
    surface_states: List[StateData] = states.surfaces
    timesteps: np.ndarray = states.timesteps
    
    # First, create a list of grids for each state
    state_grids = [grid.copy() for _ in range(len(timesteps))]
    # Add timestep to each field_data
    for i, grid in enumerate(state_grids):
        grid.field_data["timestep"] = [timesteps[i]]
        
    # Add node states
    for node_state in node_states:
        name = node_state.name
        data = node_state.data
        for i, grid in enumerate(state_grids):
            grid.point_data[name] = data[i]
            # special case: displacement
            if name == "displacement":
                grid.points += data[i]
    
    # Add element states
    for elem_state in element_states:
        name = elem_state.name
        data = elem_state.data
        for i, grid in enumerate(state_grids):
            grid.cell_data[name] = data[i]
    
    # Add surface states
    for surf_state in surface_states:
        name = surf_state.name
        data = surf_state.data
        for i, grid in enumerate(state_grids):
            grid.cell_data[name] = data[i]
    
    # Now, we need to interpolate the load curves, based on the timesteps
    if apply_load_curves:
        interpolators = _load_curvers_to_interpolators(container)
        for lc_id, interpolator in interpolators.items():
            for i, grid in enumerate(state_grids):
                grid.field_data[f"lc_{lc_id}"] = [interpolator(timesteps[i])]
        
        # Additionally, we should also interpolate some Input data;
        # for default we interpolate: 
        # - nodal_load
        # - pressure_load
        
        # handle nodal loads
        # get nodal loads
        nodal_loads = container.nodal_loads
        for nodal_load in nodal_loads:
            # get the related data
            bc = nodal_load.dof.lower()
            axis = 'xyz'.index(bc)
            node_set = nodal_load.node_set
            node_indices = np.where(grid.point_data[node_set] == 1)[0]
            lc_id = nodal_load.load_curve
            # get the interpolator
            interpolator = interpolators[lc_id]
            for i, grid in enumerate(state_grids):
                # get the data
                time_scale = interpolator(timesteps[i])
                # apply the modification to the load in the grid
                current_data = grid.point_data["nodal_load"][node_indices, axis]
                new_data = current_data * time_scale
                grid.point_data["nodal_load"][node_indices, axis] = new_data
        
        # handle pressure loads
        # NOTE: NOT YET IMPLEMENTED

    return state_grids
    