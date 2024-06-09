import pyvista as pv
import numpy as np
from copy import deepcopy

from febio_python import FEBioContainer, Feb
# from febio_python.feb import Feb
from febio_python.core import (
    Nodes,
    Elements,
    FixCondition,
    RigidBodyCondition,
)
from typing import Union, List

from febio_python.core.element_types import FebioElementTypeToVTKElementType
# from copy import deepcopy
from collections import OrderedDict

def febio_to_pyvista(data: Union[FEBioContainer, Feb]) -> pv.MultiBlock:
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
    multiblock: pv.MultiBlock = create_multiblock_from_febio_container(container)
    
    # Add nodal sets, element sets, and surface sets
    multiblock = add_nodalsets(container, multiblock)
    multiblock = add_elementsets(container, multiblock)
    multiblock = add_surfacesets(container, multiblock)
    
    # Add mesh data -> point data, cell data
    multiblock = add_nodaldata(container, multiblock)
    multiblock = add_elementdata(container, multiblock)
    multiblock = add_surface_data(container, multiblock)
    
    # Add materials -> cell data (parameters), field data (parameter names, type, material name)
    multiblock = add_material(container, multiblock)
    
    # Add loads -> point data (resultant nodal load), cell data (resultant pressure load)
    multiblock = add_nodalload(container, multiblock)
    multiblock = add_pressure_load(container, multiblock)
    
    # Add boundary conditions -> point data (fixed boundary conditions), cell data (rigid body boundary conditions
    multiblock = add_boundary_conditions(container, multiblock)
    
    return multiblock

# =============================================================================
# Validation functions
# =============================================================================

def ensure_febio_container(data: Union[FEBioContainer, Feb]) -> FEBioContainer:
    """Ensure the input data is a FEBioContainer object."""
    if isinstance(data, Feb):
        return FEBioContainer(feb=data)
    elif isinstance(data, FEBioContainer):
        return data
    else:
        raise ValueError("Input data must be a Feb or FEBioContainer object")

# =============================================================================
# Create mesh (multiblock) from FEBioContainer
# =============================================================================

def create_multiblock_from_febio_container(container: FEBioContainer) -> pv.MultiBlock:
    """
    Converts an FEBioContainer object containing mesh data into a PyVista MultiBlock object.
    This function handles the conversion of node coordinates and element connectivity from the FEBio format (1-based indexing)
    to the PyVista format (0-based indexing). For each node set in the container, it creates a corresponding unstructured grid in the MultiBlock.

    Parameters:
        container (FEBioContainer): The FEBio container with mesh data.

    Returns:
        pv.MultiBlock: A MultiBlock object containing the mesh data.
    """
    nodes: List[Nodes] = container.nodes
    elements: List[Elements] = container.elements
    # create a MultiBlock object
    multiblock = pv.MultiBlock()
    for node in nodes:
        # get the coordinates
        coordinates = node.coordinates
        # create a cells_dict.
        # This is a dictionary that maps the element type to the connectivity
        cells_dict = {}
        for elem in elements:
            el_type: str = elem.type
            connectivity: np.ndarray = elem.connectivity - 1 # FEBio uses 1-based indexing
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

        mesh = pv.UnstructuredGrid(cells_dict, coordinates)
        multiblock.append(mesh, f"{node.name}")
    return multiblock

# =============================================================================
# Helper functions related to mesh data
# =============================================================================

def add_nodalsets(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    """
    Adds nodal sets from the FEBioContainer to the PyVista MultiBlock.
    Nodal sets define specific groups of nodes. This function maps these groups to the corresponding nodes in the PyVista grids,
    storing the indices of the nodes in the field_data of the appropriate grid.

    Parameters:
        container (FEBioContainer): The container containing nodal sets.
        multiblock (pv.MultiBlock): The MultiBlock to which the nodal sets will be added.

    Returns:
        pv.MultiBlock: The updated MultiBlock with nodal sets added.
    """
    # Calculate cumulative node counts across grids
    cumulative_nodes = np.cumsum([grid.n_points for grid in multiblock])

    for node_set in container.nodesets:
        name = node_set.name
        ids = node_set.ids - 1  # zero-based index
        grid_index = int(np.searchsorted(cumulative_nodes, ids[-1], side='right'))
        
        if grid_index == len(cumulative_nodes):
            raise ValueError(f"Could not find the proper grid for node set {name}")

        # Adjust indices for the selected grid
        if grid_index > 0:
            ids -= cumulative_nodes[grid_index - 1]

        print(f"grid_index: {grid_index}")
        multiblock[grid_index].field_data[name] = ids

    return multiblock

def add_elementsets(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    """
    Adds element sets from the FEBioContainer to the PyVista MultiBlock.
    Element sets define specific groups of elements. This function maps these groups to the corresponding elements in the PyVista grids,
    storing the indices of the elements in the field_data of the appropriate grid.

    Parameters:
        container (FEBioContainer): The container containing element sets.
        multiblock (pv.MultiBlock): The MultiBlock to which the element sets will be added.

    Returns:
        pv.MultiBlock: The updated MultiBlock with element sets added.
    """
    # Calculate cumulative element counts across grids
    cumulative_elements = np.cumsum([grid.n_cells for grid in multiblock])

    for elem_set in container.elementsets:
        name = elem_set.name
        ids = elem_set.ids - 1  # zero-based index
        grid_index = int(np.searchsorted(cumulative_elements, ids[-1], side='right'))

        if grid_index == len(cumulative_elements):
            raise ValueError(f"Could not find the proper grid for element set {name}")

        # Adjust indices for the selected grid
        if grid_index > 0:
            ids -= cumulative_elements[grid_index - 1]

        # Access the correct grid and update field data
        selected_grid = multiblock[grid_index]
        selected_grid.field_data[name] = ids

    return multiblock

def add_surfacesets(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    if len(container.surfacesets) > 0:
        print("WARNING: Surface sets are not yet supported.")
    return multiblock

# Data
# -----------------------------------------------------------------------------

def add_nodaldata(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    """
    Adds nodal data from the FEBioContainer to the PyVista MultiBlock.
    This function finds the corresponding nodeset and updates the 'point_data' of the respective grid in the MultiBlock.
    NaNs are used to fill the gaps in the data arrays to ensure consistent dimensions across the grid.

    Parameters:
        container (FEBioContainer): The container containing nodal data.
        multiblock (pv.MultiBlock): The MultiBlock where nodal data will be added.

    Returns:
        pv.MultiBlock: The updated MultiBlock with nodal data added.
    """
    nodesets = container.nodesets
    nodal_data = container.nodal_data
    # Add nodal data
    if len(nodesets) > 0:
        cumulative_nodes = np.cumsum([grid.n_points for grid in multiblock])
        for nd in nodal_data:
            # Get the nodal data and reshape if necessary
            data = nd.data.reshape(-1, 1) if len(nd.data.shape) == 1 else nd.data
            node_set = nd.node_set
            name = nd.name

            # Find the nodeset and corresponding grid using a more efficient method
            related_nodeset = next((ns for ns in nodesets if ns.name == node_set), None)
            if related_nodeset is None:
                raise ValueError(f"Node set {node_set} not found.")

            # Use binary search to find the grid
            last_id = related_nodeset.ids[-1] - 1
            grid_index = int(np.searchsorted(cumulative_nodes, last_id, side='right'))

            if grid_index == len(cumulative_nodes):
                raise ValueError(f"Could not find the proper grid for node set {node_set}")

            grid = multiblock[grid_index]
            if grid_index > 0:
                related_nodeset.ids -= cumulative_nodes[grid_index - 1]

            # Create a full data array with NaNs and assign the actual data
            full_data = np.full((grid.n_points, data.shape[1]), np.nan)
            full_data[related_nodeset.ids - 1] = data  # Adjusting for zero indexing
            grid.point_data[name] = full_data

    return multiblock

def add_elementdata(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    """
    Adds element data from the FEBioContainer to the PyVista MultiBlock.
    This function maps these properties to their corresponding elements in the PyVista grids.
    Element data is stored in the 'cell_data' of the appropriate grid.
    NaNs are used to fill the gaps in the data arrays to ensure consistent dimensions across the grid.

    Parameters:
        container (FEBioContainer): The container containing element data.
        multiblock (pv.MultiBlock): The MultiBlock where element data will be added.

    Returns:
        pv.MultiBlock: The updated MultiBlock with element data added.
        
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
        elem_ids = el_data.ids - 1
        # get the name of the data
        name = el_data.name
        # Find the proper grid
        grid = multiblock[elem_set]
        full_data = np.full((grid.n_cells, data.shape[1]), np.nan)
        full_data[elem_ids] = data
        grid[name] = full_data
    return multiblock

def add_surface_data(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    if len(container.surface_data) > 0:
        print("WARNING: Surface data is not yet supported.")
    return multiblock

# =============================================================================
# Material helper functions
# =============================================================================

def add_material(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    """
    Adds material properties from the FEBioContainer to the PyVista MultiBlock. Material properties such as Young's modulus,
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
        multiblock (pv.MultiBlock): The MultiBlock where material properties will be added.

    Returns:
        pv.MultiBlock: The updated MultiBlock with material properties added.

    Example:
        If a material in FEBio with mat_id=1 has a Young's modulus of 210 GPa and a Poisson's ratio of 0.3, after
        running this function, the parameters can be accessed in PyVista as follows:
        - Access material parameters:
          multiblock['ElementBlockName'].cell_data['mat_parameters:1']  # Array of shape [n_elements, 2]
          where the first column is Young's modulus and the second is Poisson's ratio.
        - Access material type and name:
          multiblock['ElementBlockName'].field_data['mat_type:1']  # Returns ['Elastic']
          multiblock['ElementBlockName'].field_data['mat_name:1']  # Returns ['GenericElasticMaterial']
        - Access the order of parameters:
          multiblock['ElementBlockName'].field_data['mat_parameters:1']  # Returns ['Young's modulus', 'Poisson's ratio']
    """
    elements = container.elements
    materials = container.materials
    # Create a map from element names to their corresponding grids for quick access
    element_to_grid = {elem.name: multiblock[elem.name] for elem in elements}
    
    for mat in materials:
        mat_name = mat.name
        mat_type = mat.type
        mat_id = mat.id
        parameters = OrderedDict(mat.parameters)

        # Find the corresponding element for the material
        target_element = next((elem for elem in elements if str(elem.mat) == str(mat_id)), None)
        if not target_element:
            raise ValueError(f"Could not find the proper element for material {mat_name}")
        
        # Get the corresponding grid from the map
        grid = element_to_grid.get(target_element.name)
        if not grid:
            raise ValueError(f"Could not find the proper grid for element {target_element.name}")
        
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

    return multiblock

# =============================================================================
# Load helper functions
# =============================================================================

def add_nodalload(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    """
    Adds nodal force loads from the FEBioContainer to the PyVista MultiBlock. This function interprets force loads applied to specific
    nodes as described in the FEBio model. It processes these loads, assigning them to the correct nodes based on the node sets
    specified in the loads, and stores a resultant vector for each node in point_data under the key "nodal_load". The resultant
    load vector for each node is calculated by summing all applicable force vectors along the x, y, and z axes.

    Parameters:
        container (FEBioContainer): The container containing nodal force load data.
        multiblock (pv.MultiBlock): The MultiBlock where nodal loads will be added.

    Returns:
        pv.MultiBlock: The updated MultiBlock with nodal force loads aggregated and added.

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
    cumulative_nodes = np.cumsum([grid.n_points for grid in multiblock])

    for nodal_load in nodal_loads:
        bc = nodal_load.bc.lower()  # 'x', 'y', or 'z' axis
        node_set = nodal_load.node_set
        scale = nodal_load.scale  # scale can be numeric or a reference to another data field

        related_nodeset = next((ns for ns in nodesets if ns.name == node_set), None)
        if related_nodeset is None:
            raise ValueError(f"Node set {node_set} not found.")

        last_id = related_nodeset.ids[-1] - 1
        grid_index = int(np.searchsorted(cumulative_nodes, last_id, side='right'))
        if grid_index == len(cumulative_nodes):
            raise ValueError(f"Could not find the proper grid for node set {node_set}")

        grid = multiblock[grid_index]
        if grid_index > 0:
            related_nodeset.ids -= cumulative_nodes[grid_index - 1]

        if "nodal_load" not in grid.point_data:
            grid.point_data["nodal_load"] = np.zeros((grid.n_points, 3))
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_index = axis_map[bc]
        load_indices = related_nodeset.ids - 1  # Adjust indices for zero-based indexing

        # Handle scale being a reference to other data fields or a numeric scale
        if isinstance(scale, str) and '*' in scale:
            parts = scale.split('*')
            scale_factor = float(parts[0]) if parts[0].replace('-', '', 1).isdigit() else float(parts[1])
            data_field = parts[1] if parts[0].replace('-', '', 1).isdigit() else parts[0]

            if data_field not in grid.point_data:
                raise ValueError(f"Referenced data field '{data_field}' not found in grid point data.")

            # Extract only the relevant scale data for the specified indices
            scale_data = grid.point_data[data_field][load_indices] * scale_factor
        else:
            scale_data = np.full(len(load_indices), float(scale))  # Create a full array of the scale value

        # Update the nodal load data
        grid.point_data["nodal_load"][load_indices, axis_index] += scale_data

    return multiblock

def add_pressure_load(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    if len(container.pressure_loads) > 0:
        print("WARNING: Pressure loads are not yet supported.")
    return multiblock

# =============================================================================
# Boundary condition helper functions
# =============================================================================

def add_boundary_conditions(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    """
    Adds boundary conditions from the FEBioContainer to the PyVista MultiBlock. This function handles two main types of boundary 
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
        multiblock (pv.MultiBlock): The MultiBlock where boundary conditions will be added.

    Returns:
        pv.MultiBlock: The updated MultiBlock with boundary conditions processed and added.

    Example:
        After processing, to access the constraints:
        - Displacement constraints for a specific mesh block:
          multiblock['MeshBlockName'].point_data['fix']  # Outputs a binary array where 1 indicates a fixed displacement.
        - Shell rotation constraints for the same block:
          multiblock['MeshBlockName'].point_data['fix_shell']  # Outputs a binary array where 1 indicates a fixed shell rotation.
        - For rigid body constraints related to a specific material ID:
          multiblock['MeshBlockName'].point_data['rigid_body']  # Fixed position constraints.
          multiblock['MeshBlockName'].point_data['rigid_body_rot']  # Fixed rotational constraints.
    """
    # Mapping of node sets and material ids to their corresponding grids with precomputed indices
    node_set_to_grid = {}
    element_to_grid = {}
    cumulative_nodes = np.cumsum([grid.n_points for grid in multiblock])
    
    for nodeset in container.nodesets:
        last_id = nodeset.ids[-1] - 1
        grid_index = int(np.searchsorted(cumulative_nodes, last_id, side='right'))
        if grid_index < len(multiblock):
            node_set_to_grid[nodeset.name] = (multiblock[grid_index], nodeset.ids - 1)
    
    for element in container.elements:
        element_to_grid[element.name] = multiblock[element.name]

    for bc in container.boundary_conditions:
        if isinstance(bc, FixCondition):
            node_set = bc.node_set
            if node_set not in node_set_to_grid:
                raise ValueError(f"Node set {node_set} not found.")
            
            grid, indices = node_set_to_grid[node_set]
            fixed_axes = np.zeros((grid.n_points, 3))  # For 'x', 'y', 'z'
            fixed_shells = np.zeros((grid.n_points, 3))  # For 'sx', 'sy', 'sz'
            
            # Apply constraints to axes
            for axis in ['x', 'y', 'z']:
                if axis in bc.bc:
                    fixed_axes[indices, 'xyz'.index(axis)] = 1
            for axis in ['sx', 'sy', 'sz']:
                if axis in bc.bc:
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
            material = bc.material
            for grid_name, grid in element_to_grid.items():
                if grid.material == material:
                    for axis in ['x', 'y', 'z', 'Rx', 'Ry', 'Rz']:
                        key = "rigid_body" if 'R' not in axis else "rigid_body_rot"
                        rigid_body_axes = np.zeros((grid.n_points, 3))
                        rigid_body_axes[:, 'xyz'.index(axis[-1])] = 1
                        
                        if key in grid.point_data:
                            grid.point_data[key] = grid.point_data[key].astype(int) | rigid_body_axes.astype(int)
                        else:
                            grid.point_data[key] = rigid_body_axes

    return multiblock
