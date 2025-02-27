import pyvista as pv
import numpy as np
from copy import deepcopy
from pathlib import Path
from febio_python import FEBioContainer
from febio_python.feb import FebType, Feb25, Feb30, Feb40
from febio_python.xplt import Xplt

# from febio_python.feb import Feb
from febio_python.core import (
    Nodes,
    Elements,
    FixCondition,
    RigidBodyCondition,
    States,
    StateData,
)
from typing import Union, List, Tuple

from febio_python.core.element_types import FebioElementTypeToVTKElementType
# from copy import deepcopy
from collections import OrderedDict


def febio_to_pyvista(data: Union[str, Path, FEBioContainer, Tuple, FebType, Xplt], apply_load_curves=True, **kwargs) -> List[pv.UnstructuredGrid]:
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
    container: FEBioContainer = ensure_febio_container(data, **kwargs)

    # Create a multiblock object from the FEBioContainer (nodes, elements, etc.)
    grid: pv.UnstructuredGrid = create_unstructured_grid_from_febio_container(container)

    # Add nodal sets, element sets, and surface sets
    grid = add_nodalsets(container, grid)
    if container.feb is not None:
        grid = add_element_sets(container, grid)
        grid = add_surface_sets(container, grid)

        # Add mesh data -> point data, cell data
        grid = add_nodaldata(container, grid)
        grid = add_elementdata(container, grid)
        grid = add_surface_data(container, grid)

        # Add materials -> cell data (parameters), field data (parameter names, type, material name)
        grid = add_material(container, grid)

        # Add loads -> point data (resultant nodal load), cell data (resultant pressure load)
        grid = add_nodalload(container, grid)
        grid = add_pressure_load(container, grid)
        grid = add_surface_traction_load(container, grid)

        # Add boundary conditions -> point data (fixed boundary conditions), cell data (rigid body boundary conditions
        grid = add_boundary_conditions(container, grid)

    # default return data:
    grid_or_list_of_grids = [grid]
    # Add data from xplt (results, state data)
    if container.xplt is not None:
        # If states data is available, we should create a list of grids for each state
        grid_or_list_of_grids = add_states_to_grid(container, grid, apply_load_curves=apply_load_curves)

    # # Make sure to always return a list of grids
    # if not isinstance(grid_or_list_of_grids, list):
    #     return [grid_or_list_of_grids]

    return grid_or_list_of_grids

# =============================================================================
# Validation functions
# =============================================================================


def ensure_febio_container(data: Union[FEBioContainer, FebType], **kwargs) -> FEBioContainer:
    """Ensure the input data is a FEBioContainer object."""
    if isinstance(data, (str, Path)):
        # ensure it is a path:
        filepath = Path(data)
        # make sure it is a file and exists:
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} not found.")
        # try to get the file extension:
        extension = filepath.suffix
        if extension == ".feb":
            return FEBioContainer(feb=filepath, **kwargs)
        elif extension == ".xplt":
            return FEBioContainer(xplt=filepath, **kwargs)
    elif isinstance(data, tuple):
        feb_data, xplt_data = data
        return FEBioContainer(feb=feb_data, xplt=xplt_data, **kwargs)
    elif isinstance(data, (Feb25, Feb30, Feb40)):
        return FEBioContainer(feb=data, **kwargs)
    elif isinstance(data, FEBioContainer):
        return data
    else:
        raise ValueError("Input data must be a Path, Feb or Xplt or FEBioContainer object")

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
    elements: List[Elements] = container.elements
    surfaces: List[Elements] = container.surfaces

    all_elements = deepcopy(elements) + deepcopy(surfaces)  # deep copy to avoid modifying the original data

    # create a MultiBlock object
    # First, stack all the node coordinates; this will be the points of the mesh
    coordinates = np.vstack([node.coordinates for node in nodes])

    # Next, create a cells_dict.
    # This is a dictionary that maps the element type to the connectivity
    cells_dict = OrderedDict()
    domain_identifiers = []
    element_sets = []
    for i, elem in enumerate(all_elements):
        elem_type: str = elem.type
        connectivity: np.ndarray = elem.connectivity

        if elem_type in FebioElementTypeToVTKElementType.__members__.keys():
            # get name of the element type
            elem_type = FebioElementTypeToVTKElementType[elem_type].value
        try:
            elem_type = pv.CellType[elem_type]
        except KeyError:
            raise ValueError(f"Failed to convert element type {elem_type} to a PyVista cell type.")

        # if the element type already exists in the cells_dict, append the connectivity
        if elem_type in cells_dict:
            cells_dict[elem_type] = np.vstack([cells_dict[elem_type], connectivity])
        else:
            cells_dict[elem_type] = connectivity
        
        domain_identifiers.extend([i] * len(connectivity))
        elem_name = elem.name or elem.part or elem.type
        element_sets.extend([elem_name] * len(connectivity))
    
    grid = pv.UnstructuredGrid(cells_dict, coordinates)
    grid.cell_data["domain"] = domain_identifiers
    grid["element_sets"] = element_sets

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


def add_element_sets(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
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


def add_surface_sets(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
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
            full_data[related_nodeset.ids] = data
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
        try:
            if len(data.shape) == 3:
                for col in range(data.shape[1]):
                    full_data = np.full((grid.n_cells, data.shape[2]), np.nan)
                    full_data[elem_ids] = data[:, col, :]
                    grid.cell_data[f"{name}_{col}"] = full_data
            else:
                full_data = np.full((grid.n_cells, data.shape[1]), np.nan)
                full_data[elem_ids] = data
                if name is not None:
                    grid.cell_data[name] = full_data
                elif var is not None:
                    grid.cell_data[var] = full_data
                else:
                    grid.cell_data[f"element_data_{elem_set}"] = full_data
        except ValueError as e:
            print(f"Error adding element data {name} to grid: {e}")
            continue
    return grid


def add_surface_data(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    # if len(container.surface_data) > 0:
    # print("WARNING: Surface data is not yet supported.")
    surface_data = container.surface_data

    for surf_data in surface_data:
        # print(f"Adding surface data {surf_data.name}")
        # get the surface data
        data: np.ndarray = surf_data.data
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        # get the surface set
        surf_set = surf_data.surf_set
        surf_ids = surf_data.ids
        # map ids to the grid
        mapping = grid["element_sets"] == surf_set
        selected_ids = np.where(mapping)[0]
        surf_ids = selected_ids[surf_ids]
        # get the name of the data
        name = surf_data.name
        # Find the proper grid
        full_data = np.full((grid.n_cells, data.shape[1]), np.nan)
        full_data[surf_ids] = data
        if name is not None:
            grid.cell_data[name] = full_data
        else:
            grid.cell_data[f"surface_data_{surf_set}"] = full_data

    return grid

# =============================================================================
# Material helper functions
# =============================================================================


def add_material(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """
    Adds material properties from the FEBioContainer to the PyVista UnstructuredGrid. Material properties such as Young's modulus,
    Poisson's ratio, or any other parameters defined in FEBio are associated with specific elements based on their material IDs.

    Material Parameters:
        These are transferred to PyVista as arrays in `cell_data` under "mat_parameters:{mat_id}", where each row corresponds to an
        element and each column to a material parameter. The order of parameters is consistent across `cell_data` and `field_data`.

    Material Type and Name:
        These are stored in `field_data` under "mat_type:{mat_id}" and "mat_name:{mat_id}", respectively, providing a reference to
        the type and name of the material used.

    Parameters Order:
        The names of the parameters (e.g., 'Young's modulus', 'Poisson's ratio') are stored in `field_data` under "mat_parameters:{mat_id}"
        to maintain an understanding of the data structure in `cell_data`.

    Args:
        container (FEBioContainer): The container containing material data.
        grid (pv.UnstructuredGrid): The UnstructuredGrid where material properties will be added.

    Returns:
        pv.UnstructuredGrid: The updated UnstructuredGrid with material properties added.

    Example:
        If a material in FEBio with mat_id=1 has a Young's modulus of 210 GPa and a Poisson's ratio of 0.3, after running this function,
        the parameters can be accessed in PyVista as follows:

        - Access material parameters:
          `grid.cell_data['mat_parameters:1']`  # Array of shape [n_elements, 2]
          where the first column is Young's modulus and the second is Poisson's ratio.

        - Access material type and name:
          `grid.field_data['mat_type:1']`  # Returns ['Elastic']
          `grid.field_data['mat_name:1']`  # Returns ['GenericElasticMaterial']

        - Access the order of parameters:
          `grid.field_data['mat_parameters:1']`  # Returns ['Young's modulus', 'Poisson's ratio']
    """
    # elements = container.elements
    materials = container.materials
    # domains = container.mesh_domains

    # match materials to elements based on domains
    # mat_elems_by_domain = dict()
    # if domains is not None and len(domains) > 0:

    for mat in materials:
        mat_name = mat.name
        mat_type = mat.type
        mat_id = mat.id
        mat_load_curve = mat.load_curve
        parameters = OrderedDict(mat.parameters)
        num_params = len(parameters)
        params_names = list(parameters.keys())
        params_values = list(parameters.values())

        # Initialize parameter array with NaNs
        params_array = np.full((grid.n_cells, num_params), np.nan)

        # Assign values to the parameter array
        for i, (name, value) in enumerate(zip(params_names, params_values)):
            # Directly assign scalar values
            if isinstance(value, (int, float)):
                params_array[:, i] = value
            elif isinstance(value, str):
                # If the value corresponds to existing cell data, use that data
                if value in grid.cell_data.keys():
                    params_array[:, i] = grid.cell_data[value]
                else:
                    # raise ValueError(f"Value {value} is not a valid cell data for material {mat_name}")
                    print(f"Value {value} is not a valid cell data for material {mat_name}"
                          f"Adding it as a field data instead: mat_parameter:{name}:{mat_id}")
                    grid.field_data[f"mat_parameter:{name}:{mat_id}"] = [value]
            elif value is None:
                pass
            elif isinstance(value, dict):
                pass
            else:
                raise ValueError(f"Unsupported material parameter format for {value} in material {mat_name}")

        # Store material properties in the grid
        grid.cell_data[f"mat_parameters:{mat_id}"] = params_array
        grid.field_data[f"mat_parameters:{mat_id}"] = np.array(params_names, dtype=object)
        grid.field_data[f"mat_type:{mat_id}"] = [mat_type]
        grid.field_data[f"mat_name:{mat_id}"] = [mat_name]
        grid.field_data[f"mat_load_curve:{mat_id}"] = [mat_load_curve]

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
        node_set = nodal_load.node_set
        scale = nodal_load.scale  # scale can be a tuple of length 3 or a numeric value

        related_nodeset = next((ns for ns in nodesets if ns.name == node_set), None)
        if related_nodeset is None:
            raise ValueError(f"Node set {node_set} not found.")

        if "nodal_load" not in grid.point_data:
            grid.point_data["nodal_load"] = np.zeros((grid.n_points, 3))

        load_indices = related_nodeset.ids  # Adjust indices for zero-based indexing

        # Handle scale being a tuple of length 3 (new version) or a string/numeric (old version)
        if container.feb.version >= 4.0 and isinstance(scale, tuple):
            # New version: scale is a tuple representing (scale_x, scale_y, scale_z) - THIS IS OPTIONAL
            for i, axis_scale in enumerate(scale):
                if axis_scale != 0:
                    if isinstance(axis_scale, str) and '*' in axis_scale:
                        parts = axis_scale.split('*')
                        scale_factor = float(parts[0]) if parts[0].replace('-', '', 1).isdigit() else float(parts[1])
                        data_field = parts[1] if parts[0].replace('-', '', 1).isdigit() else parts[0]

                        if data_field not in grid.point_data:
                            raise ValueError(f"Referenced data field '{data_field}' not found in grid point data.")

                        # Extract only the relevant scale data for the specified indices
                        scale_data = grid.point_data[data_field][load_indices] * scale_factor
                    elif isinstance(axis_scale, str):
                        if axis_scale not in grid.point_data:
                            raise ValueError(f"Referenced data field '{axis_scale}' not found in grid point data.")
                        scale_data = grid.point_data[axis_scale][load_indices]
                    else:
                        scale_data = np.full(len(load_indices), float(axis_scale))  # Create a full array of the scale value

                    # Update the nodal load data
                    grid.point_data["nodal_load"][load_indices, i] += scale_data
        else:
            # Old version: scale is a single value, and nodal_load.dof determines the direction
            bc = nodal_load.dof.lower()  # 'x', 'y', or 'z' axis
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            axis_index = axis_map[bc]

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

    pressure_loads = container.pressure_loads
    # surface_data = container.surface_data
    if len(pressure_loads) == 0:
        return grid  # No pressure loads to process

    # add default pressure load
    grid.cell_data["pressure_load"] = np.zeros(grid.n_cells)

    # get elements
    # elements_by_name = {elem.name: elem for elem in container.elements}

    for load in pressure_loads:
        # get the surface set
        surf_set = load.surface
        # # get related elements
        # if not surf_set in elements_by_name:
        #     raise ValueError(f"Surface {surf_set} not found."
        #                      "Make sure the 'surface' matches one of the element names.")
        # related_elements = elements_by_name[surf_set]
        # get the scale
        scale = load.scale
        # if load is number, we should convert to a numpy array
        # check if it is a str that can be converted to a number directly
        if isinstance(scale, str):
            try:
                scale = float(scale)
            except ValueError:
                pass
        
        if isinstance(scale, (int, float, np.number)):
            scale_factor = float(scale)
            scale = np.full(grid.n_cells, 0.0)
            # apply the scale to the elements
            mapping = grid["element_sets"] == surf_set
            selected_ids = np.where(mapping == True)[0]  # noqa: E712
            scale[selected_ids] = scale_factor
        elif isinstance(scale, str):
            # check if there is a '*' in the scale (indicating a multiplication)
            if "*" in scale:
                parts = scale.split('*')
                scale_factor = float(parts[0]) if parts[0].replace('-', '', 1).isdigit() else float(parts[1])
                data_field = parts[1] if parts[0].replace('-', '', 1).isdigit() else parts[0]
                if data_field not in grid.cell_data:
                    raise ValueError(f"Referenced data field '{data_field}' not found in grid cell data.")
                scale = grid.cell_data[data_field] * scale_factor
            else:
                if scale not in grid.cell_data:
                    raise ValueError(f"Referenced data field '{scale}' not found in grid cell data.")
                scale = grid.cell_data[scale]
        # add the pressure load to the grid
        grid.cell_data["pressure_load"] += scale

    return grid

def add_surface_traction_load(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:

    traction_loads = container.surface_traction_loads
    # surface_data = container.surface_data
    if len(traction_loads) == 0:
        return grid  # No pressure loads to process

    # add default pressure load
    grid.cell_data["traction_load"] = np.zeros((grid.n_cells, 3))

    # get elements
    for load in traction_loads:
        # get the surface set
        surf_set = load.surface
        # get the scale
        scale = load.scale
        # if load is number, we should convert to a numpy array
        if isinstance(scale, str):
            try:
                scale = float(scale)
            except ValueError:
                pass
        # get the traction vector
        traction = load.traction_vector
        
        if isinstance(scale, (int, float, np.number)):
            scale_factor = float(scale)
            scale = np.full(grid.n_cells, 0.0)
            # apply the scale to the elements
            mapping = grid["element_sets"] == surf_set
            selected_ids = np.where(mapping == True)[0]  # noqa: E712
            scale[selected_ids] = scale_factor
        elif isinstance(scale, str):
            # check if there is a '*' in the scale (indicating a multiplication)
            if "*" in scale:
                parts = scale.split('*')
                scale_factor = float(parts[0]) if parts[0].replace('-', '', 1).isdigit() else float(parts[1])
                data_field = parts[1] if parts[0].replace('-', '', 1).isdigit() else parts[0]
                if data_field not in grid.cell_data:
                    raise ValueError(f"Referenced data field '{data_field}' not found in grid cell data.")
                scale = grid.cell_data[data_field] * scale_factor
            else:
                if scale not in grid.cell_data:
                    raise ValueError(f"Referenced data field '{scale}' not found in grid cell data.")
                scale = grid.cell_data[scale]
        
        # add the pressure load to the grid
        # make sure that traction and scale have the same shape
        if isinstance(scale, (int, float, np.number)):
            scaled_traction = traction * scale
        else:
            num_scalars = len(scale)
            # need to repeat the traction vector for each scalar
            traction = traction[np.newaxis, :]
            repeated_traction = np.repeat(traction, num_scalars, axis=0)
            scaled_traction = repeated_traction * scale[:, np.newaxis]
        
        # print("traction: ", traction)
        # print("scaled_traction: ", scaled_traction)
            
        grid.cell_data["traction_load"] += scaled_traction

    return grid

# =============================================================================
# Boundary condition helper functions
# =============================================================================


def add_boundary_conditions(container: FEBioContainer, grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """
    Adds boundary conditions from the FEBioContainer to the PyVista UnstructuredGrid. This function
    manages two main types of boundary conditions: fixed conditions (FixCondition) and rigid body
    conditions (RigidBodyCondition).

    Args:
        container (FEBioContainer): The container containing boundary conditions.
        grid (pv.UnstructuredGrid): The PyVista UnstructuredGrid where boundary conditions
            will be added.

    Returns:
        pv.UnstructuredGrid: The updated UnstructuredGrid with processed and added boundary conditions.

    Example:
        After processing, to access the constraints:

        - Displacement constraints for a specific mesh block:
          `grid.point_data['fix']`  # Outputs a binary array where 1 indicates a fixed displacement.

        - Shell rotation constraints for the same block:
          `grid.point_data['fix_shell']`  # Outputs a binary array where 1 indicates a fixed shell rotation.

        - For rigid body constraints related to a specific material ID:
          `grid.point_data['rigid_body']`  # Fixed position constraints.
          `grid.point_data['rigid_body_rot']`  # Fixed rotational constraints.

    Fixed Conditions:
        Applies constraints on node displacements ('x', 'y', 'z') or shell rotations ('sx', 'sy', 'sz') for
        specific node sets. Recorded as binary arrays in `point_data`, each element represents whether
        a node is fixed along a certain axis:

        - 'fix': Binary array of shape [n_points, 3], indicating fixed displacements in X, Y, and Z directions.
        - 'fix_shell': Binary array of shape [n_points, 3], indicating fixed rotations in X, Y, and Z directions.

        Both arrays consolidate all applicable constraints per node, summing constraints where multiple
        conditions affect the same node.

    Rigid Body Conditions:
        Restrict the movement or rotation of all nodes associated with a specific material, denoted by:

        - 'rigid_body': Binary array of shape [n_points, 3], indicating fixed positions in X, Y, and Z
          directions for nodes associated with a material.
        - 'rigid_body_rot': Binary array of shape [n_points, 3], indicating fixed rotations in X, Y, and
          Z directions for nodes associated with a material.

        These conditions are labeled with specific material IDs, enhancing traceability and management
        in complex models.
    """

    for bc in container.boundary_conditions:
        if isinstance(bc, FixCondition):
            node_set = bc.node_set
            if node_set not in grid.point_data:
                raise ValueError(f"Node set {node_set} not found.")

            # grid, indices = grid[node_set]
            indices = np.where(grid.point_data[node_set] == 1)[0]  # Get the indices of the nodes in the node set
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
        lc_id = int(lc.id)
        # lc_type = lc.interpolate_type
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


def add_states_to_grid(container: FEBioContainer, grid: pv.UnstructuredGrid, apply_load_curves=True) -> List[pv.UnstructuredGrid]:

    # First, check if .xplt is provided
    if container.xplt is None:
        return grid  # No states to add

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
        dom_index = elem_state.dom
        # domain = container.mesh_domains[dom_index]
        # print(f"Adding element state {name} with shape {data.shape} on grid with number of cells {grid.n_cells} | element domain: {domain.name}")
        for i, grid in enumerate(state_grids):
            this_data = data[i]
            # if data does not match the number of cells, we need to add to specific cells based on the domain
            if this_data.shape[0] != grid.n_cells:
                this_data = np.full((grid.n_cells, this_data.shape[1]), np.nan)
                mask = grid.cell_data["domain"] == dom_index
                this_data[mask] = data[i]
            grid.cell_data[name] = this_data

    # Add surface states
    for surf_state in surface_states:
        name = surf_state.name
        data = surf_state.data
        for i, grid in enumerate(state_grids):
            grid.cell_data[name] = data[i]

    # Now, we need to interpolate the load curves, based on the timesteps
    if container.feb is not None and apply_load_curves:
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
        pressure_loads = container.pressure_loads
        for load in pressure_loads:
            # get related load curve
            lc = interpolators[int(load.load_curve)]
            # get related cell ids
            surf_set = load.surface
            mapping = grid["element_sets"] == surf_set
            selected_ids = np.where(mapping)[0]
            # modify the load magnitude
            for i, grid in enumerate(state_grids):
                grid.cell_data["pressure_load"][selected_ids] *= lc(timesteps[i])
            # # update the pressure load vector
            # for i, grid in enumerate(state_grids):
            #     # re-calculate the pressure load vector
            #     # Now, we need to add the pressure load as a vector
            #     grid.cell_data["pressure_load"] = np.zeros((grid.n_cells, 3))
            #     # The vector is based on the normal of the elements
            #     # extract the normals
            #     extracted_surface = grid.extract_surface()
            #     extracted_surface = extracted_surface.compute_normals(cell_normals=True,
            #                                                 point_normals=False,
            #                                                 flip_normals=False,
            #                                                 consistent_normals=True)
            #     normals = extracted_surface["Normals"]
            #     # get original cell ids
            #     cell_ids = extracted_surface["vtkOriginalCellIds"]
            #     # add the normals to the grid based on the cell ids
            #     grid.cell_data["pressure_load"][cell_ids] = normals
            #     # multiply the normals by the pressure load magnitude
            #     grid.cell_data["pressure_load"] *= grid.cell_data["pressure_load_magnitude"][:, None]
        
        # handle traction loads
        traction_loads = container.surface_traction_loads
        for load in traction_loads:
            # get related load curve
            lc = interpolators[int(load.load_curve)]
            # get related cell ids
            surf_set = load.surface
            mapping = grid["element_sets"] == surf_set
            selected_ids = np.where(mapping)[0]
            # modify the load magnitude
            for i, grid in enumerate(state_grids):
                grid.cell_data["traction_load"][selected_ids] *= lc(timesteps[i])

        # -----
        # handle material loads
        materials = container.materials
        for mat in materials:
            mat.load_curve  # dict -> [mat_param_name, lc_id]
            if mat.load_curve is None:
                continue  # no load curve to apply
            for mat_param_name, lc_id in mat.load_curve.items():
                # get the interpolator
                lc = interpolators[lc_id]
                # get the material id
                mat_id = mat.id
                # get the index of the parameter
                param_index = list(grid.field_data[f"mat_parameters:{mat_id}"]).index(mat_param_name)
                # apply the modification
                for i, grid in enumerate(state_grids):
                    # get the data
                    time_scale = lc(timesteps[i])
                    # get the material data
                    mat_data = grid.cell_data[f"mat_parameters:{mat_id}"].copy()
                    # apply the modification to the load in the grid
                    current_data = mat_data[:, param_index]
                    new_data = current_data * time_scale
                    mat_data[:, param_index] = new_data
                    # update the grid
                    grid.cell_data[f"mat_parameters:{mat_id}"] = mat_data
        
    return state_grids


# =============================================================================
# OTHER UTILITIES
# =============================================================================

def split_mesh_into_surface_and_volume(mesh: pv.UnstructuredGrid, surface_cell_types=None) -> Tuple[pv.UnstructuredGrid, pv.UnstructuredGrid]:
    """
    Splits a mesh into a surface mesh and a volume mesh. The surface mesh contains only the outer surface of the original mesh,
    while the volume mesh contains the original mesh without the outer surface.

    Parameters:
        mesh (pv.UnstructuredGrid): The mesh to split.

    Returns:
        Tuple[pv.UnstructuredGrid, pv.UnstructuredGrid]: A tuple containing the surface mesh and the volume mesh, respectively.
    """
    if surface_cell_types is None:
        from febio_python.core import SURFACE_ELEMENT_TYPES
        surface_cell_types = set([pv.CellType[k].value for k in SURFACE_ELEMENT_TYPES.__members__.keys()])
    # Extract the surface mesh
    surface_mesh = mesh.copy().extract_cells_by_type(list(surface_cell_types))

    # # Remove duplicate faces
    # unique_cell_ids = set()
    # if len(surface_mesh.cells_dict) == 1:
    #     for cell_type, cells in surface_mesh.cells_dict.items():
    #         unique_cells, ids = np.unique(cells, axis=0, return_index=True)
    #         unique_cell_ids.update(list(ids))
    # else:  # THIS VERSION WORKS, BUT IT IS TOO SLOW
    #     all_cells = list()
    #     for cell_idx in range(surface_mesh.n_cells):
    #         all_cells.append(surface_mesh.get_cell(cell_idx).point_ids)
    #     unique_cells, unique_cell_ids = np.unique(all_cells, axis=0, return_index=True)

    # if len(unique_cell_ids) < surface_mesh.n_cells:
    #     unique_cell_ids = list(unique_cell_ids)
    #     all_cell_ids = list(range(surface_mesh.n_cells))
    #     # selected_cell_ids = list(set(all_cell_ids) - set(unique_cell_ids))
    #     selected_cell_ids = unique_cell_ids
    #     surface_mesh = surface_mesh.extract_cells(selected_cell_ids)
    #     surface_mesh.cell_data.pop("vtkOriginalCellIds", None)

    # Extract the volume mesh
    other_cells = list(set(mesh.celltypes) - surface_cell_types)
    volume_mesh = mesh.copy().extract_cells_by_type([pv.CellType(k) for k in other_cells])

    # Add point data from the original mesh to the new meshes (if they are not already present)
    for key, value in mesh.point_data.items():
        if key not in volume_mesh.point_data.keys():
            volume_mesh.point_data[key] = value

    for key, value in mesh.cell_data.items():
        if key not in volume_mesh.cell_data.keys():
            volume_mesh.cell_data[key] = value

    # Add field data from the original mesh to the new meshes (if they are not already present)
    for key, value in mesh.field_data.items():
        if key not in volume_mesh.field_data.keys():
            volume_mesh.field_data[key] = value
        if key not in surface_mesh.field_data.keys():
            surface_mesh.field_data[key] = value

    return surface_mesh, volume_mesh


def carefully_pass_cell_data_to_point_data(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """This function corrects the 'cell_data_to_point_data' behavior from pyvista when there are NaN values in the cell data.
    In the original implementation, if there are NaN values in the cell data, all the point data is set to NaN. This function
    corrects this behavior by only setting the point data to NaN when there are no valid data to interpolate from the cell data.
    This is done by finding the cells that have valid data first, then converting only those cells to point data, while the
    remaining cells are ignored in the conversion (not affecting the point data and leaving it as NaN for those cells).
    This is useful when converting data that is defined only for a portion of the cells in the mesh, such as surface loads.
    e.g. surface loads are usually defined for only one side of the mesh, so the other side will have NaN values in the cell data.
    If the original implementation is used, the entire point data will be set to NaN, which is not desired. If we try to
    fill the NaN values with zeros, it will affect the interpolation and the results will be incorrect (border values will be
    interpolated incorrectly). Thus, this function is a workaround to handle this issue.

    Args:
        mesh (pv.UnstructuredGrid): The mesh to convert cell data to point data.

    Returns:
        pv.UnstructuredGrid: The mesh with cell data converted to point data, handling NaN values correctly.
    """

    from pykdtree.kdtree import KDTree
    import xxhash

    # Use the original implementation to convert cell data to point data
    new_mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
    # Before we proceed, first check if there are NaN values in any point data
    found_nan_in_keys = []
    for key, value in new_mesh.point_data.items():
        # First, check if the key is in the cell data, if not,
        # we do not need to correct the point data (since it is not related to this function)
        if key not in mesh.cell_data.keys():
            continue
        if np.isnan(value).any():
            found_nan_in_keys.append(key)
    # If there are no NaN values, we can skip the correction
    if len(found_nan_in_keys) == 0:
        return new_mesh

    # Otherwise, we need to correct the point data with NaN values
    # Initialize a KDTree to find the closest points
    tree = KDTree(new_mesh.points.astype(np.double))
    # Create a new dictionary to store the extracted data
    dyn_extracted = dict()
    # Correct the point data with NaN values
    for key in found_nan_in_keys:
        # Get the original cell data
        original_cell_data = mesh.cell_data[key]
        # find cells that have valid data
        valid_cell_indexes = np.where(~np.isnan(original_cell_data))[0]
        # If all cells have NaN values, we can skip this key, we cannot interpolate the data
        # If the number of valid cells is the same as the total number of cells, we can skip the correction
        if valid_cell_indexes.size == 0 or valid_cell_indexes.size == mesh.n_cells:
            # no valid data found
            continue
        # hash the valid cell indexes, so that we can check if we have already extracted the data
        # hash_valid_cell_indexes = hash(tuple(valid_cell_indexes))
        contiguous_array = np.ascontiguousarray(valid_cell_indexes)
        hash_valid_cell_indexes = xxhash.xxh64(contiguous_array.tobytes()).hexdigest()
        if hash_valid_cell_indexes not in dyn_extracted:
            # extract the cells that have valid data
            valid_cells = mesh.extract_cells(valid_cell_indexes)
            # convert the valid cells to point data
            valid_cells = valid_cells.cell_data_to_point_data(pass_cell_data=False)
            # get the corresponding point indexes
            pts = np.array(valid_cells.points)
            _, point_map = tree.query(pts)
            # Add the extracted data to the dictionary
            dyn_extracted[hash_valid_cell_indexes] = (valid_cells, point_map)
        else:
            # retrieve the data from the dictionary
            (valid_cells, point_map) = dyn_extracted[hash_valid_cell_indexes]

        # send the valid data to the original mesh
        new_mesh.point_data[key][point_map] = valid_cells.point_data[key]

    return new_mesh


# =============================================================================
from febio_python.core import FEBioElementType

class PyvistaToFEBioElementType:
    # Mapping between PyVista CellType and FEBioElementType
    MAPPING = {
        pv.CellType.HEXAHEDRON: FEBioElementType.HEXAHEDRON,
        pv.CellType.TRIANGLE: FEBioElementType.TRIANGLE,
        pv.CellType.QUAD: FEBioElementType.QUAD,
        pv.CellType.TETRA: FEBioElementType.TETRA,
        pv.CellType.WEDGE: FEBioElementType.WEDGE,
        # Quadratic elements
        pv.CellType.QUADRATIC_TRIANGLE: FEBioElementType.QUADRATIC_TRIANGLE,
        pv.CellType.QUADRATIC_QUAD: FEBioElementType.QUADRATIC_QUAD,
        pv.CellType.QUADRATIC_TETRA: FEBioElementType.QUADRATIC_TETRA,
        pv.CellType.QUADRATIC_WEDGE: FEBioElementType.QUADRATIC_WEDGE,
        pv.CellType.QUADRATIC_HEXAHEDRON: FEBioElementType.QUADRATIC_HEXAHEDRON,
        # Higher-order elements
        pv.CellType.BIQUADRATIC_QUAD: FEBioElementType.BIQUADRATIC_QUAD,
        pv.CellType.TRIQUADRATIC_HEXAHEDRON: FEBioElementType.TRIQUADRATIC_HEXAHEDRON,
        pv.CellType.HIGHER_ORDER_TETRAHEDRON: FEBioElementType.HIGHER_ORDER_TETRA
    }

    @staticmethod
    def map(cell_type):
        """Map PyVista CellType to FEBioElementType."""
        return PyvistaToFEBioElementType.MAPPING.get(cell_type, None)

    def __call__(self, cell_type):
        return self.map(cell_type)