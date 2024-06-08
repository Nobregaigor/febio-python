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
    # Extract the nodal sets
    nodesets = container.nodesets
    for node_set in nodesets:
        name = node_set.name
        ids = node_set.ids - 1
        node_count = 0
        selected_grid = None
        for grid in multiblock:
            node_count += grid.n_points
            if ids[-1] < node_count:
                selected_grid = grid
                break
        if selected_grid is None:
            raise ValueError(f"Could not find the proper grid for node set {name}")
        grid.field_data[name] = ids
    
    return multiblock

def add_elementsets(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    # Extract the element sets
    elementsets = container.elementsets
    for elem_set in elementsets:
        name = elem_set.name
        ids = elem_set.ids - 1
        elem_count = 0
        selected_grid = None
        for grid in multiblock:
            elem_count += grid.n_cells
            if ids[-1] < elem_count:
                selected_grid = grid
                break
        if selected_grid is None:
            raise ValueError(f"Could not find the proper grid for element set {name}")
        grid.field_data[name] = ids  
    
    return multiblock

def add_surfacesets(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    if len(container.surfacesets) > 0:
        print("WARNING: Surface sets are not yet supported.")
    # # Extract the surface sets
    # surfacesets = container.surfacesets
    # for surf_set in surfacesets:
    #     name = surf_set.name
    #     ids = surf_set.ids - 1
    #     surf_count = 0
    #     selected_grid = None
    #     for grid in multiblock:
    #         surf_count += grid.n_cells
    #         if ids[-1] < surf_count:
    #             selected_grid = grid
    #             break
    #     if selected_grid is None:
    #         raise ValueError(f"Could not find the proper grid for surface set {name}")
    #     grid.field_data[name] = ids  
    
    return multiblock

def add_nodaldata(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    nodesets = container.nodesets
    nodal_data = container.nodal_data
    # Add nodal data
    if len(nodesets) > 0:
        for nd in nodal_data:
            # get the nodal data
            data: np.ndarray = nd.data
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            # get the node set
            node_set = nd.node_set
            node_ids = nd.ids - 1
            # get the name of the data
            name = nd.name
            # Find the proper grid
            # In febio, node data is related to a nodeset;
            # this nodeset indicated the id of the nodes;
            # there is not information about which "Nodes" object
            # is related to this nodeset. e.g. they have only the node ids
            # Thus, we must first find which nodeset this data is related to
            # then we must get the ids of this nodeset, and finally
            # we must find the proper grid based on the node ids and number
            # of nodes in each block of the multiblock.
            # related_nodeset = 
            related_nodeset = None
            for nodeset_item in nodesets:
                if nodeset_item.name == node_set:
                    related_nodeset = nodeset_item
                    break
            if related_nodeset is None:
                raise ValueError(f"Node set {node_set} not found.")
            
            # Now, we must find the proper grid based on ids
            last_id = related_nodeset.ids[-1] - 1
            node_count = 0
            grid = None
            for this_grid in multiblock:
                node_count += this_grid.n_points
                if last_id < node_count:
                    grid = this_grid
                    break
            if grid is None:
                raise ValueError(f"Could not find the proper grid for node set {node_set}")
            # grid = multiblock[node_set]
            full_data = np.full((grid.n_points, data.shape[1]), np.nan)
            full_data[node_ids] = data
            grid[name] = full_data
    return multiblock

def add_elementdata(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
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
    elements: List[Elements] = container.elements
    # Extract the materials
    materials = container.materials
    mat_grid_params_list = []
    for mat in materials:
        mat_name = mat.name
        mat_type = mat.type
        mat_id = mat.id
        parameters = OrderedDict(mat.parameters)        
        # For material, we must find the element that matched the material id
        # then we must add the material as field data to the grid corresponding to this element
        added_mat = False
        for elem in elements:
            if str(elem.mat) == str(mat_id):
                grid = multiblock[elem.name]
                
                num_params = len(parameters)
                params_names = list(parameters.keys())
                params_values = list(parameters.values())
                
                params_array = np.full((grid.n_cells, num_params), np.nan)
                
                for i, value in enumerate(params_values):                    
                    # value can be a scalar (int, float) or a string
                    if isinstance(value, (int, float)):
                        params_array[:, i] = value
                    elif isinstance(value, str):
                        # check if the value is a valid cell data (e.g. mesh data)
                        if value in grid.cell_data.keys():
                            # full_data = deepcopy(grid.cell_data[value])
                            params_array[:, i] = deepcopy(grid.cell_data[value])
                            del grid.cell_data[value]
                        else:
                            raise ValueError(f"Value {value} is not a valid cell data"
                                             "FEBio allows to input material parameters in 3 formats: "
                                             "1. scalar (int, float), "
                                             "2. string (corresponding to a mesh data), "
                                             "3. math expression (e.g. 2*value) or based on constants (e.g. 2*PI) "
                                             "or based on coordinates (e.g. 2*X). "
                                             "However, we currently only support the first two formats."
                                             "Please make sure that the value is a valid cell data. "
                                             )
                    else:
                        raise ValueError(f"Value {value} is not a valid material parameter. "
                                         "FEBio allows to input material parameters in 3 formats: "
                                         "1. scalar (int, float), "
                                         "2. string (corresponding to a mesh data), "
                                         "3. math expression (e.g. 2*value) or based on constants (e.g. 2*PI) "
                                         "or based on coordinates (e.g. 2*X). "
                                         "However, we currently only support the first two formats."
                                         )                
                grid.cell_data[f"mat_parameters:{mat_id}"] = params_array
                grid.field_data[f"mat_parameters:{mat_id}"] = np.array(params_names, dtype=object)
                grid.field_data[f"mat_type:{mat_id}"] = [mat_type]
                grid.field_data[f"mat_name:{mat_id}"] = [mat_name]
                added_mat = True
                break
        if not added_mat:
            raise ValueError(f"Could not find the proper grid for material {mat_name}")

    return multiblock

# =============================================================================
# Load helper functions
# =============================================================================

def add_nodalload(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    nodesets = container.nodesets
    # Extract the nodal loads
    nodal_loads = container.nodal_loads
    for nodal_load in nodal_loads:
        bc = nodal_load.bc
        node_set = nodal_load.node_set
        scale = nodal_load.scale
        load_curve = nodal_load.load_curve
        # Find the proper grid
        related_nodeset = None
        for nodeset_item in nodesets:
            if nodeset_item.name == node_set:
                related_nodeset = nodeset_item
                break
        if related_nodeset is None:
            raise ValueError(f"Node set {node_set} not found.")
        last_id = related_nodeset.ids[-1] - 1
        node_count = 0
        grid = None
        for this_grid in multiblock:
            node_count += this_grid.n_points
            if last_id < node_count:
                grid = this_grid
                break
        if grid is None:
            raise ValueError(f"Could not find the proper grid for node set {node_set}")
        
        scale = str(scale)
        scale_sign = 1
        if scale.startswith("1*"):
            scale = scale[2:]
        elif scale.startswith("-1*"):
            scale = "-" + scale[3:]
            scale_sign = -1
        elif scale.endswith("*1"):
            scale = scale[:-2]
        elif scale.endswith("*-1"):
            scale = scale[:-3]
            scale_sign = -1

        scale_name = f"nodal_load:{bc.lower()}:{node_set}"
        scale_data = scale
        if scale in grid.point_data.keys():
            scale_name = f"nodal_load:{bc.lower()}:{scale}"
            scale_data = deepcopy(grid.point_data[scale])
            # delete the scale data
            del grid.point_data[scale]            
            scale_data = scale_data[related_nodeset.ids - 1]
        full_data = np.full((grid.n_points, 3), 0.0)
        
        if str(bc).lower() == "x":
            full_data[related_nodeset.ids - 1, 0] = scale_data * scale_sign
        elif str(bc).lower() == "y":
            full_data[related_nodeset.ids - 1, 1] = scale_data * scale_sign
        elif str(bc).lower() == "z":
            full_data[related_nodeset.ids - 1, 2] = scale_data * scale_sign
        
        # full_data[related_nodeset.ids - 1] = scale
        grid.point_data[scale_name] = full_data
    
    # resolve nodal loads -> final vector field of nodal loads
    for grid in multiblock:
        nodal_loads = [key for key in grid.point_data.keys() if "nodal_load" in key]
        full_data = np.full((grid.n_points, 3), 0.0)
        for nodal_load in nodal_loads:
            data = grid.point_data[nodal_load]
            full_data += data
        grid.point_data["nodal_load"] = full_data
        for nodal_load in nodal_loads:
            del grid.point_data[nodal_load]
    
    return multiblock

def add_pressure_load(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    if len(container.pressure_loads) > 0:
        print("WARNING: Pressure loads are not yet supported.")
    return multiblock

# =============================================================================
# Boundary condition helper functions
# =============================================================================

def add_boundary_conditions(container: FEBioContainer, multiblock: pv.MultiBlock) -> pv.MultiBlock:
    
    nodesets = container.nodesets
    elements: List[Elements] = container.elements
    # Extract the boundary conditions
    boundary_conditions = container.boundary_conditions
    
    for bc in boundary_conditions:
        
        if isinstance(bc, FixCondition):
            bc_type = bc.bc
            node_set = bc.node_set
            # Find the proper grid
            related_nodeset = None
            for nodeset_item in nodesets:
                if nodeset_item.name == node_set:
                    related_nodeset = nodeset_item
                    break
            if related_nodeset is None:
                raise ValueError(f"Node set {node_set} not found.")
            last_id = related_nodeset.ids[-1] - 1
            node_count = 0
            grid = None
            for this_grid in multiblock:
                node_count += this_grid.n_points
                if last_id < node_count:
                    grid = this_grid
                    break
            if grid is None:
                raise ValueError(f"Could not find the proper grid for node set {node_set}")
            
            bc_values = np.full((grid.n_points,), 0)
            bc_values[related_nodeset.ids - 1] = 1
            if "x" in bc_type:
                _fix_mode = "x"
                if "sx" in bc_type:
                    _fix_mode = "sx"
                grid.point_data[f"fix:{_fix_mode}:{node_set}"] = bc_values
            if "y" in bc_type:
                _fix_mode = "y"
                if "sy" in bc_type:
                    _fix_mode = "sy"
                grid.point_data[f"fix:{_fix_mode}:{node_set}"] = bc_values
            if "z" in bc_type:
                _fix_mode = "z"
                if "sz" in bc_type:
                    _fix_mode = "sz"
                grid.point_data[f"fix:{_fix_mode}:{node_set}"] = bc_values
        
        elif isinstance(bc, RigidBodyCondition):
            material = bc.material
            fixed_axes = bc.fixed_axes
            # Find the proper grid
            added_mat = False
            for elem in elements:
                if str(elem.mat) == str(material):
                    grid = multiblock[elem.name]
                    grid.point_data[f"rigid_body:{material}:{fixed_axes}"] = np.full((grid.n_points, 1), 1)
                    added_mat = True
                    break
            if not added_mat:
                raise ValueError(f"Could not find the proper grid for material {material}")
    
    # resolve "fix" (fixed boundary conditions) -> final vector field of fixed boundary conditions
    for grid in multiblock:
        fix_keys = [key for key in grid.point_data.keys() if "fix:" in key]
        full_data = np.full((grid.n_points, 6), 0.0)
        for key in fix_keys:
            fix_mode = key.split(":")[1]
            data = grid.point_data[key]
            if fix_mode == "x":
                full_data[:, 0] += data.flatten()
            elif fix_mode == "y":
                full_data[:, 1] += data.flatten()
            elif fix_mode == "z":
                full_data[:, 2] += data.flatten()
            elif fix_mode == "sx":
                full_data[:, 3] += data.flatten()
            elif fix_mode == "sy":
                full_data[:, 4] += data.flatten()
            elif fix_mode == "sz":
                full_data[:, 5] += data.flatten()
        # transform into a binary array (0: free, 1: fixed), but for each axis
        full_data = np.where(full_data > 0, 1, 0)
        grid.point_data["fix"] = full_data
        for fix_mode in fix_keys:
            del grid.point_data[fix_mode]

    return multiblock
