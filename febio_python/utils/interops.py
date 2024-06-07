import pyvista as pv
import numpy as np

from febio_python.feb import Feb
from febio_python.feb import (
    Nodes,
    Elements,
)
from typing import List

from febio_python.utils.element_types import FebioElementTypeToVTKElementType

def feb_to_pyvista(feb: Feb):
    
    # first, get nodes data
    nodes: List[Nodes] = feb.get_nodes()
    elements: List[Elements] = feb.get_elements()
    
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
        multiblock.append(mesh)
    
    return multiblock
    
    