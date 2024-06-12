from pathlib import Path
from typing import Union, List
from functools import cache

from .xplt_parser import read_xplt

from febio_python.core import (
    XpltMesh,
    States,
    SURFACE_EL_TYPE,
    Nodes,
    Elements,
    NodeSet,
    StateData,
    XpltMeshPart
)

class Xplt():
    def __init__(self, filepath: Union[Path, str]=None, xplt_mesh: Union[XpltMesh, None] = None, states: Union[States, None] = None) -> None:
        
        if filepath is not None:
            xplt_mesh, states = read_xplt(filepath)
        
        if xplt_mesh is None:
            raise ValueError("xplt_mesh is None. Check input file or input parameters.")
        if states is None:
            raise ValueError("states is None. Check input file or input parameters.")
        
        # Set attributes
        self.xplt_mesh: XpltMesh = xplt_mesh
        self.states: States = states
    
    @cache
    def __repr__(self) -> str:
        to_print = f"Xplt object [{id(self)}]:\n"
        
        to_print += "=== xplt_mesh: ===\n"
        nodes_info = ""
        if self.xplt_mesh.nodes is not None:
            for no in self.xplt_mesh.nodes:
                nodes_info += f"--->{no.name}: {no.coordinates.shape}\n"
        else:
            nodes_info += "None\n"
        to_print += f"-Nodes:\n{nodes_info}"
        
        elements_info = ""
        if self.xplt_mesh.elements is not None:
            for el in self.xplt_mesh.elements:
                elements_info += f"--->{el.name}: {el.connectivity.shape}\n"
        else:
            elements_info += "None\n"
        
        to_print += f"-Elements:\n{elements_info}"
        
        states_node_info = ""
        if self.states.nodes is not None:
            for sn in self.states.nodes:
                states_node_info += f"--->{sn.name}: {sn.data.shape}\n"
        else:
            states_node_info += "None\n"
            
        states_elem_info = ""
        if self.states.elements is not None:
            for se in self.states.elements:
                states_elem_info += f"--->{se.name}: {se.data.shape}\n"
        else:
            states_elem_info += "None\n"
            
        states_surface_info = ""
        if self.states.surfaces is not None:
            for ss in self.states.surfaces:
                states_surface_info += f"--->{ss.name}: {ss.data.shape}\n"
        else:
            states_surface_info += "None\n"
        
        states_info = ""
        states_info += f"-Nodes:\n{states_node_info}"
        states_info += f"-Elements:\n{states_elem_info}"
        states_info += f"-Surfaces:\n{states_surface_info}"
        
        to_print += f"=== States: ===\n{states_info}"
        
        return to_print

    
    # ========================================================================
    # Mesh-related properties
    # ========================================================================
    
    @property
    def nodes(self) -> List[Nodes]:
        return self.xplt_mesh.nodes

    @property
    def elements(self) -> List[Elements]:
        return self.xplt_mesh.elements
    
    @property
    def surfaces(self) -> List[Elements]:
        return self.xplt_mesh.surfaces
    
    @property
    def volumes(self) -> List[Elements]:
        # get all elements
        all_elements = self.elements
        # filter elements by surface type
        filtered = []
        for elem in all_elements:
            if elem.type not in SURFACE_EL_TYPE.__members__:
                filtered.append(elem)
        return filtered
    
    @property
    def nodesets(self) -> List[NodeSet]:
        return self.xplt_mesh.nodesets
    
    @property
    def parts(self) -> List[XpltMeshPart]:
        return self.xplt_mesh.parts
    
    # ========================================================================
    # States-related properties
    # ========================================================================
    
    @property
    def node_states(self) -> List[StateData]:
        return self.states.nodes
    
    @property
    def element_states(self) -> List[StateData]:
        return self.states.elements

    @property
    def surface_states(self) -> List[StateData]:
        return self.states.surfaces
    