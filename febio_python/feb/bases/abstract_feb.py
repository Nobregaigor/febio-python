from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree
import xml.etree.ElementTree as ET

from . import FebBaseObject
import numpy as np
from typing import Union, Dict, List
from collections import OrderedDict, deque
from abc import abstractmethod, ABC

from febio_python.core import (
    Nodes,
    Elements,
    NodeSet,
    SurfaceSet,
    ElementSet,
    Material,
    NodalLoad,
    SurfaceLoad,
    LoadCurve,
    BoundaryCondition,
    FixCondition,
    # FixedAxis,
    RigidBodyCondition,
    NodalData,
    SurfaceData,
    ElementData,
)

from .._caching import feb_instance_cache

class AbstractFebObject(FebBaseObject, ABC):
    def __init__(self, 
                 tree: Union[ElementTree, None] = None, 
                 root: Union[Element, None] = None, 
                 filepath: Union[str, Path] = None):
        super().__init__(tree, root, filepath)
        
    # =========================================================================================================
    # Retrieve methods
    # =========================================================================================================
    
    # Main geometry data
    # ------------------------------
    
    @feb_instance_cache
    @abstractmethod
    def get_nodes(self, dtype: np.dtype = np.float32) -> List[Nodes]:
        print(f"get_nodes in AbstractFebObject")
        pass

    @feb_instance_cache
    @abstractmethod
    def get_elements(self, dtype: np.dtype = np.int64) -> List[Elements]:
        pass

    @feb_instance_cache
    @abstractmethod
    def get_surface_elements(self, dtype=np.int64) -> List[Elements]:
        pass

    @feb_instance_cache
    @abstractmethod
    def get_volume_elements(self, dtype=np.int64) -> List[Elements]:
        pass

    # Node, element, surface sets
    # ------------------------------

    @feb_instance_cache
    @abstractmethod
    def get_node_sets(self, dtype=np.int64) -> List[NodeSet]:
        """
        Returns a dict with keys representing node set names and values \
        representing corresponding node ids as a numpy array of specified dtype.\

        Args:
            dtype (np.dtype): Numpy dtype.

        Returns:
            list: [Nodeset(name, node_ids)]
        """
        pass
    
    @feb_instance_cache
    @abstractmethod
    def get_surface_sets(self, dtype=np.int64) -> List[SurfaceSet]:
        """
        Returns a dict with keys representing node set names and values \
        representing corresponding node ids as a numpy array of specified dtype.\

        Args:
            dtype (np.dtype): Numpy dtype.

        Returns:
            list: [SurfaceSet(name, node_ids)]
        """
        pass
    
    @feb_instance_cache
    @abstractmethod
    def get_element_sets(self, dtype=np.int64) -> List[ElementSet]:
        """
        Returns a dict with keys representing node set names and values \
        representing corresponding node ids as a numpy array of specified dtype.\

        Args:
            dtype (np.dtype): Numpy dtype.

        Returns:
            list: [ElementSet(name, node_ids)]
        """
        pass
    
    # Materials
    # ------------------------------
    
    @feb_instance_cache
    @abstractmethod
    def get_materials(self) -> List[Material]:
        pass

    # Loads
    # ------------------------------
    
    @feb_instance_cache
    @abstractmethod
    def get_nodal_loads(self) -> List[NodalLoad]:
        pass
    
    @feb_instance_cache
    @abstractmethod
    def get_surface_loads(self) -> List[SurfaceLoad]:
        pass
    
    @feb_instance_cache
    @abstractmethod
    def get_loadcurves(self, dtype=np.float32) -> List[LoadCurve]:
        pass
        
    # Boundary conditions
    # ------------------------------
    
    @feb_instance_cache
    @abstractmethod
    def get_boundary_conditions(self) -> List[Union[FixCondition, RigidBodyCondition, BoundaryCondition]]:
        pass
    
    # Mesh data
    # ------------------------------
    
    @feb_instance_cache
    @abstractmethod
    def get_nodal_data(self, dtype=np.float32) -> List[NodalData]:
        pass
    
    @feb_instance_cache
    @abstractmethod
    def get_surface_data(self, dtype=np.float32) -> List[SurfaceData]:
        pass
    
    @feb_instance_cache
    @abstractmethod
    def get_element_data(self, dtype=np.float32) -> List[ElementData]:
        pass

    # =========================================================================================================
    # Add methods
    # =========================================================================================================
    
    # Main geometry data
    # ------------------------------
    
    @abstractmethod
    def add_nodes(self, nodes: List[Nodes]) -> None:
        """
        Adds nodes to Geometry, appending to existing nodes if they share the same name.
        Automatically detects the highest node ID to ensure unique IDs for new nodes.

        Args:
            nodes (list of Nodes): List of Nodes namedtuples, each containing a name and coordinates.

        Raises:
            ValueError: If any node coordinates do not consist of three elements.
        """
        pass

    @abstractmethod
    def add_elements(self, elements: List[Elements]) -> None:
        """
        Adds elements to Geometry, appending to existing elements if they share the same name.
        Automatically detects the highest element ID to ensure unique IDs for new elements.

        Args:
            elements (list of Elements): List of Elements namedtuples, each containing name, material, type, and connectivity.

        Raises:
            ValueError: If any element connectivity does not meet expected format or length.
        """
        pass

    @abstractmethod
    def add_surface_elements(self, elements: List[Elements]) -> None:
        pass
    
    @abstractmethod
    def add_volume_elements(self, elements: List[Elements]) -> None:
        pass
    
    # Node, element, surface sets
    # ------------------------------
    
    @abstractmethod
    def add_node_sets(self, nodesets: List[NodeSet]) -> None:
        pass
    
    @abstractmethod
    def add_surface_sets(self, surfacesets: List[SurfaceSet]) -> None:
        pass
    
    @abstractmethod
    def add_element_sets(self, elementsets: List[ElementSet]) -> None:
        pass
    
    # Materials
    # ------------------------------
    
    @abstractmethod
    def add_materials(self, materials: List[Material]) -> None:
        pass
    
    # Loads
    # ------------------------------
    
    @abstractmethod
    def add_nodal_loads(self, nodal_loads: List[NodalLoad]) -> None:
        pass
    
    @abstractmethod
    def add_surface_loads(self, pressure_loads: List[SurfaceLoad]) -> None:
        pass
            
    @abstractmethod    
    def add_loadcurves(self, load_curves: List[LoadCurve]) -> None:
        pass
    
    # Boundary conditions
    # ------------------------------
    
    @abstractmethod
    def add_boundary_conditions(self, boundary_conditions: List[Union[FixCondition, RigidBodyCondition, BoundaryCondition]]) -> None:
        pass
    
    # Mesh data
    # ------------------------------
    
    @abstractmethod
    def add_nodal_data(self, nodal_data: List[NodalData]) -> None:
        pass
    
    @abstractmethod
    def add_surface_data(self, surface_data: List[SurfaceData]) -> None:
        pass
    
    @abstractmethod
    def add_element_data(self, element_data: List[ElementData]) -> None:
        pass

    # =========================================================================================================
    # Remove methods
    # =========================================================================================================
    
    # Main geometry data
    # ------------------------------
    
    @abstractmethod
    def remove_nodes(self, names: List[str]) -> None:
        pass
    
    @abstractmethod
    def remove_elements(self, names: List[str]) -> None:
        pass
    
    @abstractmethod    
    def remove_all_surface_elements(self) -> None:
        pass
    
    @abstractmethod
    def remove_all_volume_elements(self) -> None:
        pass
                
    # Node, element, surface sets
    # ------------------------------
    
    @abstractmethod
    def remove_node_sets(self, names: List[str]) -> None:
        pass
    
    @abstractmethod
    def remove_surface_sets(self, names: List[str]) -> None:
        pass
    
    @abstractmethod
    def remove_element_sets(self, names: List[str]) -> None:
        pass
    
    # Materials
    # ------------------------------
    
    @abstractmethod
    def remove_materials(self, ids: List[Union[str, int]]) -> None:
        pass
    
    # Loads
    # ------------------------------
    
    @abstractmethod
    def remove_nodal_loads(self, bc_or_node_sets: List[str]) -> None:
        """
        Removes nodal loads from Loads by boundary condition or node set.

        Args:
            bc_or_node_sets (list of str): List of boundary conditions or node sets to remove.
        """
        pass
    
    @abstractmethod
    def remove_surface_loads(self, surfaces: List[str]) -> None:
        """
        Removes pressure loads from Loads by surface.

        Args:
            surfaces (list of str): List of surfaces to remove.
        """
        pass
    
    @abstractmethod
    def remove_loadcurves(self, ids: List[int]) -> None:
        """
        Removes load curves from LoadData by ID.

        Args:
            ids (list of int): List of load curve IDs to remove.
        """
        pass
    
    # Boundary conditions
    # ------------------------------
    
    @abstractmethod
    def remove_boundary_conditions(self, types: List[str], bc: List[str]=None) -> None:
        """
        Removes boundary conditions from Boundary by type and optionally fiter type by boundary condition.
        e.g. remove_boundary_conditions(["fix"], ["BC1"]), instead of removing all fix conditions, only BC1 will be removed.
        
        Args:
            types (list of str): List of boundary condition types to remove.
            bc (list of str): List of boundary conditions to remove.
        """
        pass
        
    # Mesh data
    # ------------------------------
    
    @abstractmethod
    def remove_nodal_data(self, nodesets_or_names: List[str]) -> None:
        """
        Removes nodal data from MeshData by node_set or name.
        
        Args:
            nodesets_or_names (list of str): List of node sets or names to remove.
        """
        pass
        
    @abstractmethod
    def remove_surface_data(self, surfacesets_or_names: List[str]) -> None:
        """
        Removes surface data from MeshData by surf_set or name.
        
        Args:
            surfacesets_or_names (list of str): List of surface sets or names to remove.
        """
        pass
    
    @abstractmethod
    def remove_element_data(self, elementsets_or_names: List[str]) -> None:
        """
        Removes element data from MeshData by elem_set or name.
        
        Args:
            elementsets_or_names (list of str): List of element sets or names to remove.
        """
        pass
    
    # =========================================================================================================
    # Clear methods (remove all)
    # =========================================================================================================
    
    @abstractmethod
    def clear_nodes(self) -> None:
        """
        Removes all nodes from Geometry.
        """
        pass
    
    @abstractmethod
    def clear_elements(self) -> None:
        """
        Removes all elements from Geometry.
        """
        pass

    @abstractmethod
    def clear_surface_elements(self) -> None:
        """
        Removes all surface elements from Geometry.
        """
        pass
    
    @abstractmethod    
    def clear_volume_elements(self) -> None:
        """
        Removes all volume elements from Geometry.
        """
        pass
    
    @abstractmethod    
    def clear_node_sets(self) -> None:
        """
        Removes all node sets from Geometry.
        """
        pass
    
    @abstractmethod    
    def clear_surface_sets(self) -> None:
        """
        Removes all surface sets from Geometry.
        """
        pass
    
    @abstractmethod    
    def clear_element_sets(self) -> None:
        """
        Removes all element sets from Geometry.
        """
        pass
    
    @abstractmethod    
    def clear_materials(self) -> None:
        """
        Removes all materials from Material.
        """
        pass
    
    @abstractmethod    
    def clear_nodal_loads(self) -> None:
        """
        Removes all nodal loads from Loads.
        """
        pass
    
    @abstractmethod    
    def clear_surface_loads(self) -> None:
        """
        Removes all pressure loads from Loads.
        """
        pass
    
    @abstractmethod    
    def clear_loadcurves(self) -> None:
        """
        Removes all load curves from LoadData.
        """
        pass
    
    @abstractmethod    
    def clear_boundary_conditions(self) -> None:    
        """
        Removes all boundary conditions from Boundary.
        """
        pass
    
    @abstractmethod    
    def clear_nodal_data(self) -> None:
        """
        Removes all nodal data from MeshData.
        """
        pass
    
    @abstractmethod    
    def clear_surface_data(self) -> None:
        """
        Removes all surface data from MeshData.
        """
        pass
    
    @abstractmethod    
    def clear_element_data(self) -> None:
        """
        Removes all element data from MeshData.
        """
        pass
    
    def clear(self, 
              nodes=True,
              elements=True,
              surfaces=True,
              volumes=True,
              nodesets=True,
              surfacesets=True,
              elementsets=True,
              materials=True,
              nodal_loads=True,
              pressure_loads=True,
              loadcurves=True,
              boundary_conditions=True,
              nodal_data=True,
              surface_data=True,
              element_data=True) -> None:
        """
        Clears the FEBio model of all data, based on the specified options.
        
        Args:
            nodes (bool): Remove all nodes.
            elements (bool): Remove all elements.
            surfaces (bool): Remove all surface elements.
            volumes (bool): Remove all volume elements.
            nodesets (bool): Remove all node sets.
            surfacesets (bool): Remove all surface sets.
            elementsets (bool): Remove all element sets.
            materials (bool): Remove all materials.
            nodal_loads (bool): Remove all nodal loads.
            pressure_loads (bool): Remove all pressure loads.
            loadcurves (bool): Remove all load curves.
            boundary_conditions (bool): Remove all boundary conditions.
            nodal_data (bool): Remove all nodal data.
            surface_data (bool): Remove all surface data.
            element_data (bool): Remove all element data.
        """
        if nodes:
            self.clear_nodes()
        if elements:    
            self.clear_elements()
        if surfaces:
            self.clear_surface_elements()
        if volumes:
            self.clear_volume_elements()
        if nodesets:
            self.clear_node_sets()
        if surfacesets:
            self.clear_surface_sets()
        if elementsets:
            self.clear_element_sets()
        if materials:
            self.clear_materials()
        if nodal_loads:
            self.clear_nodal_loads()
        if pressure_loads:
            self.clear_surface_loads()
        if loadcurves:
            self.clear_loadcurves()
        if boundary_conditions:
            self.clear_boundary_conditions()
        if nodal_data:
            self.clear_nodal_data()
        if surface_data:
            self.clear_surface_data()
        if element_data:
            self.clear_element_data()

    # =========================================================================================================
    # Update methods
    # =========================================================================================================
    
    # Main geometry data
    # ------------------------------
    
    def update_nodes(self, nodes: List[Nodes]) -> None:
        """
        Updates nodes in Geometry by name, replacing existing nodes with the same name.

        Args:
            nodes (list of Nodes): List of Nodes namedtuples, each containing a name and coordinates.
        """
        self.remove_nodes([node.name for node in nodes])
        self.add_nodes(nodes)
    
    def update_elements(self, elements: List[Elements]) -> None:
        """
        Updates elements in Geometry by name, replacing existing elements with the same name.

        Args:
            elements (list of Elements): List of Elements namedtuples, each containing name, material, type, and connectivity.
        """
        self.remove_elements([element.name for element in elements])
        self.add_elements(elements)
    
    # Node, element, surface sets
    # ------------------------------
    
    def update_node_sets(self, nodesets: List[NodeSet]) -> None:
        """
        Updates node sets in Geometry by name, replacing existing node sets with the same name.

        Args:
            nodesets (list of NodeSet): List of NodeSet namedtuples, each containing a name and node IDs.
        """
        self.remove_node_sets([nodeset.name for nodeset in nodesets])
        self.add_node_sets(nodesets)
    
    def update_surface_sets(self, surfacesets: List[SurfaceSet]) -> None:
        """
        Updates surface sets in Geometry by name, replacing existing surface sets with the same name.

        Args:
            surfacesets (list of SurfaceSet): List of SurfaceSet namedtuples, each containing a name and node IDs.
        """
        self.remove_surface_sets([surfset.name for surfset in surfacesets])
        self.add_surface_sets(surfacesets)
    
    def update_element_sets(self, elementsets: List[ElementSet]) -> None:
        """
        Updates element sets in Geometry by name, replacing existing element sets with the same name.

        Args:
            elementsets (list of ElementSet): List of ElementSet namedtuples, each containing a name and element IDs.
        """
        self.remove_element_sets([elemset.name for elemset in elementsets])
        self.add_element_sets(elementsets)
    
    # Materials
    # ------------------------------
    
    def update_materials(self, materials: List[Material]) -> None:
        """
        Updates materials in Material by ID, replacing existing materials with the same ID.

        Args:
            materials (list of Material): List of Material namedtuples, each containing an ID, type, parameters, name, and attributes.
        """
        self.remove_materials([material.id for material in materials])
        self.add_materials(materials)
        
    # Loads
    # ------------------------------
    
    def update_nodal_loads(self, nodal_loads: List[NodalLoad]) -> None:
        """
        Updates nodal loads in Loads by node set, replacing existing nodal loads with the same node set.

        Args:
            nodal_loads (list of NodalLoad): List of NodalLoad namedtuples, each containing a boundary condition, node set, scale, and load curve.
        """
        self.remove_nodal_loads([load.node_set for load in nodal_loads])
        self.add_nodal_loads(nodal_loads)
    
    def update_surface_loads(self, pressure_loads: List[SurfaceLoad]) -> None:
        """
        Updates pressure loads in Loads by surface, replacing existing pressure loads with the same surface.

        Args:
            pressure_loads (list of SurfaceLoad): List of SurfaceLoad namedtuples, each containing a surface, attributes, and multiplier.
        """
        self.remove_surface_loads([load.surface for load in pressure_loads])
        self.add_surface_loads(pressure_loads)
    
    def update_loadcurves(self, load_curves: List[LoadCurve]) -> None:
        """
        Updates load curves in LoadData by ID, replacing existing load curves with the same ID.

        Args:
            load_curves (list of LoadCurve): List of LoadCurve namedtuples, each containing an ID, type, and data.
        """
        self.remove_loadcurves([curve.id for curve in load_curves])
        self.add_loadcurves(load_curves)

    # Boundary conditions
    # ------------------------------
    
    def update_boundary_conditions(self, boundary_conditions: List[Union[FixCondition, RigidBodyCondition, BoundaryCondition]]) -> None:
        """
        Updates boundary conditions in Boundary, replacing existing boundary conditions with the same type.

        Args:
            boundary_conditions (list of Union[FixCondition, RigidBodyCondition, BoundaryCondition]): List of boundary condition namedtuples.
        """
        bc_types = [bc.type for bc in boundary_conditions]
        bc_bcs = []
        for bc in boundary_conditions:
            if hasattr(bc, "bc"):
                bc_bcs.append(bc.bc)
            else:
                bc_bcs.append(None)
        self.remove_boundary_conditions(bc_types, bc_bcs)
        self.add_boundary_conditions(boundary_conditions)
    
    # Mesh data
    # ------------------------------
    
    def update_nodal_data(self, nodal_data: List[NodalData]) -> None:
        """
        Updates nodal data in MeshData by node set, replacing existing nodal data with the same node set.

        Args:
            nodal_data (list of NodalData): List of NodalData namedtuples, each containing a node set, name, and data.
        """
        self.remove_nodal_data([data.node_set for data in nodal_data])
        self.add_nodal_data(nodal_data)
    
    def update_surface_data(self, surface_data: List[SurfaceData]) -> None:
        """
        Updates surface data in MeshData by surface set, replacing existing surface data with the same surface set.

        Args:
            surface_data (list of SurfaceData): List of SurfaceData namedtuples, each containing a surface set, name, and data.
        """
        self.remove_surface_data([data.node_set for data in surface_data])
        self.add_surface_data(surface_data)
    
    def update_element_data(self, element_data: List[ElementData]) -> None:
        """
        Updates element data in MeshData by element set, replacing existing element data with the same element set.

        Args:
            element_data (list of ElementData): List of ElementData namedtuples, each containing an element set, name, and data.
        """
        self.remove_element_data([data.node_set for data in element_data])
        self.add_element_data(element_data)
