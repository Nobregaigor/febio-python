from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree
import xml.etree.ElementTree as ET

from .bases import AbstractFebObject
import numpy as np
from typing import Union, Dict, List
from collections import OrderedDict, deque

from febio_python.core import (
    SURFACE_EL_TYPE,
    Nodes,
    Elements,
    NodeSet,
    SurfaceSet,
    ElementSet,
    Material,
    NodalLoad,
    PressureLoad,
    LoadCurve,
    BoundaryCondition,
    FixCondition,
    # FixedAxis,
    RigidBodyCondition,
    NodalData,
    SurfaceData,
    ElementData,
    FEBioElementType,
    GenericDomain,
    ShellDomain,
)

from ._caching import feb_instance_cache

class Feb30(AbstractFebObject):
    def __init__(self, 
                 tree: Union[ElementTree, None] = None, 
                 root: Union[Element, None] = None, 
                 filepath: Union[str, Path] = None):
        super().__init__(tree, root, filepath)
        if self.version != 3.0:
            raise ValueError("This class is only for FEBio 3.0 files"
                             f"Version found: {self.version}")
        
    # =========================================================================================================
    # Retrieve methods
    # =========================================================================================================
    
    # Main geometry data
    # ------------------------------
    
    @feb_instance_cache
    def get_nodes(self, dtype: np.dtype = np.float32) -> List[Nodes]:
        all_nodes: OrderedDict = self.get_tag_data(self.LEAD_TAGS.MESH, self.MAJOR_TAGS.NODES, content_type="text",  dtype=dtype)
        listed_nodes = list()
        last_id = 0
        for key, value in all_nodes.items():
            this_ids = np.arange(last_id, last_id + value.shape[0], dtype=np.int64)
            listed_nodes.append(Nodes(name=key, coordinates=value, ids=this_ids))
            last_id += value.shape[0]
        return listed_nodes

    @feb_instance_cache
    def get_elements(self, dtype: np.dtype = np.int64) -> List[Elements]:
        all_elements = []
        # last_elem_id = 0 # Initialize the last element ID to 0
        for elem_group in self.mesh.findall("Elements"):
            elem_type = elem_group.attrib.get("type")
            elem_name = elem_group.attrib.get("name", None)
            elem_part = elem_group.attrib.get("part", None)
            elem_mat = elem_group.attrib.get("mat", None)
            
            connectivity = deque()
            elem_ids = deque()
            for elem in elem_group.findall("elem"):
                # Convert the comma-separated string of node indices into an array of integers
                this_elem_connectivity = np.array(elem.text.split(','), dtype=dtype)
                connectivity.append(this_elem_connectivity)
                this_elem_id = int(elem.attrib["id"])
                elem_ids.append(this_elem_id)
            
            # Convert the list of element connectivities to a numpy array
            connectivity = np.array(connectivity, dtype=dtype) -1 # Correct ids to zero-based indexing
            # num_elems = connectivity.shape[0]
            elem_ids = np.array(elem_ids, dtype=np.int64) -1 # Correct ids to zero-based indexing
            # Create an Elements instance for each element
            element = Elements(name=elem_name, 
                               mat=elem_mat,
                               part=elem_part, 
                               type=elem_type, 
                               connectivity=connectivity, 
                               ids=elem_ids)
            all_elements.append(element)
            # last_elem_id += num_elems

        return all_elements

    @feb_instance_cache
    def get_surface_elements(self, dtype=np.int64) -> List[Elements]:
        # get all elements
        all_elements = self.get_elements(dtype=dtype)
        # filter elements by surface type
        filtered = []
        for elem in all_elements:
            if elem.type in SURFACE_EL_TYPE.__members__:
                filtered.append(elem)
        return filtered

    @feb_instance_cache
    def get_volume_elements(self, dtype=np.int64) -> List[Elements]:
        # get all elements
        all_elements = self.get_elements(dtype=dtype)
        # filter elements by surface type
        filtered = []
        for elem in all_elements:
            if elem.type not in SURFACE_EL_TYPE.__members__:
                filtered.append(elem)
        return filtered

    # Node, element, surface sets
    # ------------------------------

    @feb_instance_cache
    def get_nodesets(self, dtype=np.int64) -> List[NodeSet]:
        """
        Returns a dict with keys representing node set names and values \
        representing corresponding node ids as a numpy array of specified dtype.\

        Args:
            dtype (np.dtype): Numpy dtype.

        Returns:
            list: [Nodeset(name, node_ids)]
        """
        # Extract the nodesets dictionary from the .feb file
        nodesets: dict = self.get_tag_data(self.LEAD_TAGS.MESH, self.MAJOR_TAGS.NODESET, content_type="id", dtype=dtype)
        # Convert the nodesets dictionary to a list of Nodeset named tuples
        nodeset_list = list()
        for key, value in nodesets.items():
            # correct value to zero-based indexing
            value -= 1
            nodeset_list.append(NodeSet(name=key, ids=value))
        return nodeset_list
    
    @feb_instance_cache
    def get_surfacesets(self, dtype=np.int64) -> List[SurfaceSet]:
        """
        Returns a dict with keys representing node set names and values \
        representing corresponding node ids as a numpy array of specified dtype.\

        Args:
            dtype (np.dtype): Numpy dtype.

        Returns:
            list: [SurfaceSet(name, node_ids)]
        """
        # Extract the surfacesets dictionary from the .feb file
        surfacesets: dict = self.get_tag_data(self.LEAD_TAGS.MESH, self.MAJOR_TAGS.SURFACESET, content_type="id", dtype=dtype)
        # Convert the surfacesets dictionary to a list of Nodeset named tuples
        surfaceset_list = list()
        for key, value in surfacesets.items():
            # correct value to zero-based indexing
            value -= 1
            surfaceset_list.append(SurfaceSet(name=key, node_ids=value))
        return surfaceset_list
    
    @feb_instance_cache
    def get_elementsets(self, dtype=np.int64) -> List[ElementSet]:
        """
        Returns a dict with keys representing node set names and values \
        representing corresponding node ids as a numpy array of specified dtype.\

        Args:
            dtype (np.dtype): Numpy dtype.

        Returns:
            list: [ElementSet(name, node_ids)]
        """
        # Extract the elementsets dictionary from the .feb file
        elementsets: dict = self.get_tag_data(self.LEAD_TAGS.MESH, self.MAJOR_TAGS.ELEMENTSET, content_type="id", dtype=dtype)
        # Convert the elementsets dictionary to a list of Nodeset named tuples
        elementset_list = list()
        for key, value in elementsets.items():
            # correct value to zero-based indexing
            value -= 1
            elementset_list.append(ElementSet(name=key, ids=value))
        return elementset_list
    
    # Mesh Domains
    # ------------------------------
    
    @feb_instance_cache
    def get_mesh_domains(self) -> List[Union[GenericDomain, ShellDomain]]:
        
        all_domains = []
        # get all child elements of MeshDomains
        for i, domain in enumerate(self.meshdomains.findall("./")):
            
            name = domain.attrib.get("name", f"UnnamedDomain{i}")
            mat = domain.attrib.get("mat", None)           
            
            if domain.tag.upper() == "SHELLDOMAIN":
                # get shell_normal_nodal child element
                shell_normal_nodal = float(domain.find("shell_normal_nodal").text)
                new_domain = ShellDomain(
                    id=i,
                    name=name,
                    mat=mat,
                    shell_normal_nodal=shell_normal_nodal
                )
            
            else:
                new_domain = GenericDomain(
                    id=i,
                    name=name,
                    mat=mat
                )
                
            all_domains.append(new_domain)

        return all_domains

    # Materials
    # ------------------------------

    @feb_instance_cache
    def get_materials(self) -> List[Material]:
        materials_list = []
        for item in self.material.findall(self.MAJOR_TAGS.MATERIAL.value):
            # Initialize the dictionary for attributes
            mat_attrib = dict(item.attrib)
            
            # Extract parameters and remove them from attributes to avoid duplication
            parameters = {}
            for el in list(item)[1:]:  # Assuming the first element describes the material itself
                try:
                    p_val = float(el.text)
                except ValueError:
                    p_val = el.text
                parameters[el.tag] = p_val

            # Remove standard fields from attributes if they exist
            mat_id = mat_attrib.pop("id", None)
            try:
                mat_id = int(mat_id)
            except ValueError:
                pass
            mat_type = mat_attrib.pop("type", None)
            mat_name = mat_attrib.pop("name", "Unnamed Material")

            # Create a Material named tuple for the current material
            current_material = Material(id=mat_id, type=mat_type, parameters=parameters, name=mat_name, attributes=mat_attrib)

            # Append the created Material to the list
            materials_list.append(current_material)

        return materials_list

    # Loads
    # ------------------------------
    
    @feb_instance_cache
    def get_nodal_loads(self) -> List[NodalLoad]:
        nodal_loads = []
        for i, load in enumerate(self.loads.findall(self.MAJOR_TAGS.NODALLOAD.value)):
            scale_data = load.find("scale")
            
            # Convert scale text to float if possible, maintain as text if not
            try:
                scale_value = float(scale_data.text)
            except ValueError:
                scale_value = scale_data.text  # Keep as text if not convertible

            # Extracting the load curve id, default to 'NoCurve'
            lc_curve = scale_data.attrib.get("lc", "NoCurve")
            try:
                lc_curve = int(lc_curve)
            except ValueError:
                pass

            # Create a NodalLoad named tuple for the current load
            current_load = NodalLoad(
                dof=load.find("dof").text if load.find("dof") is not None else "UndefinedBC",  # Handling missing dof element
                node_set=load.attrib.get("node_set", f"UnnamedNodeSet{i}"),  # Default to an indexed name if not specified
                scale=scale_value,
                load_curve=lc_curve
            )

            # Add the new NodalLoad to the list
            nodal_loads.append(current_load)

        return nodal_loads
    
    @feb_instance_cache
    def get_pressure_loads(self) -> List[PressureLoad]:
        pressure_loads_list = []
        for i, load in enumerate(self.loads.findall(self.MAJOR_TAGS.SURFACELOAD.value)):
            press = load.find("pressure")
            if press is not None:
                load_info = load.attrib
                press_info = press.attrib

                # Extract the pressure multiplier, handling possible non-numeric values
                try:
                    press_mult = float(press.text)
                except ValueError:
                    press_mult = press.text  # Keep as text if not convertible

                # Create a PressureLoad named tuple for the current load
                current_load = PressureLoad(
                    surface=load_info.get("surface", f"UnnamedSurface{i}"),  # Default to index if no surface name
                    attributes=press_info,
                    multiplier=press_mult
                )

                # Append the created PressureLoad to the list
                pressure_loads_list.append(current_load)

        return pressure_loads_list
    
    @feb_instance_cache
    def get_loadcurves(self, dtype=np.float32) -> List[LoadCurve]:
        load_curves_list = []
        for loadcurve_elem in self.loaddata.findall(".//load_controller"):
            load_curve_id = loadcurve_elem.attrib['id']
            try:
                load_curve_id = int(load_curve_id)
            except ValueError:
                pass
            load_curve_type = loadcurve_elem.find('interpolate').text.lower()  # Adapting to 'interpolate' tag for curve type

            points = deque()

            # Extract points from each 'point' element within 'points' container
            for point_elem in loadcurve_elem.find('points').findall('point'):
                # Split the point text by ',' and convert to float
                point = tuple(map(float, point_elem.text.split(',')))
                points.append(point)

            # Convert list of points to a numpy array of the specified dtype
            points_array = np.array(points, dtype=dtype)

            # Create a LoadCurve instance
            current_load_curve = LoadCurve(id=load_curve_id, interpolate_type=load_curve_type, data=points_array)
            load_curves_list.append(current_load_curve)

        return load_curves_list
        
    # Boundary conditions
    # ------------------------------
    
    @feb_instance_cache
    def get_boundary_conditions(self) -> List[Union[FixCondition, RigidBodyCondition, BoundaryCondition]]:
        if self.boundary is None:
            return []
        
        boundary_conditions_list = []
        for elem in self.boundary.findall(".//bc"):  # Update to find 'bc' elements
            if elem.attrib.get('type') == 'fix':
                # Create an instance of FixCondition for each 'fix' element
                fix_condition = FixCondition(
                    dof=elem.find("dofs").text if elem.find("dofs") is not None else "UndefinedDOFs", 
                    node_set=elem.attrib['node_set'], 
                    name=elem.attrib['name'] if 'name' in elem.attrib else None
                )
                boundary_conditions_list.append(fix_condition)
            
            elif elem.tag == 'rigid_body':  # Assuming rigid_body handling remains similar
                # Gather all 'fixed' sub-elements for a 'rigid_body'
                fixed_axes = [fixed.attrib['bc'] for fixed in elem.findall('fixed')]
                fixed_axes = ",".join(fixed_axes) if fixed_axes else ""
                # Create an instance of RigidBodyCondition for each 'rigid_body' element
                rigid_body_condition = RigidBodyCondition(material=elem.attrib['mat'], dof=fixed_axes)
                boundary_conditions_list.append(rigid_body_condition)
            
            else:
                # Fallback to a generic BoundaryCondition for unrecognized types
                generic_condition = BoundaryCondition(type=elem.tag, attributes=elem.attrib)
                boundary_conditions_list.append(generic_condition)

        return boundary_conditions_list
    
    # Mesh data
    # ------------------------------
    
    @feb_instance_cache
    def get_nodal_data(self, dtype=np.float32) -> List[NodalData]:
        nodal_data_list = []
        for data in self.meshdata.findall(self.MAJOR_TAGS.NODEDATA.value):
            _this_data = deque()
            _these_ids = deque()
            for x in data.findall("node"):
                if ',' in x.text:
                    # Split the string by commas and convert each to float
                    _this_data.append([float(num) for num in x.text.split(',')])
                elif x.text.isdigit():
                    # Convert single digit strings to float
                    _this_data.append(float(x.text))
                else:
                    # Add non-numeric strings as is
                    _this_data.append(x.text)
                _these_ids.append(int(x.attrib["lid"]))
            ref = data.attrib["node_set"]
            name = data.attrib["name"]
            
            # Create a NodalData instance
            current_data = NodalData(
                node_set=ref,
                name=name,
                data=np.array(_this_data, dtype=dtype),  # Ensure data is in the correct dtype
                ids=np.array(_these_ids, dtype=np.int64) - 1 # Correct ids to zero-based indexing
            )

            # Add the NodalData instance to the list
            nodal_data_list.append(current_data)

        return nodal_data_list
    
    @feb_instance_cache
    def get_surface_data(self, dtype=np.float32) -> List[SurfaceData]:
        surf_data_list = []
        for data in self.meshdata.findall(self.MAJOR_TAGS.SURFACE_DATA.value):
            _this_data = deque()
            _these_ids = deque()
            for x in data.findall("surf"):
                if ',' in x.text:
                    # Split the string by commas and convert each to float
                    _this_data.append([float(num) for num in x.text.split(',')])
                elif x.text.isdigit():
                    # Convert single digit strings to float
                    _this_data.append(float(x.text))
                else:
                    # Add non-numeric strings as is
                    _this_data.append(x.text)
                _these_ids.append(int(x.attrib["lid"]))
            ref = data.attrib["surf_set"]
            name = data.attrib["name"]
            
            # Create a NodalData instance
            current_data = SurfaceData(
                surf_set=ref,
                name=name,
                data=np.array(_this_data, dtype=dtype),  # Ensure data is in the correct dtype
                ids=np.array(_these_ids, dtype=np.int64) - 1 # Correct ids to zero-based indexing
            )

            # Add the NodalData instance to the list
            surf_data_list.append(current_data)

        return surf_data_list
    
    @feb_instance_cache
    def get_element_data(self, dtype=np.float32) -> List[ElementData]:
        elem_data_list = []
        for data in self.meshdata.findall(self.MAJOR_TAGS.ELEMENTDATA.value):
            _this_data = deque()
            _these_ids = deque()
            for x in data.findall("elem"):
                if ',' in x.text:
                    # Split the string by commas and convert each to float
                    _this_data.append([float(num) for num in x.text.split(',')])
                elif x.text.isdigit():
                    # Convert single digit strings to float
                    _this_data.append(float(x.text))
                else:
                    # Add non-numeric strings as is
                    _this_data.append(x.text)
                _these_ids.append(int(x.attrib["lid"]))
            
            ref = data.attrib["elem_set"]
            name = data.attrib.get("name", None)
            var = data.attrib.get("var", None)   
            
            # Create a ElementData instance
            current_data = ElementData(
                elem_set=ref,
                name=name,
                var=var,
                data=np.array(_this_data, dtype=dtype),  # Ensure data is in the correct dtype
                ids=np.array(_these_ids, dtype=np.int64) -1 # Correct ids to zero-based indexing
            )

            # Add the ElementData instance to the list
            elem_data_list.append(current_data)

        return elem_data_list

    # =========================================================================================================
    # Add methods
    # =========================================================================================================
    
    # Main geometry data
    # ------------------------------
    
    def add_nodes(self, nodes: List[Nodes]) -> None:
        """
        Adds nodes to Geometry, appending to existing nodes if they share the same name.
        Automatically detects the highest node ID to ensure unique IDs for new nodes.

        Args:
            nodes (list of Nodes): List of Nodes namedtuples, each containing a name and coordinates.

        Raises:
            ValueError: If any node coordinates do not consist of three elements.
        """
        # Retrieve existing nodes and determine the last node ID
        existing_nodes_list = self.get_nodes()
        last_initial_id = existing_nodes_list[-1].ids[-1] if existing_nodes_list else 1 # when adding nodes, FEBio starts from 1

        existing_nodes = {node_elem.name: node_elem for node_elem in existing_nodes_list}

        for node in nodes:
            if node.name in existing_nodes:
                # Append to existing Nodes element
                el_root = self.mesh.find(f".//Nodes[@name='{node.name}']")
            else:
                # Create a new Nodes element if no existing one matches the name
                el_root = ET.Element("Nodes")
                el_root.set("name", node.name)
                self.mesh.append(el_root)  # Append new "Nodes" at the end of the geometry

            for i, node_xyz in enumerate(node.coordinates):
                if len(node_xyz) != 3:  # Ensure each node has exactly three coordinates
                    raise ValueError(f"Node {i + last_initial_id} does not have the correct number of coordinates. It should contain [x, y, z] values.")
                subel = ET.SubElement(el_root, "node")
                subel.set("id", str(i + last_initial_id))
                subel.text = ",".join(map(str, node_xyz))  # Convert coordinates to comma-separated string

            # Update the last_initial_id for the next node group
            last_initial_id += len(node.coordinates)

    def add_elements(self, elements: List[Elements]) -> None:
        """
        Adds elements to Geometry, appending to existing elements if they share the same name.
        Automatically detects the highest element ID to ensure unique IDs for new elements.

        Args:
            elements (list of Elements): List of Elements namedtuples, each containing name, material, type, and connectivity.

        Raises:
            ValueError: If any element connectivity does not meet expected format or length.
        """
        # Retrieve existing elements and determine the last element ID
        existing_elements_list = self.get_elements()
        last_initial_id = existing_elements_list[-1].ids[-1] if existing_elements_list else 1  # when adding elements, FEBio starts from 1

        existing_elements = {element.name: element for element in existing_elements_list}

        for element in elements:
            if element.name in existing_elements:
                # Append to existing Elements group
                el_root = self.mesh.find(f".//Elements[@name='{element.name}']")
            else:
                # Make sure the element type is valid, it must be a valid FEBio element type
                # However, user can also use VTK element types as input, but they must be 
                # converted to FEBio types
                el_type = element.type
                if el_type not in FEBioElementType.__members__:
                    try:
                        el_type = FEBioElementType[str(el_type)]
                    except KeyError:
                        raise ValueError(f"Element type {el_type} is not a valid FEBio element type.")                        
                        
                # Create a new Elements group if no existing one matches the name
                el_root = ET.Element("Elements")
                el_root.set("name", element.name)
                el_root.set("type", element.type)
                el_root.set("mat", element.mat)
                self.mesh.append(el_root)  # Append new "Elements" at the end of the geometry
            for i, connectivity in enumerate(element.connectivity):
                subel = ET.SubElement(el_root, "elem")
                subel.set("id", str(i + last_initial_id))
                subel.text = ",".join(map(str, connectivity))  # Convert connectivity to comma-separated string

            # Update the last_initial_id for the next element group
            last_initial_id += len(element.connectivity)

    def add_surface_elements(self, elements: List[Elements]) -> None:
        # Filter elements by surface type
        filtered = [elem for elem in elements if elem.type in SURFACE_EL_TYPE.__members__]
        if len(filtered) == 0:
            raise ValueError("No surface elements found in the input list. Try using add_elements() instead.")
        self.add_elements(filtered)
        
    def add_volume_elements(self, elements: List[Elements]) -> None:
        # Filter elements by volume type
        filtered = [elem for elem in elements if elem.type not in SURFACE_EL_TYPE.__members__]
        if len(filtered) == 0:
            raise ValueError("No volume elements found in the input list. Try using add_elements() instead.")
        self.add_elements(filtered)
    
    # Node, element, surface sets
    # ------------------------------
    
    def add_nodesets(self, nodesets: List[NodeSet]) -> None:
        """
        Adds node sets to Geometry, appending to existing node sets if they share the same name.

        Args:
            nodesets (list of NodeSet): List of NodeSet namedtuples, each containing a name and node IDs.
        """
        existing_nodesets = {nodeset.name: nodeset for nodeset in self.get_nodesets()}

        for nodeset in nodesets:
            if nodeset.name in existing_nodesets:
                # Append to existing NodeSet element
                el_root = self.mesh.find(f".//NodeSet[@name='{nodeset.name}']")
            else:
                # Create a new NodeSet element if no existing one matches the name
                el_root = ET.Element("NodeSet")
                el_root.set("name", nodeset.name)
                self.mesh.append(el_root)

            # Add node IDs as sub-elements
            for node_id in nodeset.ids:
                subel = ET.SubElement(el_root, "node")
                subel.set("id", str(int(node_id + 1)))  # Convert to one-based indexing
    
    def add_surfacesets(self, surfacesets: List[SurfaceSet]) -> None:
        """
        Adds surface sets to Geometry, appending to existing surface sets if they share the same name.

        Args:
            surfacesets (list of SurfaceSet): List of SurfaceSet namedtuples, each containing a name and node IDs.
        """
        existing_surfacesets = {surfset.name: surfset for surfset in self.get_surfacesets()}

        for surfset in surfacesets:
            if surfset.name in existing_surfacesets:
                # Append to existing SurfaceSet element
                el_root = self.mesh.find(f".//SurfaceSet[@name='{surfset.name}']")
            else:
                # Create a new SurfaceSet element if no existing one matches the name
                el_root = ET.Element("SurfaceSet")
                el_root.set("name", surfset.name)
                self.mesh.append(el_root)
                
            # Add node IDs as sub-elements
            for node_id in surfset.node_ids:
                subel = ET.SubElement(el_root, "node")
                subel.set("id", str(int(node_id + 1))) # Convert to one-based indexing
    
    def add_elementsets(self, elementsets: List[ElementSet]) -> None:
        """
        Adds element sets to Geometry, appending to existing element sets if they share the same name.

        Args:
            elementsets (list of ElementSet): List of ElementSet namedtuples, each containing a name and element IDs.
        """
        existing_elementsets = {elemset.name: elemset for elemset in self.get_elementsets()}

        for elemset in elementsets:
            if elemset.name in existing_elementsets:
                # Append to existing ElementSet element
                el_root = self.mesh.find(f".//ElementSet[@name='{elemset.name}']")
            else:
                # Create a new ElementSet element if no existing one matches the name
                el_root = ET.Element("ElementSet")
                el_root.set("name", elemset.name)
                self.mesh.append(el_root)
                
            # Add element IDs as sub-elements
            for elem_id in elemset.ids:
                subel = ET.SubElement(el_root, "elem")
                subel.set("id", str(int(elem_id + 1))) # Convert to one-based indexing
    
    # Materials
    # ------------------------------
    
    def add_materials(self, materials: List[Material]) -> None:
        """
        Adds materials to Material, appending to existing materials if they share the same ID.

        Args:
            materials (list of Material): List of Material namedtuples, each containing an ID, type, parameters, name, and attributes.
        """
        existing_materials = {material.id: material for material in self.get_materials()}

        for material in materials:
            if material.id in existing_materials or str(material.id) in existing_materials:
                print(f"Material with ID {material.id} already exists")
                # Append to existing Material element
                el_root = self.material.find(f".//material[@id='{material.id}']")
            else:
                # Create a new Material element if no existing one matches the ID
                el_root = ET.Element("material")
                el_root.set("id", material.id)
                el_root.set("type", material.type)
                el_root.set("name", material.name)
                self.material.append(el_root)

            # Add parameters as sub-elements
            for key, value in material.parameters.items():
                subel = ET.SubElement(el_root, key)
                subel.text = str(value)
    
    # Loads
    # ------------------------------
    
    def add_nodal_loads(self, nodal_loads: List[NodalLoad]) -> None:
        """
        Adds nodal loads to Loads, appending to existing nodal loads if they share the same node set.

        Args:
            nodal_loads (list of NodalLoad): List of NodalLoad namedtuples, each containing a boundary condition, node set, scale, and load curve.
        """
        existing_nodal_loads = {load.node_set: load for load in self.get_nodal_loads()}

        for load in nodal_loads:
            if load.node_set in existing_nodal_loads:
                # Append to existing NodalLoad element
                el_root = self.loads.find(f".//nodal_load[@node_set='{load.node_set}']")
            else:
                # Create a new NodalLoad element if no existing one matches the node set
                el_root = ET.Element("nodal_load")
                el_root.set("node_set", load.node_set)
                self.loads.append(el_root)

            # Setting the type attribute which is required in new version
            el_root.set("type", "nodal_load")
            # Setting the dof
            dof_element = ET.SubElement(el_root, "dof")
            dof_element.text = load.dof
            # Set the scale and load curve
            scale_element = ET.SubElement(el_root, "scale")
            scale_element.text = str(load.scale)
            scale_element.set("lc", str(load.load_curve))
    
    def add_pressure_loads(self, pressure_loads: List[PressureLoad]) -> None:
        """
        Adds pressure loads to Loads, appending to existing pressure loads if they share the same surface.

        Args:
            pressure_loads (list of PressureLoad): List of PressureLoad namedtuples, each containing a surface, attributes, and multiplier.
        """
        existing_pressure_loads = {load.surface: load for load in self.get_pressure_loads()}

        for load in pressure_loads:
            if load.surface in existing_pressure_loads:
                # Append to existing PressureLoad element
                el_root = self.loads.find(f".//surface_load[@surface='{load.surface}']")
            else:
                # Create a new PressureLoad element if no existing one matches the surface
                el_root = ET.Element("surface_load")
                el_root.set("surface", load.surface)
                self.loads.append(el_root)

            el_root.text = str(load.multiplier)
            for key, value in load.attributes.items():
                el_root.set(key, str(value))
                
    def add_loadcurves(self, load_curves: List[LoadCurve]) -> None:
        """
        Adds load curves to LoadData, appending to existing load curves if they share the same ID.

        Args:
            load_curves (list of LoadCurve): List of LoadCurve namedtuples, each containing an ID, type, and data.
        """
        existing_load_curves = {curve.id: curve for curve in self.get_loadcurves()}

        for curve in load_curves:
            if curve.id in existing_load_curves:
                # Find existing LoadCurve element
                el_root = self.loaddata.find(f".//load_controller[@id='{curve.id}']")
            else:
                # Create a new LoadCurve element if no existing one matches the ID
                el_root = ET.Element("load_controller")
                el_root.set("id", str(curve.id))
                self.loaddata.append(el_root)

            # Set type as 'loadcurve' and add 'interpolate' element
            el_root.set("type", "loadcurve")
            interpolate_elem = ET.SubElement(el_root, "interpolate")
            interpolate_elem.text = curve.interpolate_type.upper()  # Ensure the correct case

            # Create a 'points' container element
            points_elem = ET.SubElement(el_root, "points")
            for point in curve.data:
                point_elem = ET.SubElement(points_elem, "point")
                point_elem.text = ",".join(map(str, point))
    
    # Boundary conditions
    # ------------------------------
    
    def add_boundary_conditions(self, boundary_conditions: List[Union[FixCondition, RigidBodyCondition, BoundaryCondition]]) -> None:
        """
        Adds boundary conditions to Boundary, appending to existing boundary conditions if they share the same type and node set.

        Args:
            boundary_conditions (list of Union[FixCondition, RigidBodyCondition, BoundaryCondition]): List of boundary condition namedtuples.
        """
        for bc in boundary_conditions:
            # Find or create a new 'bc' element based on type and name
            el_root = self.boundary.find(f".//bc[@name='{bc.name}']")
            if el_root is None:
                el_root = ET.Element("bc")
                el_root.set("type", bc.type)
                el_root.set("name", bc.name)
                self.boundary.append(el_root)

            # Set 'node_set'
            el_root.set("node_set", bc.node_set)

            # Handle different types of boundary conditions
            if isinstance(bc, FixCondition):
                dof_elem = ET.SubElement(el_root, "dofs")
                dof_elem.text = bc.dof
            elif isinstance(bc, RigidBodyCondition):
                for fixed in bc.dof.split(','):
                    subel = ET.SubElement(el_root, "fixed")
                    subel.set("bc", fixed)
            else:
                for key, value in bc.attributes.items():
                    el_root.set(key, value)
    
    # Mesh data
    # ------------------------------
       
    def add_nodal_data(self, nodal_data: List[NodalData]) -> None:
        """
        Adds nodal data to MeshData, appending to existing nodal data if they share the same node set.

        Args:
            nodal_data (list of NodalData): List of NodalData namedtuples, each containing a node set, name, and data.
        """
        existing_nodal_data = {data.node_set: data for data in self.get_nodal_data()}

        for data in nodal_data:
            if data.node_set in existing_nodal_data:
                # Append to existing NodalData element
                el_root = self.meshdata.find(f".//{self.MAJOR_TAGS.NODEDATA.value}[@node_set='{data.node_set}']")
            else:
                # Create a new NodalData element if no existing one matches the node set
                el_root = ET.Element(self.MAJOR_TAGS.NODEDATA.value)
                el_root.set("node_set", data.node_set)
                el_root.set("name", data.name)
                self.meshdata.append(el_root)

            # Add node IDs and data as sub-elements
            for i, node_data in enumerate(data.data):
                subel = ET.SubElement(el_root, "node")
                subel.set("lid", str(data.ids[i] + 1))  # Convert to one-based indexing
                subel.text = ",".join(map(str, node_data))
    
    def add_surface_data(self, surface_data: List[SurfaceData]) -> None:
        """
        Adds surface data to MeshData, appending to existing surface data if they share the same surface set.

        Args:
            surface_data (list of SurfaceData): List of SurfaceData namedtuples, each containing a surface set, name, and data.
        """
        existing_surface_data = {data.node_set: data for data in self.get_surface_data()}

        for data in surface_data:
            if data.node_set in existing_surface_data:
                # Append to existing SurfaceData element
                el_root = self.meshdata.find(f".//{self.MAJOR_TAGS.SURFACE_DATA.value}[@surf_set='{data.node_set}']")
            else:
                # Create a new SurfaceData element if no existing one matches the surface set
                el_root = ET.Element(self.MAJOR_TAGS.SURFACE_DATA.value)
                el_root.set("surf_set", data.node_set)
                el_root.set("name", data.name)
                self.meshdata.append(el_root)

            for i, surf_data in enumerate(data.data):
                subel = ET.SubElement(el_root, "surf")
                subel.set("lid", str(data.ids[i] + 1))  # Convert to one-based indexing
                subel.text = ",".join(map(str, surf_data))
    
    def add_element_data(self, element_data: List[ElementData]) -> None:
        """
        Adds element data to MeshData, appending to existing element data if they share the same element set.

        Args:
            element_data (list of ElementData): List of ElementData namedtuples, each containing an element set, name, and data.
        """
        existing_element_data = {data.node_set: data for data in self.get_element_data()}

        for data in element_data:
            if data.node_set in existing_element_data:
                # Append to existing ElementData element
                el_root = self.meshdata.find(f".//{self.MAJOR_TAGS.ELEMENTDATA.value}[@elem_set='{data.node_set}']")
            else:
                # Create a new ElementData element if no existing one matches the element set
                el_root = ET.Element(self.MAJOR_TAGS.ELEMENTDATA.value)
                el_root.set("elem_set", data.node_set)
                if data.name is None and data.var is None:
                    raise ValueError("ElementData must have either a name or var attribute.")
                if data.name is not None:
                    el_root.set("name", data.name)
                if data.var is not None:
                    el_root.set("var", data.var)
                self.meshdata.append(el_root)

            for i, elem_data in enumerate(data.data):
                subel = ET.SubElement(el_root, "elem")
                subel.set("lid", str(data.ids[i] + 1)) # Convert to one-based indexing
                subel.text = ",".join(map(str, elem_data))

    # =========================================================================================================
    # Remove methods
    # =========================================================================================================
    
    # Main geometry data
    # ------------------------------
    
    def remove_nodes(self, names: List[str]) -> None:
        """
        Removes nodes from Geometry by name.

        Args:
            names (list of str): List of node names to remove.
        """
        for name in names:
            el = self.mesh.find(f".//Nodes[@name='{name}']")
            if el is not None:
                self.mesh.remove(el)
    
    def remove_elements(self, names: List[str]) -> None:
        """
        Removes elements from Geometry by name.

        Args:
            names (list of str): List of element names to remove.
        """
        for name in names:
            el = self.mesh.find(f".//Elements[@name='{name}']")
            if el is not None:
                self.mesh.remove(el)
                
    def remove_all_surface_elements(self) -> None:
        """
        Removes all surface elements from Geometry.
        """
        for el in self.mesh.findall("Elements"):
            if el.attrib.get("type") in SURFACE_EL_TYPE.__members__:
                self.mesh.remove(el)
    
    def remove_all_volume_elements(self) -> None:
        """
        Removes all volume elements from Geometry.
        """
        for el in self.mesh.findall("Elements"):
            if el.attrib.get("type") not in SURFACE_EL_TYPE.__members__:
                self.mesh.remove(el)
                
    # Node, element, surface sets
    # ------------------------------
    
    def remove_nodesets(self, names: List[str]) -> None:
        """
        Removes node sets from Geometry by name.

        Args:
            names (list of str): List of node set names to remove.
        """
        for name in names:
            el = self.mesh.find(f".//NodeSet[@name='{name}']")
            if el is not None:
                self.mesh.remove(el)
    
    def remove_surfacesets(self, names: List[str]) -> None:
        """
        Removes surface sets from Geometry by name.

        Args:
            names (list of str): List of surface set names to remove.
        """
        for name in names:
            el = self.mesh.find(f".//SurfaceSet[@name='{name}']")
            if el is not None:
                self.mesh.remove(el)
    
    def remove_elementsets(self, names: List[str]) -> None:
        """
        Removes element sets from Geometry by name.

        Args:
            names (list of str): List of element set names to remove.
        """
        for name in names:
            el = self.mesh.find(f".//ElementSet[@name='{name}']")
            if el is not None:
                self.mesh.remove(el)
    
    # Materials
    # ------------------------------
    
    def remove_materials(self, ids: List[Union[str, int]]) -> None:
        """
        Removes materials from Material by ID, name or type.

        Args:
            ids (list of int): List of material IDs, types, remove.
        """
        for id in ids:
            el = self.material.find(f".//material[@id='{id}']")
            if el is not None:
                self.material.remove(el)
                continue
            el = self.material.find(f".//material[@name='{id}']")
            if el is not None:
                self.material.remove(el)
                continue
            el = self.material.find(f".//material[@type='{id}']")
            if el is not None:
                self.material.remove(el)
                continue
    
    # Loads
    # ------------------------------
    
    def remove_nodal_loads(self, name_or_dof_or_node_sets: List[str]) -> None:
        """
        Removes nodal loads from Loads by name, degrees of freedom, or node set.

        Args:
            name_or_dof_or_node_sets (list of str): List of names, degrees of freedom, or node sets to remove.
        """
        for identifier in name_or_dof_or_node_sets:
            # Remove by name attribute
            els_by_name = self.loads.findall(f".//nodal_load[@name='{identifier}']")
            for el in els_by_name:
                self.loads.remove(el)

            # Remove by node_set attribute
            els_by_node_set = self.loads.findall(f".//nodal_load[@node_set='{identifier}']")
            for el in els_by_node_set:
                self.loads.remove(el)

            # Remove by degrees of freedom (dof)
            els_by_dof = self.loads.findall(".//nodal_load[dof]")
            for el in els_by_dof:
                if el.find("dof").text == identifier:
                    self.loads.remove(el)
    
    def remove_pressure_loads(self, surfaces: List[str]) -> None:
        """
        Removes pressure loads from Loads by surface.

        Args:
            surfaces (list of str): List of surfaces to remove.
        """
        for surf in surfaces:
            el = self.loads.find(f".//surface_load[@surface='{surf}']")
            if el is not None:
                self.loads.remove(el)
                
    def remove_loadcurves(self, ids: List[int]) -> None:
        """
        Removes load curves from LoadData by ID.

        Args:
            ids (list of int): List of load curve IDs to remove.
        """
        for id in ids:
            el = self.loaddata.find(f".//loadcurve[@id='{id}']")
            if el is not None:
                self.loaddata.remove(el)
    
    # Boundary conditions
    # ------------------------------
    
    def remove_boundary_conditions(self, types: List[str], names: List[str]=None) -> None:
        """
        Removes boundary conditions from the Boundary section by type and optionally filters type by name.
        e.g., remove_boundary_conditions(["fix"], ["FixedDisplacement01"]), instead of removing all fix conditions, only "FixedDisplacement01" will be removed.
        
        Args:
            types (list of str): List of boundary condition types to remove.
            names (list of str): Optional list of boundary condition names to specifically target for removal.
        """
        for _type in types:
            if names is None:
                # Remove all boundary conditions of a specific type
                elements = self.boundary.findall(f".//bc[@type='{_type}']")
                for el in elements:
                    self.boundary.remove(el)
            else:
                for name in names:
                    # Find boundary condition by type and name
                    el = self.boundary.find(f".//bc[@type='{_type}'][@name='{name}']")
                    if el is not None:
                        self.boundary.remove(el)
        
    # Mesh data
    # ------------------------------
    
    def remove_nodal_data(self, nodesets_or_names: List[str]) -> None:
        """
        Removes nodal data from MeshData by node_set or name.
        
        Args:
            nodesets_or_names (list of str): List of node sets or names to remove.
        """
        
        for ns in nodesets_or_names:
            el = self.meshdata.find(f".//{self.MAJOR_TAGS.NODEDATA.value}[@node_set='{ns}']")
            if el is not None:
                self.meshdata.remove(el)
                continue
            el = self.meshdata.find(f".//{self.MAJOR_TAGS.NODEDATA.value}[@name='{ns}']")
            if el is not None:
                self.meshdata.remove(el)
                continue
        
    def remove_surface_data(self, surfacesets_or_names: List[str]) -> None:
        """
        Removes surface data from MeshData by surf_set or name.
        
        Args:
            surfacesets_or_names (list of str): List of surface sets or names to remove.
        """
        for ss in surfacesets_or_names:
            el = self.meshdata.find(f".//{self.MAJOR_TAGS.SURFACE_DATA.value}[@surf_set='{ss}']")
            if el is not None:
                self.meshdata.remove(el)
                continue
            el = self.meshdata.find(f".//{self.MAJOR_TAGS.SURFACE_DATA.value}[@name='{ss}']")
            if el is not None:
                self.meshdata.remove(el)
                continue
    
    def remove_element_data(self, elementsets_or_names: List[str]) -> None:
        """
        Removes element data from MeshData by elem_set or name.
        
        Args:
            elementsets_or_names (list of str): List of element sets or names to remove.
        """
        for es in elementsets_or_names:
            el = self.meshdata.find(f".//{self.MAJOR_TAGS.ELEMENTDATA.value}[@elem_set='{es}']")
            if el is not None:
                self.meshdata.remove(el)
                continue
            el = self.meshdata.find(f".//{self.MAJOR_TAGS.ELEMENTDATA.value}[@name='{es}']")
            if el is not None:
                self.meshdata.remove(el)
                continue
    
    # =========================================================================================================
    # Clear methods (remove all)
    # =========================================================================================================
    
    def clear_nodes(self) -> None:
        """
        Removes all nodes from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.NODES.value):
            self.mesh.remove(el)
    
    def clear_elements(self) -> None:
        """
        Removes all elements from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.ELEMENTS.value):
            self.mesh.remove(el)

    def clear_surface_elements(self) -> None:
        """
        Removes all surface elements from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.ELEMENTS.value):
            if el.attrib.get("type") in SURFACE_EL_TYPE.__members__:
                self.mesh.remove(el)
    
    def clear_volume_elements(self) -> None:
        """
        Removes all volume elements from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.ELEMENTS.value):
            if el.attrib.get("type") not in SURFACE_EL_TYPE.__members__:
                self.mesh.remove(el)
    
    def clear_nodesets(self) -> None:
        """
        Removes all node sets from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.NODESET.value):
            self.mesh.remove(el)
    
    def clear_surfacesets(self) -> None:
        """
        Removes all surface sets from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.SURFACESET.value):
            self.mesh.remove(el)
    
    def clear_elementsets(self) -> None:
        """
        Removes all element sets from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.ELEMENTSET.value):
            self.mesh.remove(el)
    
    def clear_materials(self) -> None:
        """
        Removes all materials from Material.
        """
        for el in self.material.findall(self.MAJOR_TAGS.MATERIAL.value):
            self.material.remove(el)
    
    def clear_nodal_loads(self) -> None:
        """
        Removes all nodal loads from Loads.
        """
        for el in self.loads.findall(self.MAJOR_TAGS.NODALLOAD.value):
            self.loads.remove(el)
    
    def clear_pressure_loads(self) -> None:
        """
        Removes all pressure loads from Loads.
        """
        for el in self.loads.findall(self.MAJOR_TAGS.SURFACELOAD.value):
            self.loads.remove(el)
    
    def clear_loadcurves(self) -> None:
        """
        Removes all load curves from LoadData.
        """
        for el in self.loaddata.findall(self.MAJOR_TAGS.LOADCURVE.value):
            self.loaddata.remove(el)
    
    def clear_boundary_conditions(self) -> None:    
        """
        Removes all boundary conditions from Boundary.
        """
        for el in self.boundary:
            self.boundary.remove(el)
    
    def clear_nodal_data(self) -> None:
        """
        Removes all nodal data from MeshData.
        """
        for el in self.meshdata.findall(self.MAJOR_TAGS.NODEDATA.value):
            self.meshdata.remove(el)
    
    def clear_surface_data(self) -> None:
        """
        Removes all surface data from MeshData.
        """
        for el in self.meshdata.findall(self.MAJOR_TAGS.SURFACE_DATA.value):
            self.meshdata.remove(el)
    
    def clear_element_data(self) -> None:
        """
        Removes all element data from MeshData.
        """
        for el in self.meshdata.findall(self.MAJOR_TAGS.ELEMENTDATA.value):
            self.meshdata.remove(el)
    
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
            self.clear_nodesets()
        if surfacesets:
            self.clear_surfacesets()
        if elementsets:
            self.clear_elementsets()
        if materials:
            self.clear_materials()
        if nodal_loads:
            self.clear_nodal_loads()
        if pressure_loads:
            self.clear_pressure_loads()
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
    
    def update_nodesets(self, nodesets: List[NodeSet]) -> None:
        """
        Updates node sets in Geometry by name, replacing existing node sets with the same name.

        Args:
            nodesets (list of NodeSet): List of NodeSet namedtuples, each containing a name and node IDs.
        """
        self.remove_nodesets([nodeset.name for nodeset in nodesets])
        self.add_nodesets(nodesets)
    
    def update_surfacesets(self, surfacesets: List[SurfaceSet]) -> None:
        """
        Updates surface sets in Geometry by name, replacing existing surface sets with the same name.

        Args:
            surfacesets (list of SurfaceSet): List of SurfaceSet namedtuples, each containing a name and node IDs.
        """
        self.remove_surfacesets([surfset.name for surfset in surfacesets])
        self.add_surfacesets(surfacesets)
    
    def update_elementsets(self, elementsets: List[ElementSet]) -> None:
        """
        Updates element sets in Geometry by name, replacing existing element sets with the same name.

        Args:
            elementsets (list of ElementSet): List of ElementSet namedtuples, each containing a name and element IDs.
        """
        self.remove_elementsets([elemset.name for elemset in elementsets])
        self.add_elementsets(elementsets)
    
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
    
    def update_pressure_loads(self, pressure_loads: List[PressureLoad]) -> None:
        """
        Updates pressure loads in Loads by surface, replacing existing pressure loads with the same surface.

        Args:
            pressure_loads (list of PressureLoad): List of PressureLoad namedtuples, each containing a surface, attributes, and multiplier.
        """
        self.remove_pressure_loads([load.surface for load in pressure_loads])
        self.add_pressure_loads(pressure_loads)
    
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
