from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree
import xml.etree.ElementTree as ET

from .bases import AbstractFebObject
import numpy as np
from typing import Union, List
from collections import OrderedDict, deque

from febio_python.core import (
    SURFACE_EL_TYPE,
    Nodes,
    Elements,
    Surfaces,
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
    GenericDomain,
    FEBioElementType,
    DiscreteSet,
    DiscreteMaterial
)

from ._caching import feb_instance_cache


class Feb25(AbstractFebObject):
    def __init__(self,
                 tree: Union[ElementTree, None] = None,
                 root: Union[Element, None] = None,
                 filepath: Union[str, Path] = None):
        self._default_version = 2.5
        super().__init__(tree, root, filepath)

        if self.version != 2.5:
            raise ValueError("This class is only for FEBio 2.5 files"
                             f"Version found: {self.version}")

    # =========================================================================================================
    # Retrieve methods
    # =========================================================================================================

    # Main geometry data
    # ------------------------------

    @feb_instance_cache
    def get_nodes(self, dtype: np.dtype = np.float32) -> List[Nodes]:
        all_nodes: OrderedDict = self.get_tag_data(self.LEAD_TAGS.GEOMETRY, self.MAJOR_TAGS.NODES, content_type="text", dtype=dtype)
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
        # last_elem_id = 1
        for elem_group in self.geometry.findall("Elements"):
            elem_type = elem_group.attrib.get("type")
            mat_id = elem_group.attrib.get("mat", None)
            elem_part_id = elem_group.attrib.get("part", None)
            if mat_id is not None:
                try:
                    mat_id = int(mat_id)
                except TypeError:
                    pass
            elem_name = elem_group.attrib.get("name")

            connectivity = deque()
            elem_ids = deque()
            for elem in elem_group.findall("elem"):
                # Convert the comma-separated string of node indices into an array of integers
                this_elem_connectivity = np.array(elem.text.split(','), dtype=dtype)
                connectivity.append(this_elem_connectivity)
                this_elem_id = int(elem.attrib["id"])
                elem_ids.append(this_elem_id)

            # Convert the list of element connectivities to a numpy array
            connectivity = np.array(connectivity, dtype=dtype) - 1  # Convert to zero-based indexing
            elem_ids = np.array(elem_ids, dtype=np.int64) - 1  # Convert to zero-based indexing
            # num_elems = connectivity.shape[0]
            # elem_ids = np.arange(last_elem_id, last_elem_id + num_elems, dtype=np.int64)
            # Create an Elements instance for each element
            element = Elements(name=elem_name,
                               mat=mat_id,
                               part=elem_part_id,
                               type=elem_type,
                               connectivity=connectivity,
                               ids=elem_ids)
            all_elements.append(element)
            # last_elem_id += num_elems

        return all_elements

    @feb_instance_cache
    def get_surfaces(self, dtype=np.int64) -> List[Surfaces]:
        all_surfaces = []
        # last_elem_id = 1
        for surf_group in self.geometry.findall("Surface"):
            # surf_type = surf_group.attrib.get("type")
            mat_id = surf_group.attrib.get("mat", None)
            surf_part_id = surf_group.attrib.get("part", None)
            if mat_id is not None:
                try:
                    mat_id = int(mat_id)
                except TypeError:
                    pass
            surf_name = surf_group.attrib.get("name")

            connectivity = deque()
            surf_ids = deque()
            
            # get first child element of the surf_group
            surf_elem = surf_group[0]
            surf_type = surf_elem.tag
            
            for elem in surf_group.findall(surf_type):
                # Convert the comma-separated string of node indices into an array of integers
                this_surf_connectivity = np.array(elem.text.split(','), dtype=dtype)
                connectivity.append(this_surf_connectivity)
                this_surf_id = int(elem.attrib["id"])
                surf_ids.append(this_surf_id)

            # Convert the list of surfent connectivities to a numpy array
            connectivity = np.array(connectivity, dtype=dtype) - 1  # Convert to zero-based indexing
            surf_ids = np.array(surf_ids, dtype=np.int64) - 1  # Convert to zero-based indexing
            # num_surfs = connectivity.shape[0]
            # surf_ids = np.arange(last_surf_id, last_surf_id + num_surfs, dtype=np.int64)
            # Create an surfents instance for each surfent
            surface = Surfaces(name=surf_name,
                               mat=mat_id,
                               part=surf_part_id,
                               type=surf_type,
                               connectivity=connectivity,
                               ids=surf_ids)
            all_surfaces.append(surface)
            # last_elem_id += num_elems

        return all_surfaces

    # Node, element, surface sets
    # ------------------------------

    @feb_instance_cache
    def get_node_sets(self, dtype=np.int64) -> List[NodeSet]:
        """
        Returns a dict with keys representing node set names and values \
        representing corresponding node ids as a numpy array of specified dtype.\

        Args:
            dtype (np.dtype): Numpy dtype.

        Returns:
            list: [Nodeset(name, node_ids)]
        """
        # Extract the nodesets dictionary from the .feb file
        nodesets: dict = self.get_tag_data(self.LEAD_TAGS.GEOMETRY, self.MAJOR_TAGS.NODESET, content_type="id", dtype=dtype)
        # Convert the nodesets dictionary to a list of Nodeset named tuples
        nodeset_list = list()
        for key, value in nodesets.items():
            value -= 1  # Convert to zero-based indexing
            nodeset_list.append(NodeSet(name=key, ids=value))
        return nodeset_list

    @feb_instance_cache
    def get_surface_sets(self, dtype=np.int64) -> List[SurfaceSet]:
        """
        Returns a dict with keys representing node set names and values \
        representing corresponding node ids as a numpy array of specified dtype.\

        Args:
            dtype (np.dtype): Numpy dtype.

        Returns:
            list: [SurfaceSet(name, node_ids)]
        """
        # Extract the surfacesets dictionary from the .feb file
        surfacesets: dict = self.get_tag_data(self.LEAD_TAGS.GEOMETRY, self.MAJOR_TAGS.SURFACESET, content_type="id", dtype=dtype)
        # Convert the surfacesets dictionary to a list of Nodeset named tuples
        surfaceset_list = list()
        for key, value in surfacesets.items():
            value -= 1  # Convert to zero-based indexing
            surfaceset_list.append(SurfaceSet(name=key, node_ids=value))
        return surfaceset_list

    @feb_instance_cache
    def get_element_sets(self, dtype=np.int64) -> List[ElementSet]:
        """
        Returns a dict with keys representing node set names and values \
        representing corresponding node ids as a numpy array of specified dtype.\

        Args:
            dtype (np.dtype): Numpy dtype.

        Returns:
            list: [ElementSet(name, node_ids)]
        """
        # Extract the elementsets dictionary from the .feb file
        elementsets: dict = self.get_tag_data(self.LEAD_TAGS.GEOMETRY, self.MAJOR_TAGS.ELEMENTSET, content_type="id", dtype=dtype)
        # Convert the elementsets dictionary to a list of Nodeset named tuples
        elementset_list = list()
        for key, value in elementsets.items():
            value -= 1  # Convert to zero-based indexing
            elementset_list.append(ElementSet(name=key, ids=value))
        return elementset_list

    # Parts
    # ------------------------------

    @feb_instance_cache
    def get_mesh_domains(self) -> List[GenericDomain]:
        # FEB 2.5 does not have a direct way to store mesh domains
        # It is based on "materials"
        materials = self.get_materials()
        mesh_domains = []
        for i, mat in enumerate(materials):
            new_domain = GenericDomain(
                id=i,
                name=mat.name,
                mat=mat.id
            )
            mesh_domains.append(new_domain)
        return mesh_domains

    # Materials
    # ------------------------------

    @feb_instance_cache
    def get_materials(self) -> List[Material]:
        materials_list = []
        for item in self.material.findall("material"):
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
        for i, load in enumerate(self.loads.findall("nodal_load")):
            scale_data = load.find("scale")

            # Convert scale text to float if possible, maintain as text if not
            try:
                scale_value = float(scale_data.text)
            except ValueError:
                scale_value = scale_data.text  # Keep as text if not convertible

            # Create a NodalLoad named tuple for the current load
            lc_curve = scale_data.attrib.get("lc", "NoCurve")  # Default to 'NoCurve' if not specified
            try:
                lc_curve = int(lc_curve)
            except ValueError:
                pass
            current_load = NodalLoad(
                dof=load.attrib.get("bc", "UndefinedBC"),  # Default to 'UndefinedBC' if not specified
                node_set=load.attrib.get("node_set", f"UnnamedNodeSet{i}"),  # Default to an indexed name if not specified
                scale=scale_value,
                load_curve=lc_curve  # Default to 'NoCurve' if not specified
            )

            # Add the new NodalLoad to the list
            nodal_loads.append(current_load)

        return nodal_loads

    @feb_instance_cache
    def get_surface_loads(self) -> List[SurfaceLoad]:
        pressure_loads_list = []
        for i, load in enumerate(self.loads.findall("surface_load")):

            # get the attributes info (surface, name and type)
            load_type = load.attrib.get("type")
            surf = load.attrib.get("surface", f"UnnamedSurface{i}")
            name = load.attrib.get("name", f"UnnamedSurfaceLoad{i}")
            # get the pressue (lc attribute and data value)
            el_press = load.find("pressure")
            lc_curve = el_press.attrib.get("lc", 1)
            scale = el_press.text
            # scale is a string representing either: float, name
            scale = float(scale) if scale.replace(".", "").isdigit() else scale
            # get the linear and symmetric stiffness tags
            linear_el = load.find("linear")
            linear = bool(int(linear_el.text)) if linear_el is not None else False       
            symm_el = load.find("symmetric_stiffness")
            symm = bool(int(symm_el.text)) if symm_el is not None else True
            
            # Create a SurfaceLoad instance for the current load
            current_load = SurfaceLoad(
                surface=surf,
                load_curve=lc_curve,
                scale=scale,
                type=load_type,
                name=name,
                linear=linear,
                symmetric_stiffness=symm
            )                
            
            # Append the created SurfaceLoad to the list
            pressure_loads_list.append(current_load)

        return pressure_loads_list

    @feb_instance_cache
    def get_load_curves(self, dtype=np.float32) -> List[LoadCurve]:
        load_curves_list = []
        for loadcurve_elem in self.loaddata.findall(self.MAJOR_TAGS.LOADCURVE.value):
            load_curve_id = loadcurve_elem.attrib['id']
            try:
                load_curve_id = int(load_curve_id)
            except ValueError:
                pass
            load_curve_type = loadcurve_elem.attrib['type']
            points = deque()

            # Extract points from each 'point' element
            for point_elem in loadcurve_elem.findall('point'):
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
        for elem in self.boundary:
            if elem.tag == 'fix':
                # Create an instance of FixCondition for each 'fix' element
                fix_condition = FixCondition(dof=elem.attrib['bc'], node_set=elem.attrib['node_set'], name=None)
                boundary_conditions_list.append(fix_condition)

            elif elem.tag == 'rigid_body':
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
                ids=np.array(_these_ids, dtype=np.int64) - 1  # Ensure IDs are zero-based
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
            for x in data.findall("face"):
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
            ref = data.attrib["surface"]
            name = data.attrib["name"]

            # Create a NodalData instance
            current_data = SurfaceData(
                surf_set=ref,
                name=name,
                data=np.array(_this_data, dtype=dtype),  # Ensure data is in the correct dtype
                ids=np.array(_these_ids, dtype=np.int64) - 1  # Ensure IDs are zero-based
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
                ids=np.array(_these_ids, dtype=np.int64) - 1  # Ensure IDs are zero-based
            )

            # Add the ElementData instance to the list
            elem_data_list.append(current_data)

        return elem_data_list

    @feb_instance_cache
    def get_discrete_sets(self, dtype=np.int64, find_related_nodesets=True) -> List[DiscreteSet]:
        
        # get all node sets by name and ids (for reference)
        node_sets_by_name = {n.name: n.ids for n in self.get_node_sets()}
        # find all discrete sets
        discrete_sets = self.geometry.findall("DiscreteSet")
        discrete_set_list = []
        for dset in discrete_sets:
            # get name
            name = dset.attrib.get("name", None)
            
            # Try to get the related discrete material
            mat_id = None # default value 
            # find discrete data associated with the set (if any)
            all_related_discrete_data = self.discrete.findall("discrete")
            related_discrete_data = [d for d in all_related_discrete_data if d.attrib.get("discrete_set", None) == name]
            if len(related_discrete_data) > 0:
                # get first data
                dset_related_data = related_discrete_data[0]
                # get the material id
                mat_id = dset_related_data.attrib.get("dmat", None)
            
            # get the source and destination node sets
            dset_ids = deque()
            for delem in dset.findall("delem"):
                # src and dst are in delem text and are comma separated
                src, dst = delem.text.split(",")
                src, dst = int(src), int(dst)
                dset_ids.append((src, dst))
            
            # transform to numpy array
            dset_ids = np.array(dset_ids, dtype=dtype)
            
            # get the source and destination node sets
            src_ids = dset_ids[:, 0]
            dst_ids = dset_ids[:, 1]
            
            # default values
            src = src_ids
            dst = dst_ids
            
            # Try to match with any existing node set
            if find_related_nodesets:
                src_names = [k for k, v in node_sets_by_name.items() if np.array_equal(v, src_ids)]
                dst_names = [k for k, v in node_sets_by_name.items() if np.array_equal(v, dst_ids)]

                # If there is only one match, use it
                if len(src_names) == 1:
                    src = src_names[0]
                
                if len(dst_names) == 1:
                    dst = dst_names[0]
                
            # Create a DiscreteSet instance
            current_dset = DiscreteSet(
                name=name,
                src=src,
                dst=dst,
                dmat=mat_id
            )

            # Add the DiscreteSet instance to the list
            discrete_set_list.append(current_dset)
            
        return discrete_set_list 

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
            nodes (list of Nodes): List of Nodes objects, each containing a name and coordinates.

        Raises:
            ValueError: If any node coordinates do not consist of three elements.
        """
        # Retrieve existing nodes and determine the last node ID
        existing_nodes_list = self.get_nodes()
        last_initial_id = existing_nodes_list[-1].ids[-1] + 1 if existing_nodes_list else 0

        existing_nodes = {node_elem.name: node_elem for node_elem in existing_nodes_list}

        for node in nodes:
            if node.name in existing_nodes:
                # Append to existing Nodes element
                el_root = self.geometry.find(f".//Nodes[@name='{node.name}']")
            else:
                # Create a new Nodes element if no existing one matches the name
                el_root = ET.Element("Nodes")
                el_root.set("name", node.name)
                self.geometry.append(el_root)  # Append new "Nodes" at the end of the geometry

            for i, node_xyz in enumerate(node.coordinates):
                if len(node_xyz) != 3:  # Ensure each node has exactly three coordinates
                    raise ValueError(f"Node {i + last_initial_id} does not have the correct number of coordinates. It should contain [x, y, z] values.")
                subel = ET.SubElement(el_root, "node")
                subel.set("id", str(i + last_initial_id + 1))
                subel.text = ",".join(map(str, node_xyz))  # Convert coordinates to comma-separated string

            # Update the last_initial_id for the next node group
            last_initial_id += len(node.coordinates)

    def add_elements(self, elements: List[Elements]) -> None:
        """
        Adds elements to Geometry, appending to existing elements if they share the same name.
        Automatically detects the highest element ID to ensure unique IDs for new elements.

        Args:
            elements (list of Elements): List of Elements objects, each containing name, material, type, and connectivity.

        Raises:
            ValueError: If any element connectivity does not meet expected format or length.
        """
        # Retrieve existing elements and determine the last element ID
        existing_elements_list = self.get_elements()
        last_elem_initial_id = existing_elements_list[-1].ids[-1] if existing_elements_list else 1

        existing_elements = {element.name: element for element in existing_elements_list}

        for element in elements:
            # Make sure the element type is valid, it must be a valid FEBio element type
            # However, user can also use VTK element types as input, but they must be
            # converted to FEBio types
            el_type = element.type
            # first, check if it is a VTK element type. FEBioElementType names
            # are the same as VTK element types.
            if el_type not in FEBioElementType.__members__.values():
                if str(el_type).upper() in FEBioElementType.__members__.keys():
                    el_type = FEBioElementType[el_type].value
                else:
                    raise TypeError(f"Element type {el_type} is not a valid FEBio element type.")

            if element.name in existing_elements:
                # Append to existing Elements group
                el_root = self.geometry.find(f".//Elements[@name='{element.name}']")
            else:
                # Create a new Elements group if no existing one matches the name
                el_root = ET.Element("Elements")
                el_root.set("type", el_type)  # TYPE AND MAT MUST BE SET FIRST, OTHERWISE FEBIO WILL NOT RECOGNIZE THE ELEMENTS
                if element.mat is not None:
                    el_root.set("mat", str(element.mat))
                if element.name is not None:
                    el_root.set("name", str(element.name))
                
                # Append new "Elements" at the end of the geometry
                self.geometry.append(el_root)
            
            # Add element connectivities as sub-elements
            for i, connectivity in enumerate(element.connectivity):
                subel = ET.SubElement(el_root, "elem")
                # Set the element ID and convert the connectivity to a comma-separated string
                subel.set("id", str(i + last_elem_initial_id))
                subel.text = ",".join(map(str, connectivity + 1))  # Convert connectivity to comma-separated string

            # Update the last_elem_initial_id for the next element group
            last_elem_initial_id += len(element.connectivity)

    def add_surfaces(self, surfaces: List[Surfaces]) -> None:
        # Filter surfaces by surface type
        filtered = [elem for elem in surfaces if elem.type in SURFACE_EL_TYPE.__members__]
        if len(filtered) == 0:
            raise ValueError("No surface surfaces found in the input list. Try using add_volume_surfaces() instead.")
        
        # Retrieve existing surfaces and determine the last element ID
        existing_surfaces_list = self.get_surfaces()
        last_surf_initial_id = existing_surfaces_list[-1].ids[-1] if existing_surfaces_list else 1
        
        existing_surfaces = {element.name: element for element in existing_surfaces_list}

        for surface in filtered:
            # Make sure the surface type is valid, it must be a valid FEBio surface type
            # However, user can also use VTK surface types as input, but they must be
            # converted to FEBio types
            el_type = surface.type
            # first, check if it is a VTK element type. FEBioElementType names
            # are the same as VTK element types.
            if el_type not in FEBioElementType.__members__.values():
                if str(el_type).upper() in FEBioElementType.__members__.keys():
                    el_type = FEBioElementType[el_type].value
                else:
                    raise TypeError(f"Element type {el_type} is not a valid FEBio element type.")

            if surface.name in existing_surfaces:
                # Append to existing Surfaces group
                el_root = self.geometry.find(f".//Surfaces[@name='{surface.name}']")
            else:
                
                el_root = ET.Element("Surface")
                el_root.set("name", str(surface.name))

                # Append new "Surfaces" at the end of the geometry
                self.geometry.append(el_root)
            
            # Add element connectivities as sub-surfaces
            for i, connectivity in enumerate(surface.connectivity):
                subel = ET.SubElement(el_root, el_type)  # FEBio use element type as tag name for surface surfaces
                # Set the element ID and convert the connectivity to a comma-separated string
                subel.set("id", str(i + last_surf_initial_id)) 
                subel.text = ",".join(map(str, connectivity + 1))  # Convert connectivity to comma-separated string

            # Update the last_elem_initial_id for the next element group
            last_surf_initial_id += len(surface.connectivity)

    # Node, element, surface sets
    # ------------------------------

    def add_node_sets(self, nodesets: List[NodeSet]) -> None:
        """
        Adds node sets to Geometry, appending to existing node sets if they share the same name.

        Args:
            nodesets (list of NodeSet): List of NodeSet objects, each containing a name and node IDs.
        """
        existing_nodesets = {nodeset.name: nodeset for nodeset in self.get_node_sets()}

        for nodeset in nodesets:
            node_ids = nodeset.ids
            if nodeset.name in existing_nodesets:
                # Append to existing NodeSet element
                el_root = self.geometry.find(f".//NodeSet[@name='{nodeset.name}']")
                # Merge the existing node IDs with the new ones and remove duplicates
                existing_ids = existing_nodesets[nodeset.name].ids
                node_ids = np.unique(np.concatenate((existing_ids, node_ids)))
            else:
                # Create a new NodeSet element if no existing one matches the name
                el_root = ET.Element("NodeSet")
                el_root.set("name", nodeset.name)
                self.geometry.append(el_root)

            # Add node IDs as sub-elements
            for node_id in node_ids:
                subel = ET.SubElement(el_root, "node")
                subel.set("id", str(int(node_id + 1)))  # Convert to one-based indexing

    def add_surface_sets(self, surfacesets: List[SurfaceSet]) -> None:
        """
        Adds surface sets to Geometry, appending to existing surface sets if they share the same name.

        Args:
            surfacesets (list of SurfaceSet): List of SurfaceSet objects, each containing a name and node IDs.
        """
        existing_surfacesets = {surfset.name: surfset for surfset in self.get_surface_sets()}

        for surfset in surfacesets:
            surf_ids = surfset.node_ids
            if surfset.name in existing_surfacesets:
                # Append to existing SurfaceSet element
                el_root = self.geometry.find(f".//SurfaceSet[@name='{surfset.name}']")
                # Merge the existing node IDs with the new ones and remove duplicates
                existing_ids = existing_surfacesets[surfset.name].node_ids
                surf_ids = np.unique(np.concatenate((existing_ids, surf_ids)))
            else:
                # Create a new SurfaceSet element if no existing one matches the name
                el_root = ET.Element("SurfaceSet")
                el_root.set("name", surfset.name)
                self.geometry.append(el_root)

            # Add node IDs as sub-elements
            for node_id in surf_ids:
                subel = ET.SubElement(el_root, "node")
                subel.set("id", str(int(node_id + 1)))  # Convert to one-based indexing

    def add_element_sets(self, elementsets: List[ElementSet]) -> None:
        """
        Adds element sets to Geometry, appending to existing element sets if they share the same name.

        Args:
            elementsets (list of ElementSet): List of ElementSet objects, each containing a name and element IDs.
        """
        existing_elementsets = {elemset.name: elemset for elemset in self.get_element_sets()}

        for elemset in elementsets:
            elem_ids = elemset.ids
            if elemset.name in existing_elementsets:
                # Append to existing ElementSet element
                el_root = self.geometry.find(f".//ElementSet[@name='{elemset.name}']")
                # Merge the existing element IDs with the new ones and remove duplicates
                existing_ids = existing_elementsets[elemset.name].ids
                elem_ids = np.unique(np.concatenate((existing_ids, elem_ids)))
            else:
                # Create a new ElementSet element if no existing one matches the name
                el_root = ET.Element("ElementSet")
                el_root.set("name", elemset.name)
                self.geometry.append(el_root)

            # Add element IDs as sub-elements
            for elem_id in elem_ids:
                subel = ET.SubElement(el_root, "elem")
                subel.set("id", str(int(elem_id + 1)))  # Convert to one-based indexing

    def add_discrete_sets(self, discrete_sets: List[DiscreteSet]) -> None:
        """
        Adds discrete sets to Geometry, appending to existing discrete sets if they share the same name.

        Args:
            discrete_sets (list of DiscreteSet): List of DiscreteSet objects, each containing a name and element IDs.
        """
        existing_discrete_sets = {dset.name: dset for dset in self.get_discrete_sets(find_related_nodesets=False)}
        nodesets_by_name = {nodeset.name: nodeset.ids for nodeset in self.get_node_sets()}
        
        for dset in discrete_sets:
            already_exists = dset.name in existing_discrete_sets
            src_ids = dset.src
            dst_ids = dset.dst
            
            if isinstance(src_ids, str):  # then it is a nodeset.
                # Find the node set with the same name
                node_set = nodesets_by_name.get(src_ids, None)
                # If the node set does not exist, raise an error
                if node_set is None:
                    raise ValueError(
                        f"Trying to create a DiscreteSet {dset.name} with "
                        f"source node set {src_ids}, but the node set does not exist. "
                        "Please, create the node set first and try again.")
                # Get the node IDs from the node set
                src_ids = node_set  # ids
            else:
                # make sure the source is a numpy array
                src_ids = np.array(src_ids, dtype=np.int64)

            # make sure it is one-based indexing
            src_ids = src_ids + 1

            # same for dst
            if isinstance(dst_ids, str):  # then it is a nodeset.
                # Find the node set with the same name
                node_set = nodesets_by_name.get(dst_ids, None)
                # If the node set does not exist, raise an error
                if node_set is None:
                    raise ValueError(
                        f"Trying to create a DiscreteSet {dset.name} with "
                        f"destination node set {dst_ids}, but the node set does not exist. "
                        "Please, create the node set first and try again.")
                # Get the node IDs from the node set
                dst_ids = node_set
            else:
                # make sure the destination is a numpy array
                dst_ids = np.array(dst_ids, dtype=np.int64)
            # make sure it is one-based indexing
            dst_ids = dst_ids + 1
            
            # Combine the source and destination IDs
            src_dst = np.column_stack((src_ids, dst_ids))
            if already_exists:
                # Append to existing DiscreteSet element
                el_root = self.geometry.find(f".//DiscreteSet[@name='{dset.name}']")
                # Merge the existing element IDs with the new ones and remove duplicates
                existing_src = existing_discrete_sets[dset.name].src
                existing_dst = existing_discrete_sets[dset.name].dst
                existing_ids = np.column_stack((existing_src, existing_dst))
                src_dst = np.unique(np.concatenate((existing_ids, src_dst)))
            else:
                # Create a new DiscreteSet element if no existing one matches the name
                el_root = ET.Element("DiscreteSet")
                el_root.set("name", dset.name)
                self.geometry.append(el_root)

            # Add element IDs as sub-elements
            for src, dst in zip(src_ids, dst_ids):
                subel = ET.SubElement(el_root, "delem")
                subel.text = f"{src},{dst}"
                
            # Add the discrete material if it exists
            if dset.dmat is not None and not already_exists:
                # Create a new DiscreteData element if no existing one matches the name
                el_data = ET.Element("discrete")
                el_data.set("discrete_set", dset.name)
                el_data.set("dmat", str(dset.dmat))
                self.discrete.append(el_data)

    # Materials
    # ------------------------------

    def add_materials(self, materials: List[Material]) -> None:
        """
        Adds materials to Material, appending to existing materials if they share the same ID.

        Args:
            materials (list of Material): List of Material objects, each containing an ID, type, parameters, name, and attributes.
        """
        existing_materials = {material.id: material for material in self.get_materials()}
        element_by_mat = {element.mat: element for element in self.get_elements()}

        for material in materials:
            if material.id in existing_materials or str(material.id) in existing_materials:
                print(f"Material with ID {material.id} already exists")
                # Append to existing Material element
                el_root = self.material.find(f".//material[@id='{material.id}']")
            else:
                # Create a new Material element if no existing one matches the ID
                el_root = ET.Element("material")
                el_root.set("id", str(material.id))
                el_root.set("type", material.type)
                el_root.set("name", material.name)
                self.material.append(el_root)

            # Add parameters as sub-elements
            mat_params: dict = material.parameters
            if not isinstance(mat_params, dict):
                raise ValueError(f"Material parameters should be a dictionary, not {type(mat_params)}")
            for key, value in mat_params.items():
                subel = ET.SubElement(el_root, key)
                if isinstance(value, (int, float, np.number)):
                    subel.text = str(value)
                elif isinstance(value, np.ndarray):
                    # then we must add as mesh data
                    ref_data_name = f"material_{material.id}_{key}"
                    related_elem = element_by_mat.get(material.id, None)
                    # Make sure the element exists
                    if related_elem is None:
                        raise ValueError(f"Material with ID {material.id} does not exist in the geometry."
                                         "Cannot add material parameter as mesh data. Please, add the Elements first.")
                    # Make sure that data is the same length as the number of elements
                    if len(value) != related_elem.ids.size:
                        raise ValueError(f"Material parameter '{key}' must have the same length as the number of elements.")
                    # prepare the element data
                    elem_data = ElementData(elem_set=related_elem.name,
                                            name=ref_data_name,
                                            data=value,
                                            ids=related_elem.ids)
                    # add the element data
                    self.add_element_data([elem_data])
                    # add the reference to the material
                    subel.text = f"{ref_data_name}"
                else:
                    # if it is not a number or an array, we just add it as text    
                    subel.text = str(value)

    def add_discrete_materials(self, materials: List[DiscreteMaterial]) -> None:
        """
        Adds discrete materials to Discrete.

        Args:
            materials (list of Material): List of Material objects, each containing an ID, type, parameters, name, and attributes.
        """

        for material in materials:
            # Create a new Material element if no existing one matches the ID
            el_root = ET.Element("discrete_material")
            el_root.set("id", str(material.id))
            el_root.set("type", material.type)
            el_root.set("name", material.name)

            # Add parameters as sub-elements
            mat_params: dict = material.parameters
            if not isinstance(mat_params, dict):
                raise ValueError(f"Material parameters should be a dictionary, not {type(mat_params)}")
            for key, value in mat_params.items():
                subel = ET.SubElement(el_root, key)
                subel.text = str(value)
            
            # discrete materials must be at the top, 
            # and ordered by id
            # thus, we cannot simply append the new material
            # This results in error: self.discrete.append(el_root)
            
            # we need to find the correct position to insert the new material
            # we need to find the last material with an id smaller than the new material
            # and insert the new material after that
            # if no such material exists, we insert the new material at the beginning
            
            # find all discrete materials
            all_discrete_materials = self.discrete.findall("discrete_material")
            # find the last material with an id smaller than the new material
            last_id = -1
            for i, mat in enumerate(all_discrete_materials):
                if int(mat.attrib["id"]) < material.id:
                    last_id = i
            # insert the new material after the last material with an id smaller than the new material
            if last_id == -1:
                self.discrete.insert(0, el_root)
            else:
                self.discrete.insert(last_id + 1, el_root)

    # Loads
    # ------------------------------

    def add_nodal_loads(self, nodal_loads: List[NodalLoad]) -> None:
        """
        Adds nodal loads to Loads.

        Args:
            nodal_loads (list of NodalLoad): List of NodalLoad objects, each containing a boundary condition, node set, scale, and load curve.
        """

        for load in nodal_loads:
            el_root = ET.Element("nodal_load")
            el_root.set("node_set", load.node_set)
            self.loads.append(el_root)

            el_root.set("bc", load.dof)

            scale_subel = ET.SubElement(el_root, "scale")
            scale_subel.set("lc", str(load.load_curve))
            if load.scale is None:
                scale_subel.text = "1.0"  # Default to 1.0 if no scale is provided
            elif isinstance(load.scale, (str, int, float, np.number)):
                scale_subel.text = str(load.scale)
            elif isinstance(load.scale, np.ndarray):
                # we need to add this as mesh data; and then reference it here
                ref_data_name = f"nodal_load_{load.node_set}_{load.dof}_scale"
                
                scale_subel.text = f"1*{ref_data_name}"
                # we need to retrieve the node ids for this node set
                nodesets = self.get_node_sets()
                # find the node set
                node_set = [ns for ns in nodesets if ns.name == load.node_set]
                if len(node_set) == 0:
                    raise ValueError(f"Node set {load.node_set} not found in the geometry."
                                     "Please, either add 'scale' as an str and manually provide "
                                     "the scale value as a mesh data and add the node set to the geometry.")
                node_set = node_set[0]
                # prepare the nodal data
                nodal_data = NodalData(node_set=load.node_set,
                                       name=ref_data_name,
                                       data=load.scale,
                                       ids=np.arange(0, len(node_set.ids) + 1))  # add_nodal_data will convert to one-based indexing
                # add the nodal data
                self.add_nodal_data([nodal_data])

    def add_surface_loads(self, pressure_loads: List[SurfaceLoad]) -> None:
        """
        Adds pressure loads to Loads.

        Args:
            pressure_loads (list of SurfaceLoad): List of SurfaceLoad objects, each containing a surface, attributes, and multiplier.
        """

        for load in pressure_loads:

            # Create a new SurfaceLoad element if no existing one matches the surface
            el_root = ET.Element("surface_load")
            # set the type of surface load
            el_root.set("type", str(load.type))
            # set the surface name
            el_root.set("surface", str(load.surface))
            # set the name (optional)
            if load.name is not None:
                el_root.set("name", str(load.name))

            # Add pressure tag, with load curve and scale data
            el_pressure = ET.SubElement(el_root, "pressure")
            el_pressure.set("lc", str(load.load_curve))
            if load.scale is None:
                el_pressure.text = "1.0"  # Default to 1.0 if no scale is provided
            elif isinstance(load.scale, (str, int, float, np.number)):
                el_pressure.text = str(load.scale)
            elif isinstance(load.scale, np.ndarray):
                # we need to add this as mesh data; and then reference it here
                ref_data_name = f"surface_load_{load.surface}_scale"
                el_pressure.text = f"1*{ref_data_name}"
                # prepare the nodal data
                surf_data = SurfaceData(surf_set=load.surface,
                                         name=ref_data_name,
                                         data=load.scale,
                                         ids=np.arange(0, len(load.scale) + 1))
                # add the surface data
                self.add_surface_data([surf_data])
            
            # add linear tag with text data
            el_linear = ET.SubElement(el_root, "linear")
            el_linear.text = str(int(load.linear)) # convert boolean to int
            
            # add symmetric_stiffness tag with text data
            el_symmetric_stiffness = ET.SubElement(el_root, "symmetric_stiffness")
            el_symmetric_stiffness.text = str(int(load.symmetric_stiffness))            

            # Append the new SurfaceLoad element to the list
            self.loads.append(el_root)

    def add_load_curves(self, load_curves: List[LoadCurve]) -> None:
        """
        Adds load curves to LoadData, appending to existing load curves if they share the same ID.

        Args:
            load_curves (list of LoadCurve): List of LoadCurve objects, each containing an ID, type, and data.
        """
        existing_load_curves = {curve.id: curve for curve in self.get_load_curves()}

        for curve in load_curves:
            if curve.id in existing_load_curves:
                # Append to existing LoadCurve element
                el_root = self.loaddata.find(f".//loadcurve[@id='{curve.id}']")
            else:
                # Create a new LoadCurve element if no existing one matches the ID
                el_root = ET.Element("loadcurve")
                el_root.set("id", str(curve.id))
                el_root.set("type", curve.interpolate_type)
                self.loaddata.append(el_root)

            for point in curve.data:
                subel = ET.SubElement(el_root, "point")
                subel.text = ",".join(map(str, point))

    # Boundary conditions
    # ------------------------------

    def add_boundary_conditions(self, boundary_conditions: List[BoundaryCondition]) -> None:
        """
        Adds boundary conditions to Boundary.

        Args:
            boundary_conditions (list of BoundaryCondition): List of boundary condition objects.
        """

        for bc in boundary_conditions:
            bc_type = bc.type if hasattr(bc, "type") else bc.__class__.__name__
            if bc_type == "FixCondition":
                el_root = ET.Element("fix")
            else:
                bc_type_tag = bc_type.replace(" ", "_")
                el_root = ET.Element(bc_type_tag)
            self.boundary.append(el_root)

            if hasattr(bc, "node_set") and bc.node_set is not None:
                el_root.set("node_set", bc.node_set)
            if hasattr(bc, "material") and bc.material is not None:
                el_root.set("mat", str(bc.material))
            if hasattr(bc, "name") and bc.name is not None:
                el_root.set("name", str(bc.name))
            if hasattr(bc, "dof") and bc.dof is not None:
                if not bc_type == "rigid body":
                    el_root.set("bc", bc.dof)
                else:
                    for fixed in bc.dof.split(","):
                        subel = ET.SubElement(el_root, "fixed")
                        subel.set("bc", fixed)

    # Mesh data
    # ------------------------------

    def add_nodal_data(self, nodal_data: List[NodalData]) -> None:
        """
        Adds nodal data to MeshData

        Args:
            nodal_data (list of NodalData): List of NodalData objects, each containing a node set, name, and data.
        """
        # existing_nodal_data = {data.name: data for data in self.get_nodal_data()}

        for data in nodal_data:
            # if data.name in existing_nodal_data:
            #     # Append to existing NodalData element
            #     el_root = self.meshdata.find(f".//{self.MAJOR_TAGS.NODEDATA.value}[@name='{data.name}']")
            # else:
            # Create a new NodalData element if no existing one matches the node set
            el_root = ET.Element(self.MAJOR_TAGS.NODEDATA.value)
            el_root.set("node_set", data.node_set)
            el_root.set("name", data.name)
            self.meshdata.append(el_root)

            # Add node IDs and data as sub-elements
            for i, node_data in enumerate(data.data):
                subel = ET.SubElement(el_root, "node")
                subel.set("lid", str(data.ids[i] + 1))  # Convert to one-based indexing
                if isinstance(node_data, (str, int, float, np.number)):
                    subel.text = str(node_data)
                else:
                    try:
                        subel.text = ",".join(map(str, node_data))
                    except TypeError:
                        raise ValueError(f"Node data for node set {data.node_set} is not in the correct format.")

    def add_surface_data(self, surface_data: List[SurfaceData]) -> None:
        """
        Adds surface data to MeshData, appending to existing surface data if they share the same surface set.

        Args:
            surface_data (list of SurfaceData): List of SurfaceData objects, each containing a surface set, name, and data.
        """
        existing_surface_data = {data.surf_set: data for data in self.get_surface_data()}

        for data in surface_data:
            if data.surf_set in existing_surface_data:
                # Append to existing SurfaceData element
                el_root = self.meshdata.find(f".//{self.MAJOR_TAGS.SURFACE_DATA.value}[@surf_set='{data.surf_set}']")
            else:
                # Create a new SurfaceData element if no existing one matches the surface set
                el_root = ET.Element(self.MAJOR_TAGS.SURFACE_DATA.value)
                el_root.set("surface", data.surf_set)
                el_root.set("name", data.name)
                self.meshdata.append(el_root)

            for i, surf_data in enumerate(data.data):
                subel = ET.SubElement(el_root, "face")
                subel.set("lid", str(data.ids[i] + 1))  # Convert to one-based indexing
                if isinstance(surf_data, (str, int, float, np.number)):
                    subel.text = str(surf_data)
                else:
                    subel.text = ",".join(map(str, surf_data))

    def add_element_data(self, element_data: List[ElementData]) -> None:
        """
        Adds element data to MeshData, appending to existing element data if they share the same element set.

        Args:
            element_data (list of ElementData): List of ElementData objects, each containing an element set, name, and data.
        """
        # existing_element_data = {data.elem_set: data for data in self.get_element_data()}

        for data in element_data:
            # if data.elem_set in existing_element_data:
            #     # Append to existing ElementData element
            #     el_root = self.meshdata.find(f".//{self.MAJOR_TAGS.ELEMENTDATA.value}[@elem_set='{data.elem_set}']")
            # else:
            # Create a new ElementData element if no existing one matches the element set
            el_root = ET.Element(self.MAJOR_TAGS.ELEMENTDATA.value)
            el_root.set("elem_set", data.elem_set)
            if data.name is not None:
                el_root.set("name", data.name)
            if data.var is not None:
                el_root.set("var", data.var)
            if data.data_type is not None:
                el_root.set("datatype", data.data_type)
            else:  # try to assume based on 
                if isinstance(data.data, np.ndarray):
                    if data.data.ndim == 1:
                        el_root.set("datatype", "scalar")
                    else:
                        if data.data.shape[1] == 3:
                            el_root.set("datatype", "vector")
                        else:
                            el_root.set("datatype", "matrix")
            self.meshdata.append(el_root)

            for i, elem_data in enumerate(data.data):
                subel = ET.SubElement(el_root, "elem")
                subel.set("lid", str(data.ids[i] + 1))  # Convert to one-based indexing
                if isinstance(elem_data, (str, int, float, np.number)):
                    subel.text = str(elem_data)
                else:
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
            el = self.geometry.find(f".//Nodes[@name='{name}']")
            if el is not None:
                self.geometry.remove(el)

    def remove_elements(self, names: List[str]) -> None:
        """
        Removes elements from Geometry by name.

        Args:
            names (list of str): List of element names to remove.
        """
        for name in names:
            el = self.geometry.find(f".//Elements[@name='{name}']")
            if el is not None:
                self.geometry.remove(el)

    def remove_all_surface_elements(self) -> None:
        """
        Removes all surface elements from Geometry.
        """
        for el in self.geometry.findall("Elements"):
            if el.attrib.get("type") in SURFACE_EL_TYPE.__members__:
                self.geometry.remove(el)

    def remove_all_volume_elements(self) -> None:
        """
        Removes all volume elements from Geometry.
        """
        for el in self.geometry.findall("Elements"):
            if el.attrib.get("type") not in SURFACE_EL_TYPE.__members__:
                self.geometry.remove(el)

    # Node, element, surface sets
    # ------------------------------

    def remove_node_sets(self, names: List[str]) -> None:
        """
        Removes node sets from Geometry by name.

        Args:
            names (list of str): List of node set names to remove.
        """
        for name in names:
            el = self.geometry.find(f".//NodeSet[@name='{name}']")
            if el is not None:
                self.geometry.remove(el)

    def remove_surface_sets(self, names: List[str]) -> None:
        """
        Removes surface sets from Geometry by name.

        Args:
            names (list of str): List of surface set names to remove.
        """
        for name in names:
            el = self.geometry.find(f".//SurfaceSet[@name='{name}']")
            if el is not None:
                self.geometry.remove(el)

    def remove_element_sets(self, names: List[str]) -> None:
        """
        Removes element sets from Geometry by name.

        Args:
            names (list of str): List of element set names to remove.
        """
        for name in names:
            el = self.geometry.find(f".//ElementSet[@name='{name}']")
            if el is not None:
                self.geometry.remove(el)

    def remove_discrete_sets(self, names: List[str]) -> None:
        """
        Removes discrete sets from Geometry by name.

        Args:
            names (list of str): List of discrete set names to remove.
        """
        for name in names:
            el = self.geometry.find(f".//DiscreteSet[@name='{name}']")
            if el is not None:
                self.geometry.remove(el)
            el = self.discrete.find(f".//discrete[@discrete_set='{name}']")
            if el is not None:
                self.discrete.remove(el)

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

    def remove_discrete_materials(self, ids: List[Union[str, int]]) -> None:
        """
        Removes discrete materials from Discrete by ID, name or type.

        Args:
            ids (list of int): List of discrete material IDs, types, remove.
        """
        for id in ids:
            el = self.discrete.find(f".//discrete_material[@id='{id}']")
            if el is not None:
                self.discrete.remove(el)
                continue
            el = self.discrete.find(f".//discrete_material[@name='{id}']")
            if el is not None:
                self.discrete.remove(el)
                continue
            el = self.discrete.find(f".//discrete_material[@type='{id}']")
            if el is not None:
                self.discrete.remove(el)
                continue

    # Loads
    # ------------------------------

    def remove_nodal_loads(self, bc_or_node_sets: List[str]) -> None:
        """
        Removes nodal loads from Loads by boundary condition or node set.

        Args:
            bc_or_node_sets (list of str): List of boundary conditions or node sets to remove.
        """
        for bc in bc_or_node_sets:
            el = self.loads.find(f".//nodal_load[@bc='{bc}']")
            if el is not None:
                self.loads.remove(el)
                continue
            el = self.loads.find(f".//nodal_load[@node_set='{bc}']")
            if el is not None:
                self.loads.remove(el)
                continue

    def remove_surface_loads(self, surfaces: List[str]) -> None:
        """
        Removes pressure loads from Loads by surface.

        Args:
            surfaces (list of str): List of surfaces to remove.
        """
        for surf in surfaces:
            el = self.loads.find(f".//surface_load[@surface='{surf}']")
            if el is not None:
                self.loads.remove(el)

    def remove_load_curves(self, ids: List[int]) -> None:
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

    def remove_boundary_conditions(self, types: List[str], bc: List[str] = None) -> None:
        """
        Removes boundary conditions from Boundary by type and optionally fiter type by boundary condition.
        e.g. remove_boundary_conditions(["fix"], ["BC1"]), instead of removing all fix conditions, only BC1 will be removed.

        Args:
            types (list of str): List of boundary condition types to remove.
            bc (list of str): List of boundary conditions to remove.
        """
        for type in types:
            if bc is None:
                el = self.boundary.find(f".//{type}")
                if el is not None:
                    self.boundary.remove(el)
            else:
                for b in bc:
                    el = self.boundary.find(f".//{type}[@bc='{b}']")
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
        for el in self.geometry.findall(self.MAJOR_TAGS.NODES.value):
            self.geometry.remove(el)

    def clear_elements(self) -> None:
        """
        Removes all elements from Geometry.
        """
        for el in self.geometry.findall(self.MAJOR_TAGS.ELEMENTS.value):
            self.geometry.remove(el)

    def clear_surface_elements(self) -> None:
        """
        Removes all surface elements from Geometry.
        """
        for el in self.geometry.findall(self.MAJOR_TAGS.ELEMENTS.value):
            if el.attrib.get("type") in SURFACE_EL_TYPE.__members__:
                self.geometry.remove(el)

    def clear_volume_elements(self) -> None:
        """
        Removes all volume elements from Geometry.
        """
        for el in self.geometry.findall(self.MAJOR_TAGS.ELEMENTS.value):
            if el.attrib.get("type") not in SURFACE_EL_TYPE.__members__:
                self.geometry.remove(el)

    def clear_node_sets(self) -> None:
        """
        Removes all node sets from Geometry.
        """
        for el in self.geometry.findall(self.MAJOR_TAGS.NODESET.value):
            self.geometry.remove(el)

    def clear_surface_sets(self) -> None:
        """
        Removes all surface sets from Geometry.
        """
        for el in self.geometry.findall(self.MAJOR_TAGS.SURFACESET.value):
            self.geometry.remove(el)

    def clear_element_sets(self) -> None:
        """
        Removes all element sets from Geometry.
        """
        for el in self.geometry.findall(self.MAJOR_TAGS.ELEMENTSET.value):
            self.geometry.remove(el)

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

    def clear_surface_loads(self) -> None:
        """
        Removes all pressure loads from Loads.
        """
        for el in self.loads.findall(self.MAJOR_TAGS.SURFACELOAD.value):
            self.loads.remove(el)

    def clear_load_curves(self) -> None:
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

    def clear_discrete_sets(self) -> None:
        """
        Removes all discrete sets from Geometry.
        """
        for el in self.geometry.findall(self.MAJOR_TAGS.DISCRETESET.value):
            self.geometry.remove(el)
        for el in self.discrete.findall(self.MAJOR_TAGS.DISCRETE.value):
            self.discrete.remove(el)
    
    def clear_discrete_materials(self) -> None:
        """
        Removes all discrete materials from Discrete.
        """
        for el in self.discrete.findall(self.MAJOR_TAGS.DISCRETEMATERIAL.value):
            self.discrete.remove(el)

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
            self.clear_load_curves()
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
            nodes (list of Nodes): List of Nodes objects, each containing a name and coordinates.
        """
        self.remove_nodes([node.name for node in nodes])
        self.add_nodes(nodes)

    def update_elements(self, elements: List[Elements]) -> None:
        """
        Updates elements in Geometry by name, replacing existing elements with the same name.

        Args:
            elements (list of Elements): List of Elements objects, each containing name, material, type, and connectivity.
        """
        self.remove_elements([element.name for element in elements])
        self.add_elements(elements)

    # Node, element, surface sets
    # ------------------------------

    def update_node_sets(self, nodesets: List[NodeSet]) -> None:
        """
        Updates node sets in Geometry by name, replacing existing node sets with the same name.

        Args:
            nodesets (list of NodeSet): List of NodeSet objects, each containing a name and node IDs.
        """
        self.remove_node_sets([nodeset.name for nodeset in nodesets])
        self.add_node_sets(nodesets)

    def update_surface_sets(self, surfacesets: List[SurfaceSet]) -> None:
        """
        Updates surface sets in Geometry by name, replacing existing surface sets with the same name.

        Args:
            surfacesets (list of SurfaceSet): List of SurfaceSet objects, each containing a name and node IDs.
        """
        self.remove_surface_sets([surfset.name for surfset in surfacesets])
        self.add_surface_sets(surfacesets)

    def update_element_sets(self, elementsets: List[ElementSet]) -> None:
        """
        Updates element sets in Geometry by name, replacing existing element sets with the same name.

        Args:
            elementsets (list of ElementSet): List of ElementSet objects, each containing a name and element IDs.
        """
        self.remove_element_sets([elemset.name for elemset in elementsets])
        self.add_element_sets(elementsets)

    # Materials
    # ------------------------------

    def update_materials(self, materials: List[Material]) -> None:
        """
        Updates materials in Material by ID, replacing existing materials with the same ID.

        Args:
            materials (list of Material): List of Material objects, each containing an ID, type, parameters, name, and attributes.
        """
        self.remove_materials([material.id for material in materials])
        self.add_materials(materials)

    # Loads
    # ------------------------------

    def update_nodal_loads(self, nodal_loads: List[NodalLoad]) -> None:
        """
        Updates nodal loads in Loads by node set, replacing existing nodal loads with the same node set.

        Args:
            nodal_loads (list of NodalLoad): List of NodalLoad objects, each containing a boundary condition, node set, scale, and load curve.
        """
        self.remove_nodal_loads([load.node_set for load in nodal_loads])
        self.add_nodal_loads(nodal_loads)

    def update_surface_loads(self, pressure_loads: List[SurfaceLoad]) -> None:
        """
        Updates pressure loads in Loads by surface, replacing existing pressure loads with the same surface.

        Args:
            pressure_loads (list of SurfaceLoad): List of SurfaceLoad objects, each containing a surface, attributes, and multiplier.
        """
        self.remove_surface_loads([load.surface for load in pressure_loads])
        self.add_surface_loads(pressure_loads)

    def update_load_curves(self, load_curves: List[LoadCurve]) -> None:
        """
        Updates load curves in LoadData by ID, replacing existing load curves with the same ID.

        Args:
            load_curves (list of LoadCurve): List of LoadCurve objects, each containing an ID, type, and data.
        """
        self.remove_load_curves([curve.id for curve in load_curves])
        self.add_load_curves(load_curves)

    # Boundary conditions
    # ------------------------------

    def update_boundary_conditions(self, boundary_conditions: List[Union[FixCondition, RigidBodyCondition, BoundaryCondition]]) -> None:
        """
        Updates boundary conditions in Boundary, replacing existing boundary conditions with the same type.

        Args:
            boundary_conditions (list of Union[FixCondition, RigidBodyCondition, BoundaryCondition]): List of boundary condition objects.
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
            nodal_data (list of NodalData): List of NodalData objects, each containing a node set, name, and data.
        """
        self.remove_nodal_data([data.node_set for data in nodal_data])
        self.add_nodal_data(nodal_data)

    def update_surface_data(self, surface_data: List[SurfaceData]) -> None:
        """
        Updates surface data in MeshData by surface set, replacing existing surface data with the same surface set.

        Args:
            surface_data (list of SurfaceData): List of SurfaceData objects, each containing a surface set, name, and data.
        """
        self.remove_surface_data([data.node_set for data in surface_data])
        self.add_surface_data(surface_data)

    def update_element_data(self, element_data: List[ElementData]) -> None:
        """
        Updates element data in MeshData by element set, replacing existing element data with the same element set.

        Args:
            element_data (list of ElementData): List of ElementData objects, each containing an element set, name, and data.
        """
        self.remove_element_data([data.node_set for data in element_data])
        self.add_element_data(element_data)

    # =========================================================================================================
    # Base control setup methods
    # =========================================================================================================

    def setup_module(self, module_type="solid"):
        self.module.attrib["type"] = module_type

    def setup_controls(self,
                       analysis_type="static",
                       time_steps=10,
                       step_size=0.1,
                       max_refs=15,
                       max_ups=10,
                       diverge_reform=1,
                       reform_each_time_step=1,
                       dtol=0.001,
                       etol=0.01,
                       rtol=0,
                       lstol=0.75,
                       min_residual=1e-20,
                       qnmethod=0,
                       rhoi=0,
                       dtmin=0.01,
                       dtmax=0.1,
                       max_retries=5,
                       opt_iter=10,
                       **control_settings):
        """
        Set up or replace the control settings in an FEBio .feb file.

        Args:
            analysis_type (str): Type of analysis (e.g., 'static').
            time_steps (int): Number of time steps.
            step_size (float): Time step size.
            max_refs (int): Maximum number of reformations.
            max_ups (int): Maximum number of updates.
            diverge_reform (int): Flag for divergence reform.
            reform_each_time_step (int): Flag to reform each time step.
            dtol (float): Displacement tolerance.
            etol (float): Energy tolerance.
            rtol (float): Residual tolerance.
            lstol (float): Line search tolerance.
            min_residual (float): Minimum residual.
            qnmethod (int): Quasi-Newton method.
            rhoi (int): Rhoi parameter.
            dtmin (float): Minimum time step size.
            dtmax (float): Maximum time step size.
            max_retries (int): Maximum retries.
            opt_iter (int): Optimal iterations.
            **control_settings: Additional control settings to add to the control element. Any nested elements should be passed as dictionaries.
        """
        # Clear any existing control settings
        if self.control is not None:
            self.root.remove(self.control)

        # Create new control element
        self.control  # will trigger the creation of the control element

        # Add individual settings
        settings = {
            "time_steps": time_steps,
            "step_size": step_size,
            "max_refs": max_refs,
            "max_ups": max_ups,
            "diverge_reform": diverge_reform,
            "reform_each_time_step": reform_each_time_step,
            "dtol": dtol,
            "etol": etol,
            "rtol": rtol,
            "lstol": lstol,
            "min_residual": min_residual,
            "qnmethod": qnmethod,
            "rhoi": rhoi,
            "time_stepper": {
                "dtmin": dtmin,
                "dtmax": dtmax,
                "max_retries": max_retries,
                "opt_iter": opt_iter
            },
            "analysis": {
                "type": analysis_type
            }
        }
        settings.update(control_settings)

        for key, value in settings.items():
            if key == "analysis":
                sub_element = ET.SubElement(self.control, key)
                sub_element.attrib["type"] = value["type"]
            elif isinstance(value, dict):  # handle nested elements like time_stepper and analysis
                sub_element = ET.SubElement(self.control, key)
                for subkey, subvalue in value.items():
                    subsub_element = ET.SubElement(sub_element, subkey)
                    subsub_element.text = str(subvalue)
            else:
                element = ET.SubElement(self.control, key)
                element.text = str(value)

    def setup_globals(self, T=0, R=0, Fc=0):
        """
        Set up or replace the globals settings in an FEBio .feb file.

        Args:
            T (float): Temperature constant.
            R (float): Universal gas constant.
            Fc (float): Force constant.

        Returns:
            None
        """
        # Clear any existing globals settings
        globals_tag = self.root.find("Globals")
        if globals_tag is not None:
            self.root.remove(globals_tag)

        # Create new Globals element
        globals_tag = self.globals

        # Create Constants sub-element under Globals
        constants = ET.SubElement(globals_tag, "Constants")

        # Add individual constants
        constants_dict = {
            "T": T,
            "R": R,
            "Fc": Fc
        }

        for key, value in constants_dict.items():
            element = ET.SubElement(constants, key)
            element.text = str(value)

    def setup_output(self, variables=None):
        """
        Set up or replace the output settings in an FEBio .feb file.

        Args:
            variables (list of str): List of variables to output. If None, defaults to a predefined set.

        Returns:
            None
        """
        # Default variables if none are provided
        if variables is None:
            variables = ["displacement", "element strain energy", "Lagrange strain", "stress"]

        # Clear any existing output settings
        output_tag = self.root.find("Output")
        if output_tag is not None:
            self.root.remove(output_tag)

        # Create new Output element
        output_tag = self.output  # will trigger the creation of the output element

        # Create plotfile sub-element under Output
        plotfile = ET.SubElement(output_tag, "plotfile")
        plotfile.set("type", "febio")

        # Add variables
        for var in variables:
            var_element = ET.SubElement(plotfile, "var")
            var_element.set("type", var)
