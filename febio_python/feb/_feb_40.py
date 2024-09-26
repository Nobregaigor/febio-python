from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree
import xml.etree.ElementTree as ET

from .bases import AbstractFebObject
import numpy as np
from typing import Union, List
from collections import OrderedDict, deque

from febio_python.core import (
    SURFACE_EL_TYPE,
    FEBioElementType,
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
    ZeroDisplacementCondition,
    ZeroShellDisplacementCondition,
    # FixedAxis,
    RigidBodyCondition,
    NodalData,
    SurfaceData,
    ElementData,
    GenericDomain,
    ShellDomain,
    DiscreteSet,
    DiscreteMaterial,
    DiscreteNonlinearSpringMaterial,
    RigidBodyConstraint,
    RigidBodyFixedConstraint,
)

from ._caching import feb_instance_cache


class Feb40(AbstractFebObject):
    def __init__(self,
                 tree: Union[ElementTree, None] = None,
                 root: Union[Element, None] = None,
                 filepath: Union[str, Path] = None):
        self._default_version = 4.0
        super().__init__(tree, root, filepath)
        if self.version != 4.0:
            raise ValueError("This class is only for FEBio 4.0 files"
                             f"Version found: {self.version}")

    # =========================================================================================================
    # Retrieve methods
    # =========================================================================================================

    # Main geometry data
    # ------------------------------

    @feb_instance_cache
    def get_nodes(self, dtype: np.dtype = np.float32) -> List[Nodes]:
        all_nodes: OrderedDict = self.get_tag_data(self.LEAD_TAGS.MESH, self.MAJOR_TAGS.NODES, content_type="text", dtype=dtype)
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

            if elem_mat is not None:
                elem_mat = int(elem_mat)

            connectivity = deque()
            elem_ids = deque()
            for elem in elem_group.findall("elem"):
                # Convert the comma-separated string of node indices into an array of integers
                this_elem_connectivity = np.array(elem.text.split(','), dtype=dtype)
                connectivity.append(this_elem_connectivity)
                this_elem_id = int(elem.attrib["id"])
                elem_ids.append(this_elem_id)

            # Convert the list of element connectivities to a numpy array
            connectivity = np.array(connectivity, dtype=dtype) - 1  # Correct ids to zero-based indexing
            # num_elems = connectivity.shape[0]
            elem_ids = np.array(elem_ids, dtype=np.int64) - 1  # Correct ids to zero-based indexing
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
    def get_surfaces(self, dtype=np.int64) -> List[Surfaces]:
        all_surfaces = []
        for surf_group in self.mesh.findall("Surface"):
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
            # Create an surfents instance for each surfent
            surface = Surfaces(name=surf_name,
                               mat=mat_id,
                               part=surf_part_id,
                               type=surf_type,
                               connectivity=connectivity,
                               ids=surf_ids)
            all_surfaces.append(surface)
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
        nodesets = self.mesh.findall(self.MAJOR_TAGS.NODESET.value)
        # Convert the nodesets dictionary to a list of Nodeset named tuples
        nodeset_list = list()
        for item in nodesets:
            key = item.attrib.get("name")
            value = np.fromstring(item.text, sep=",", dtype=dtype)
            # convert value text to numpy array
            value -= 1
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
        # Extract the nodesets dictionary from the .feb file
        surfacesets = self.mesh.findall(self.MAJOR_TAGS.SURFACESET.value)
        # Convert the surfacesets dictionary to a list of Nodeset named tuples
        surfset_list = list()
        for item in surfacesets:
            key = item.attrib.get("name")
            value = np.fromstring(item.text, sep=",", dtype=dtype)
            # convert value text to numpy array
            value -= 1
            surfset_list.append(SurfaceSet(name=key, ids=value))
        return surfset_list

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
        # Extract the nodesets dictionary from the .feb file
        elemesets = self.mesh.findall(self.MAJOR_TAGS.ELEMENTSET.value)
        # Convert the elemesets dictionary to a list of Nodeset named tuples
        elementset_list = list()
        for item in elemesets:
            key = item.attrib.get("name")
            value = np.fromstring(item.text, sep=",", dtype=dtype)
            # convert value text to numpy array
            value -= 1
            elementset_list.append(ElementSet(name=key, ids=value))
        return elementset_list

    @feb_instance_cache
    def get_discrete_sets(self, dtype=np.int64, find_related_nodesets=True) -> List[DiscreteSet]:

        # get all node sets by name and ids (for reference)
        node_sets_by_name = {n.name: n.ids for n in self.get_node_sets()}
        # find all discrete sets
        discrete_sets = self.mesh.findall("DiscreteSet")
        discrete_set_list = []
        for dset in discrete_sets:
            # get name
            name = dset.attrib.get("name", None)

            # Try to get the related discrete material
            mat_id = None  # default value
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

    # Mesh Domains
    # ------------------------------

    @feb_instance_cache
    def get_mesh_domains(self) -> List[Union[GenericDomain, ShellDomain]]:

        all_domains = []
        # get all child elements of MeshDomains
        for i, domain in enumerate(self.meshdomains.findall("./")):

            name = domain.attrib.get("name", f"UnnamedDomain{i}")
            mat = domain.attrib.get("mat", None)
            type = domain.attrib.get("type", None)

            if domain.tag.upper() == "SHELLDOMAIN":
                # get shell_normal_nodal child element
                shell_normal_nodal = float(domain.find("shell_normal_nodal").text)
                shell_thickness = float(domain.find("shell_thickness").text)
                new_domain = ShellDomain(
                    id=i,
                    name=name,
                    mat=mat,
                    type=type,
                    shell_normal_nodal=shell_normal_nodal,
                    shell_thickness=shell_thickness
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
            for el in list(item):

                # first, we need to check if there are sub-elements,
                # if so, then p_val is a dictionary

                if len(list(el)) > 0:
                    p_val = {}
                    for subel in list(el):
                        subel_val = subel.text
                        if subel_val is not None and str(subel_val).isdigit():
                            subel_val = int(subel_val)
                        elif subel_val is not None and str(subel_val).replace(".", "").isdigit():
                            subel_val = float(subel_val)
                        p_val[subel.tag] = subel_val
                else:
                    p_val = el.text
                    if p_val is not None and str(p_val).isdigit():
                        p_val = int(p_val)
                    elif p_val is not None and str(p_val).replace(".", "").isdigit():
                        p_val = float(p_val)

                parameters[el.tag] = p_val

            # Remove standard fields from attributes if they exist
            mat_id = mat_attrib.pop("id", None)
            # check if mat_id can be converted to int
            if mat_id is not None and str(mat_id).isdigit():
                mat_id = int(mat_id)
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
        for i, load in enumerate(self.loads.findall("nodal_load")):  # Update to find 'nodal_load' elements
            value_data = load.find("scale")

            # Convert value text to float if possible, maintain as text if not
            try:
                scale_value = float(value_data.text)
            except (ValueError, TypeError):
                scale_value = value_data.text  # Keep as text if not convertible

            # in ths new version, scale is a "tuple" -> text separated by commas
            if isinstance(scale_value, str) and ',' in scale_value:
                scale_value = tuple(map(float, scale_value.split(',')))

            # Extracting the load curve id, default to 'NoCurve'
            lc_curve = value_data.attrib.get("lc", None)
            try:
                lc_curve = int(lc_curve)
            except (ValueError, TypeError):
                pass

            # Extract the shell_bottom tag value (if it exists)
            shell_bottom = load.find("shell_bottom")
            if shell_bottom is not None:
                shell_bottom = shell_bottom.text
            try:
                shell_bottom = float(shell_bottom)
            except (ValueError, TypeError):
                pass

            # Extract relative tag value (if it exists)
            relative = load.find("relative")
            if relative is not None:
                relative = relative.text
            try:
                relative = bool(relative)
            except (ValueError, TypeError):
                pass

            # extract the "dof" tag value (if it exists)
            dof = load.find("dof")
            if dof is not None:
                dof = dof.text

            # Create a NodalLoad named tuple for the current load
            current_load = NodalLoad(
                dof=dof,
                node_set=load.attrib.get("node_set", None),
                type=load.attrib.get("type", None),
                relative=relative,
                scale=scale_value,
                load_curve=lc_curve,
                shell_bottom=shell_bottom
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
        for loadcurve_elem in self.loaddata.findall(".//load_controller"):
            load_curve_id = loadcurve_elem.attrib['id']
            if load_curve_id is not None and str(load_curve_id).isdigit():
                load_curve_id = int(load_curve_id)

            load_curve_type = loadcurve_elem.find('interpolate').text.lower()  # Adapting to 'interpolate' tag for curve type

            points = deque()

            # Extract points from each 'point' element within 'points' container
            for point_elem in loadcurve_elem.find('points').findall('pt'):
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
    def get_boundary_conditions(self) -> List[Union[ZeroDisplacementCondition,
                                                    ZeroShellDisplacementCondition,
                                                    RigidBodyCondition,
                                                    BoundaryCondition]]:
        if self.boundary is None:
            return []

        boundary_conditions_list = []
        for elem in self.boundary.findall(".//bc"):  # Update to find 'bc' elements
            if elem.attrib.get('type') == 'zero displacement':
                # Extract DOFs
                dof = {
                    "x_dof": elem.find("x_dof").text if elem.find("x_dof") is not None else "0",
                    "y_dof": elem.find("y_dof").text if elem.find("y_dof") is not None else "0",
                    "z_dof": elem.find("z_dof").text if elem.find("z_dof") is not None else "0"
                }
                # Create an instance of FixCondition for each 'zero displacement' element
                fix_condition = ZeroDisplacementCondition(
                    dof=dof,
                    node_set=elem.attrib['node_set'],
                    name=elem.attrib.get('name')
                )
                boundary_conditions_list.append(fix_condition)
            elif elem.attrib.get('type') == 'zero shell displacement':
                # Extract DOFs
                dof = {
                    "sx_dof": elem.find("sx_dof").text if elem.find("sx_dof") is not None else "0",
                    "sy_dof": elem.find("sy_dof").text if elem.find("sy_dof") is not None else "0",
                    "sz_dof": elem.find("sz_dof").text if elem.find("sz_dof") is not None else "0"
                }
                # Create an instance of FixCondition for each 'zero displacement' element
                fix_condition = ZeroShellDisplacementCondition(
                    dof=dof,
                    node_set=elem.attrib['node_set'],
                    name=elem.attrib.get('name')
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
                tags = {child.tag: child.text for child in elem}
                generic_condition = BoundaryCondition(type=elem.tag, attributes=elem.attrib, tags=tags)
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

            # Convert the lists to numpy arrays
            _this_data = np.array(_this_data, dtype=dtype)
            _these_ids = np.array(_these_ids, dtype=np.int64) - 1

            ref = data.attrib["node_set"]
            name = data.attrib["name"]
            data_type = data.attrib.get("type", None)
            if data_type is None:
                if _this_data.ndim == 1:
                    data_type = "scalar"
                elif _this_data.shape[1] == 3:
                    data_type = "vector"
                else:
                    data_type = "tensor"

            # Create a NodalData instance
            current_data = NodalData(
                node_set=ref,
                name=name,
                data=_this_data,  # Ensure data is in the correct dtype
                ids=_these_ids,  # Correct ids to zero-based indexing
                data_type=data_type,
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
                ids=np.array(_these_ids, dtype=np.int64) - 1  # Correct ids to zero-based indexing
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
            sub_element_tags = []
            for x in data.findall("elem"):
                sub_elements = [child for child in x]
                if sub_elements:
                    # Case with multiple sub-elements (e.g., 'a' and 'b')
                    stacked_data = []
                    for sub_elem in sub_elements:
                        if not sub_element_tags:
                            sub_element_tags = [sub_elem.tag for sub_elem in sub_elements]
                        if ',' in sub_elem.text:
                            # Case with a vector
                            stacked_data.append([float(num) for num in sub_elem.text.split(',')])
                        elif sub_elem.text.isdigit():
                            # Case with a scalar
                            stacked_data.append(float(sub_elem.text))
                        else:
                            # Handle non-numeric strings as is
                            stacked_data.append(sub_elem.text)
                    _this_data.append(np.stack(stacked_data, axis=0))
                else:
                    if ',' in x.text:
                        # Simple case with a single vector
                        _this_data.append([float(num) for num in x.text.split(',')])
                    elif x.text.isdigit():
                        # Simple case with a single scalar
                        _this_data.append(float(x.text))
                    else:
                        # Handle non-numeric strings as is
                        _this_data.append(x.text)
                _these_ids.append(int(x.attrib["lid"]))

            ref = data.attrib["elem_set"]
            name = data.attrib.get("name", None)
            var = data.attrib.get("var", None)
            data_type = data.attrib.get("type", None)

            # Create a ElementData instance
            current_data = ElementData(
                elem_set=ref,
                data=np.array(_this_data, dtype=dtype),  # Ensure data is in the correct dtype
                ids=np.array(_these_ids, dtype=np.int64) - 1,  # Correct ids to zero-based indexing
                name=name,
                var=var,
                data_type=data_type,
                sub_element_tags=sub_element_tags
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
        last_initial_id = existing_nodes_list[-1].ids[-1] if existing_nodes_list else 1  # when adding nodes, FEBio starts from 1

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
                try:
                    el_type = FEBioElementType(element.type)
                except ValueError:
                    el_type = FEBioElementType[element.type]

                # # first, check if it is a VTK element type. FEBioElementType names
                # # are the same as VTK element types.
                # if el_type not in FEBioElementType.__members__.values():
                #     if str(el_type).upper() in FEBioElementType.__members__.keys():
                #         el_type = FEBioElementType[el_type].value
                #     else:
                #         raise ValueError(f"Element type {el_type} is not a valid FEBio element type.")

                # Create a new Elements group if no existing one matches the name
                el_root = ET.Element("Elements")
                el_root.set("type", el_type.value)
                if element.name is not None:
                    el_root.set("name", str(element.name))
                if element.part is not None:
                    el_root.set("part", str(element.part))
                if element.mat is not None:
                    el_root.set("mat", str(element.mat))
                self.mesh.append(el_root)  # Append new "Elements" at the end of the geometry
            for i, connectivity in enumerate(element.connectivity):
                subel = ET.SubElement(el_root, "elem")
                subel.set("id", str(i + last_initial_id))
                subel.text = ",".join(map(str, connectivity + 1))  # Convert connectivity to comma-separated string

            # Update the last_initial_id for the next element group
            last_initial_id += len(element.connectivity)

    def add_surfaces(self, elements: List[Surfaces]) -> None:
        # # Filter elements by surface type
        # filtered = [elem for elem in elements if elem.type in SURFACE_EL_TYPE.__members__]
        # if len(filtered) == 0:
        #     raise ValueError("No surface elements found in the input list. Try using add_volume_elements() instead.")

        # Retrieve existing elements and determine the last element ID
        existing_surfaces_list = self.get_surfaces()
        last_surf_initial_id = existing_surfaces_list[-1].ids[-1] if existing_surfaces_list else 1

        existing_surfaces = {element.name: element for element in existing_surfaces_list}

        for surface in elements:
            # Make sure the surface type is valid, it must be a valid FEBio surface type
            # However, user can also use VTK surface types as input, but they must be
            # converted to FEBio types
            el_type = surface.type
            try:
                el_type = FEBioElementType(surface.type)
            except ValueError:
                el_type = FEBioElementType[surface.type]

            # # first, check if it is a VTK element type. FEBioElementType names
            # # are the same as VTK element types.
            # if el_type not in FEBioElementType.__members__.values():
            #     if str(el_type).upper() in FEBioElementType.__members__.keys():
            #         el_type = FEBioElementType[el_type].value
            #     else:
            #         raise TypeError(f"Element type {el_type} is not a valid FEBio element type.")

            if surface.name in existing_surfaces:
                # Append to existing Surfaces group
                el_root = self.mesh.find(f".//Surfaces[@name='{surface.name}']")
            else:

                el_root = ET.Element("Surface")
                el_root.set("name", str(surface.name))

                # Append new "Surfaces" at the end of the geometry
                self.mesh.append(el_root)

            # Add element connectivities as sub-surfaces
            for i, connectivity in enumerate(surface.connectivity):
                subel = ET.SubElement(el_root, el_type.value)  # FEBio use element type as tag name for surface surfaces
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
            nodesets (list of NodeSet): List of NodeSet namedtuples, each containing a name and node IDs.
        """
        existing_nodesets = {nodeset.name: nodeset for nodeset in self.get_node_sets()}

        for nodeset in nodesets:
            node_ids = nodeset.ids

            if nodeset.name in existing_nodesets:
                # Append to existing NodeSet element
                el_root = self.mesh.find(f".//NodeSet[@name='{nodeset.name}']")
                # Make sure that node IDs are unique (merge both old and new then remove duplicates)
                existing_ids = existing_nodesets[nodeset.name].ids
                node_ids = np.unique(np.concatenate([existing_ids, node_ids]))
            else:
                # Create a new NodeSet element if no existing one matches the name
                el_root = ET.Element("NodeSet")
                el_root.set("name", nodeset.name)
                self.mesh.append(el_root)

            # In spec 4.0, nodesets are text-elements and not sub-elements
            el_root.text = ",".join(map(str, node_ids + 1))

    def add_surface_sets(self, surfacesets: List[SurfaceSet]) -> None:
        """
        Adds surface sets to Geometry, appending to existing surface sets if they share the same name.

        Args:
            surfacesets (list of SurfaceSet): List of SurfaceSet namedtuples, each containing a name and node IDs.
        """
        existing_surfacesets = {surfset.name: surfset for surfset in self.get_surface_sets()}

        for surfset in surfacesets:
            surf_ids = surfset.ids
            if surfset.name in existing_surfacesets:
                # Append to existing SurfaceSet element
                el_root = self.mesh.find(f".//SurfaceSet[@name='{surfset.name}']")
                # Make sure that node IDs are unique (merge both old and new then remove duplicates)
                existing_ids = existing_surfacesets[surfset.name].ids
                surf_ids = np.unique(np.concatenate([existing_ids, surf_ids]))
            else:
                # Create a new SurfaceSet element if no existing one matches the name
                el_root = ET.Element("SurfaceSet")
                el_root.set("name", surfset.name)
                self.mesh.append(el_root)

            # In spec 4.0, surfacesets are text-elements and not sub-elements
            el_root.text = ",".join(map(str, surf_ids + 1))

    def add_element_sets(self, elementsets: List[ElementSet]) -> None:
        """
        Adds element sets to Geometry, appending to existing element sets if they share the same name.

        Args:
            elementsets (list of ElementSet): List of ElementSet namedtuples, each containing a name and element IDs.
        """
        existing_elementsets = {elemset.name: elemset for elemset in self.get_element_sets()}

        for elemset in elementsets:
            elem_ids = elemset.ids
            if elemset.name in existing_elementsets:
                # Append to existing ElementSet element
                el_root = self.mesh.find(f".//ElementSet[@name='{elemset.name}']")
                # Make sure that element IDs are unique (merge both old and new then remove duplicates)
                existing_ids = existing_elementsets[elemset.name].ids
                elem_ids = np.unique(np.concatenate([existing_ids, elem_ids]))
            else:
                # Create a new ElementSet element if no existing one matches the name
                el_root = ET.Element("ElementSet")
                el_root.set("name", elemset.name)
                self.mesh.append(el_root)

            # In spec 4.0, elements are text-elements and not sub-elements
            el_root.text = ",".join(map(str, elem_ids + 1))

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
                el_root = self.mesh.find(f".//DiscreteSet[@name='{dset.name}']")
                # Merge the existing element IDs with the new ones and remove duplicates
                existing_src = existing_discrete_sets[dset.name].src
                existing_dst = existing_discrete_sets[dset.name].dst
                existing_ids = np.column_stack((existing_src, existing_dst))
                src_dst = np.unique(np.concatenate((existing_ids, src_dst)))
            else:
                # Create a new DiscreteSet element if no existing one matches the name
                el_root = ET.Element("DiscreteSet")
                el_root.set("name", dset.name)
                self.mesh.append(el_root)

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

    # Mesh Domains
    # ------------------------------

    def add_mesh_domains(self, domains: List[Union[GenericDomain, ShellDomain]]):
        # Get the root for MeshDomains or create one if it does not exist
        mesh_domains = self.root.find("MeshDomains")
        if mesh_domains is None:
            mesh_domains = ET.SubElement(self.root, "MeshDomains")

        # Iterate through the domains to add
        for domain in domains:
            if isinstance(domain, ShellDomain):
                # Create a ShellDomain element with specific attributes
                domain_elem = ET.SubElement(mesh_domains, "ShellDomain")
                domain_elem.set("name", domain.name)
                domain_elem.set("mat", domain.mat)
                domain_elem.set("type", domain.type)
                # Add specific child for ShellDomain
                shell_thickness = ET.SubElement(domain_elem, "shell_thickness")
                shell_thickness.text = str(domain.shell_thickness)
                shell_normal_nodal_elem = ET.SubElement(domain_elem, "shell_normal_nodal")
                shell_normal_nodal_elem.text = str(domain.shell_normal_nodal)
            else:
                # Create a GenericDomain element with basic attributes
                ET.SubElement(mesh_domains, domain.tag_name, name=domain.name, mat=domain.mat)

    # Materials
    # ------------------------------

    def add_materials(self, materials: List[Material]) -> None:
        """
        Adds materials to Material, appending to existing materials if they share the same ID.

        Args:
            materials (list of Material): List of Material namedtuples, each containing an ID, type, parameters, name, and attributes.
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

            mat_load_curves = material.load_curve or dict()

            # Add parameters as sub-elements
            for key, value in material.parameters.items():
                subel = ET.SubElement(el_root, key)
                if isinstance(value, (str, int, float)):
                    subel.text = str(value)
                elif isinstance(value, (list, np.ndarray)):
                    # then we must add as mesh data
                    ref_data_name = f"material_{material.id}_{key}"
                    related_elem = element_by_mat.get(material.id, None)
                    # Make sure the element exists
                    if related_elem is None:
                        raise ValueError(f"Material with ID {material.id} does not exist in the geometry."
                                         "Cannot add material parameter as mesh data. Please, add the Elements first.")
                    # Make sure that data is the same length as the number of elements
                    if len(value) != related_elem.ids.size:
                        raise ValueError(f"Material parameter '{key}' must have the same length as the number of elements."
                                         f"Expected {related_elem.ids.size} values, got {len(value)} instead.")

                    # set the variable name
                    var = key if (key == "fiber") or (key == "mat_axis") else None
                    # prepare the element data
                    elem_data = ElementData(elem_set=related_elem.name,
                                            name=ref_data_name,
                                            data=value,
                                            ids=related_elem.ids,
                                            var=var)
                    # add the element data
                    self.add_element_data([elem_data])
                    # add the reference to the material
                    if key == "fiber":  # SPECIAL CASE
                        subel.attrib["type"] = "vector"
                    elif key == "mat_axis":  # SPECIAL CASE
                        subel.attrib["type"] = "vector"  # NOTE: NEED TO CHECK THIS
                    else:
                        subel.text = f"{ref_data_name}"
                        subel.attrib["type"] = "map"
                elif isinstance(value, dict):
                    # then dict key -> sub-element, values -> sub-sub-elements
                    subel = ET.SubElement(el_root, key)
                    for k, v in value.items():
                        if k.startswith("_"):  # then it is an attribute
                            subel.set(k[1:], str(v))
                        else:
                            subsubel = ET.SubElement(subel, k)
                            subsubel.text = str(v)

                # Add load curves as sub-elements
                if key in mat_load_curves:
                    lc = mat_load_curves[key]
                    if isinstance(lc, int):
                        subel.set("lc", str(lc))
                    elif isinstance(lc, LoadCurve):
                        subel.set("lc", str(lc.id))
                        self.add_load_curves([lc])
            
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
            mat_params: dict = material.parameters or dict()
            for key, value in mat_params.items():
                subel = ET.SubElement(el_root, key)
                subel.text = str(value)
        
            # special cases for discrete materials
            if isinstance(material, DiscreteNonlinearSpringMaterial) and material.type == "nonlinear spring":
                subel = ET.SubElement(el_root, "scale")
                subel.text = str(material.scale)
                subel = ET.SubElement(el_root, "measure")
                subel.text = str(material.measure)
                force_subel = ET.SubElement(el_root, "force")
                force_subel.attrib["type"] = material.force_type
                if material.force_type == "point":
                    subel = ET.SubElement(force_subel, "interpolate")
                    subel.text = str(material.interpolate)
                    subel = ET.SubElement(force_subel, "extend")
                    subel.text = str(material.extend)
                    points_elem = ET.SubElement(force_subel, "points")
                    for pt in material.points:
                        subel = ET.SubElement(points_elem, "pt")
                        subel.text = f"{pt[0]}, {pt[1]}"
            
            
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
            nodal_loads (list of NodalLoad): List of NodalLoad namedtuples, each containing a boundary condition, node set, scale, and load curve.
        """

        for load in nodal_loads:
            # Create a new NodalLoad element if no existing one matches the node set
            el_root = ET.Element("nodal_load")
            el_root.set("node_set", load.node_set)
            el_root.set("type", load.type)
            if load.name is not None:
                el_root.set("name", load.name)

            # Add data based on the type of load
            if load.type == "nodal_force":
                # add tag attributes (relative, shell_bottom)
                if load.relative is not None:
                    relative_subel = int(ET.SubElement(el_root, "relative"))
                    relative_subel.text = str(load.relative)
                if load.shell_bottom is not None:
                    shell_bottom_subel = ET.SubElement(el_root, "shell_bottom")
                    shell_bottom_subel.text = str(load.shell_bottom)

                # Now, set the "value" tag with lc-attribute and scale text-value
                value_subel = ET.SubElement(el_root, "value")

                # determine the type of value data
                if load.scale is not None and isinstance(load.scale, (tuple, np.ndarray)):
                    raise ValueError(
                        "If type='nodal_force', the 'scale' attribute must be a tuple or numpy array. "
                        f"Got type: {type(load.scale)}")
                # Check if scale is N-D array
                if isinstance(load.scale, np.ndarray):
                    if load.scale.ndim != 1:
                        raise ValueError("If type='nodal_force' and scale is a numpy array, "
                                         "Then the 'scale' attribute must be a 1-dimensional array of length 3. ")
                # check if dof is a string and in xyz
                if not isinstance(load.dof, str) or load.dof not in ['x', 'y', 'z']:
                    raise ValueError("If the 'dof' attribute is provided, it must be a string representing the DOF. "
                                     "Valid values are 'x', 'y', 'z'."
                                     "If you have a load in multiple directions, please, remove the 'dof' attribute and provide "
                                     "the scale as a tuple, or numpy array.")
                if load.scale is None:
                    value_subel.text = "0.0, 0.0, 0.0"  # No scale provided, default to 0.0
                else:  # either tuple or numpy array
                    value_subel.text = f"{load.scale[0]}, {load.scale[1]}, {load.scale[2]}"

            elif load.type == "nodal_load":
                # Set the dof tag element
                if load.dof is not None:
                    dof_subel = ET.SubElement(el_root, "dof")
                    dof_subel.text = load.dof
                else:
                    raise ValueError("If type='nodal_load', the 'dof' attribute must be provided. "
                                     "Valid values are 'x', 'y', 'z'.")
                # Now, set the "value" tag with lc-attribute and scale text-value
                scale_subel = ET.SubElement(el_root, "scale")
                # set the load curve
                scale_subel.set("lc", str(load.load_curve))

                # Add the scale data.
                if load.scale is None:
                    scale_subel.text = "1.0"
                elif isinstance(load.scale, (str, int, float, np.number)):
                    scale_subel.text = f"{load.scale}"
                elif isinstance(load.scale, np.ndarray):
                    # check if the scale has the correct shape: 1-dim or 2-dim with 1 column
                    if load.scale.ndim > 2:
                        raise ValueError("If type='nodal_load' and scale is a numpy array, "
                                            "Then the 'scale' attribute must be a 1-dimensional array or a 2-dimensional array with 1 column. ")
                    if load.scale.ndim == 2 and load.scale.shape[1] != 1:
                        raise ValueError("If type='nodal_load' and scale is a numpy array, "
                                            "Then the 'scale' attribute must be a 1-dimensional array or a 2-dimensional array with 1 column. ")
                    # convert to 1-dim array
                    load_scale = load.scale.flatten()
                    # We now need to add this as mesh data; and then reference it here
                    ref_data_name = f"nodal_load_{load.node_set}_scale"
                    scale_subel.text = f"1*{ref_data_name}"
                    # prepare the nodal data
                    nodal_data = NodalData(node_set=load.node_set,
                                           name=ref_data_name,
                                           data=load_scale,
                                           ids=np.arange(0, len(load.scale) + 1))
                    # add the nodal data
                    self.add_nodal_data([nodal_data])
                else:
                    raise ValueError("Invalid 'scale' attribute. It must be a string, number or numpy array.")

            # Append the new NodalLoad element to the 'loads' container
            self.loads.append(el_root)

    def add_surface_loads(self, pressure_loads: List[SurfaceLoad]) -> None:
        """
        Adds pressure loads to Loads, appending to existing pressure loads if they share the same surface.

        Args:
            pressure_loads (list of SurfaceLoad): List of SurfaceLoad namedtuples, each containing a surface, attributes, and multiplier.
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
            if load.load_curve is None:
                raise ValueError("If type='pressure', the 'load_curve' attribute must be provided.")
            if isinstance(load.load_curve, (str, int)):
                el_pressure.set("lc", str(load.load_curve))
            elif isinstance(load.load_curve, LoadCurve):
                el_pressure.set("lc", str(load.load_curve.id))
                self.add_load_curves([load.load_curve])
            else:
                raise ValueError("Invalid 'load_curve' attribute. It must be a string or LoadCurve instance.")
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
            el_linear.text = str(int(load.linear))  # convert boolean to int

            # add symmetric_stiffness tag with text data
            el_symmetric_stiffness = ET.SubElement(el_root, "symmetric_stiffness")
            el_symmetric_stiffness.text = str(int(load.symmetric_stiffness))

            # Append the new SurfaceLoad element to the list
            self.loads.append(el_root)

    def add_load_curves(self, load_curves: List[LoadCurve]) -> None:
        """
        Adds load curves to LoadData, appending to existing load curves if they share the same ID.

        Args:
            load_curves (list of LoadCurve): List of LoadCurve namedtuples, each containing an ID, type, and data.
        """
        existing_load_curves = {curve.id: curve for curve in self.get_load_curves()}

        for curve in load_curves:
            curve_points = curve.data  # 2-D array of points
            # try to convert to numpy array
            curve_points = np.array(curve_points)
            # Make sure that data is a 2-D array
            if curve_points.ndim != 2:
                raise ValueError(f"LoadCurve data must be a 2-dimensional array. Current shape: {curve_points.shape}")
            # Make sure that data has 2 columns
            if curve_points.shape[1] != 2:
                raise ValueError(f"LoadCurve data must have 2 columns. Current shape: {curve_points.shape}")

            # Check if the curve ID already exists
            if curve.id in existing_load_curves:
                # Find existing LoadCurve element
                el_root = self.loaddata.find(f".//load_controller[@id='{curve.id}']")
                existing_points = existing_load_curves[curve.id].data
                # add the new points to the existing points (stack vertically)
                curve_points = np.vstack([existing_points, curve_points])
            else:
                # Create a new LoadCurve element if no existing one matches the ID
                el_root = ET.Element("load_controller")
                el_root.set("id", str(curve.id))
                self.loaddata.append(el_root)

            # Set type as 'loadcurve'
            el_root.set("type", "loadcurve")
            # Add the 'interpolate' tage element
            interpolate_elem = ET.SubElement(el_root, "interpolate")
            interpolate_elem.text = curve.interpolate_type.upper()  # Ensure the correct case
            # Add the 'extend' tag element
            extend_elem = ET.SubElement(el_root, "extend")
            extend_elem.text = curve.extend.upper()  # Ensure the correct case

            # Create a 'points' container element and add the curve points
            points_elem = ET.SubElement(el_root, "points")
            for point in curve_points:
                point_elem = ET.SubElement(points_elem, "pt")
                point_elem.text = ",".join(map(str, point))

    # Boundary conditions
    # ------------------------------

    def add_boundary_conditions(self, boundary_conditions: List[BoundaryCondition]) -> None:
        """
        Adds boundary conditions to Boundary.

        Args:
            boundary_conditions (list of Union[FixCondition, RigidBodyCondition, BoundaryCondition]): List of boundary condition namedtuples.
        """
        for i, bc in enumerate(boundary_conditions):
            if not issubclass(type(bc), BoundaryCondition):
                raise ValueError(f"Boundary condition at index {i} is not a valid BoundaryCondition instance."
                                 "Input boundary conditions must be instaces of BoundaryCondition or its subclasses."
                                 "e.g. BoundaryCondition, ZeroDisplacementCondition, RigidBodyCondition.")
            # Make sure that bc has a type (e.g. type is not None)
            if bc.type is None:
                raise ValueError(f"Boundary condition at index {i} must have a valid 'type' attribute.")

            # Create a new BoundaryCondition element
            el_root = ET.Element("bc")

            # Add the FixedCondition
            if bc.type == "fix" or bc.type == "zero displacement":  # fix is used for backward compatibility
                if "s" in bc.dof.lower():
                    raise ValueError("FixCondition is no longer used in spec 4.0. "
                                     "It has been replaced by ZeroDisplacementCondition and ZeroShellDisplacementCondition. "
                                     "It is recommended to use ZeroDisplacementCondition instead of FixCondition. "
                                     "We are keeping this for backward compatibility, but it is only available for 'x', 'y', 'z' DOFs."
                                     "ZeroShellDisplacementCondition is used for shell elements, please use ZeroDisplacementCondition instead.")
                if bc.type == "fix":
                    el_root.set("type", "zero displacement")
                else:
                    el_root.set("type", bc.type)
                el_root.set("node_set", bc.node_set)
                if bc.name is None:
                    name = f"FixCondition_{i}_{bc.node_set}"
                else:
                    name = bc.name
                el_root.set("name", name)

                x_dof_elem = ET.SubElement(el_root, "x_dof")
                y_dof_elem = ET.SubElement(el_root, "y_dof")
                z_dof_elem = ET.SubElement(el_root, "z_dof")

                if "x" in bc.dof.lower():
                    x_dof_elem.text = "1"
                else:
                    x_dof_elem.text = "0"
                if "y" in bc.dof.lower():
                    y_dof_elem.text = "1"
                else:
                    y_dof_elem.text = "0"
                if "z" in bc.dof.lower():
                    z_dof_elem.text = "1"
                else:
                    z_dof_elem.text = "0"
            elif bc.type == "zero shell displacement":
                el_root.set("type", bc.type)
                el_root.set("node_set", bc.node_set)
                if bc.name is None:
                    name = f"ZeroShellDisplacementCondition_{i}_{bc.node_set}"
                else:
                    name = bc.name
                el_root.set("name", name)

                sx_dof_elem = ET.SubElement(el_root, "sx_dof")
                sy_dof_elem = ET.SubElement(el_root, "sy_dof")
                sz_dof_elem = ET.SubElement(el_root, "sz_dof")

                if "x" in bc.dof.lower():
                    sx_dof_elem.text = "1"
                else:
                    sx_dof_elem.text = "0"
                if "y" in bc.dof.lower():
                    sy_dof_elem.text = "1"
                else:
                    sy_dof_elem.text = "0"
                if "z" in bc.dof.lower():
                    sz_dof_elem.text = "1"
                else:
                    sz_dof_elem.text = "0"
            elif bc.type == "rigid body":
                el_root.set("type", "rigid_body")
                el_root.set("mat", bc.material)
                if bc.name is None:
                    name = f"RigidBodyCondition_{i}"
                else:
                    name = bc.name
                el_root.set("name", name)
                for fixed in bc.dof.split(','):
                    subel = ET.SubElement(el_root, "fixed")
                    subel.set("bc", fixed)
            else:  # General BoundaryCondition
                el_root.set("type", bc.type)
                if bc.node_set is not None:
                    el_root.set("node_set", bc.node_set)
                if bc.surf_set is not None:
                    el_root.set("surf_set", bc.surf_set)
                if bc.elem_set is not None:
                    el_root.set("elem_set", bc.elem_set)
                if bc.name is not None:
                    el_root.set("name", bc.name)
                if bc.material is not None:
                    el_root.set("mat", str(bc.material))
                if bc.dof is not None:
                    split_dof = bc.dof.split(',')
                    for dof_type in split_dof:
                        subel = ET.SubElement(el_root, f"{dof_type}_dof")
                        subel.text = "1"
                if bc.attributes is not None:
                    for key, value in bc.attributes.items():
                        el_root.set(key, str(value))
                if bc.tags is not None:
                    for key, value in bc.tags.items():
                        subel = ET.SubElement(el_root, key)
                        subel.text = str(value)

            self.boundary.append(el_root)

    # Rigid Bodies
    # ------------------------------
    
    def add_rigid_constraints(self, rigid_constraints: List[RigidBodyConstraint]) -> None:
        for i, rc in enumerate(rigid_constraints):
            if not issubclass(type(rc), RigidBodyConstraint):
                raise TypeError(f"Rigid body constraint at index {i} is not a valid RigidBodyConstraint instance.")
            # Make sure that rc has a type (e.g. type is not None)
            if rc.type is None:
                raise ValueError(f"Rigid body condition at index {i} must have a valid 'type' attribute.")
            # Make sure that rc has a body material (e.g. body is not None)
            if rc.body is None:
                raise ValueError(f"Rigid body condition at index {i} must have a valid 'body' attribute.")

            # Create a new BoundaryCondition element
            el_root = ET.Element("rigid_bc")
            
            # add name and type as attributes
            el_root.set("type", str(rc.type))
            el_root.set("type", str(rc.name))
            
            # create sub-element for the body
            body_elem = ET.SubElement(el_root, "rb")
            body_elem.text = str(rc.body)
            
            # Now for specific types of rigid body constraints
            if isinstance(rc, RigidBodyFixedConstraint) and rc.type == "rigid_fixed":
                el_root.attrib["type"] = "rigid_fixed"
                # We have possible constraints for the rigid body
                # check for each of them individually:
                dof = str(rc.dof).lower()
                # x y z - translation
                x_dof_elem = ET.SubElement(el_root, "Rx_dof")
                if "x" in dof:
                    x_dof_elem.text = "1"
                else:
                    x_dof_elem.text = "0"
                y_dof_elem = ET.SubElement(el_root, "Ry_dof")
                if "y" in dof:
                    y_dof_elem.text = "1"
                else:
                    y_dof_elem.text = "0"
                z_dof_elem = ET.SubElement(el_root, "Rz_dof")
                if "z" in dof:
                    z_dof_elem.text = "1"
                else:
                    z_dof_elem.text = "0"
                # y v w - rotation
                u_dof_elem = ET.SubElement(el_root, "Ru_dof")
                if "u" in dof:
                    u_dof_elem.text = "1"
                else:
                    u_dof_elem.text = "0"
                v_dof_elem = ET.SubElement(el_root, "Rv_dof")
                if "v" in dof:
                    v_dof_elem.text = "1"
                else:
                    v_dof_elem.text = "0"
                w_dof_elem = ET.SubElement(el_root, "Rw_dof")
                if "w" in dof:
                    w_dof_elem.text = "1"
                else:
                    w_dof_elem.text = "0"
            else:
                raise RuntimeError(f"Rigid body constraint of type {rc.type} is not yet implemented or not valid.")

            # Append the new RigidBodyConstraint element to the list
            self.rigid.append(el_root)

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
            surface_data (list of SurfaceData): List of SurfaceData namedtuples, each containing a surface set, name, and data.
        """
        existing_surface_data = {data.surf_set: data for data in self.get_surface_data()}

        for data in surface_data:
            if data.surf_set in existing_surface_data:
                # Append to existing SurfaceData element
                el_root = self.meshdata.find(f".//{self.MAJOR_TAGS.SURFACE_DATA.value}[@surf_set='{data.surf_set}']")
            else:
                # Create a new SurfaceData element if no existing one matches the surface set
                el_root = ET.Element(self.MAJOR_TAGS.SURFACE_DATA.value)
                el_root.set("surf_set", data.surf_set)
                el_root.set("name", data.name)
                self.meshdata.append(el_root)

            for i, surf_data in enumerate(data.data):
                subel = ET.SubElement(el_root, "surf")
                subel.set("lid", str(data.ids[i] + 1))  # Convert to one-based indexing
                if isinstance(surf_data, (str, int, float, np.number)):
                    subel.text = str(surf_data)
                else:
                    try:
                        subel.text = ",".join(map(str, surf_data))
                    except TypeError:
                        raise ValueError(f"Node data for node set {data.surf_set} is not in the correct format.")

    def add_element_data(self, element_data: List[ElementData]) -> None:
        """
        Adds element data to MeshData, appending to existing element data if they share the same element set.

        Args:
            element_data (list of ElementData): List of ElementData namedtuples, each containing an element set, name, and data.
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
            if data.name is None and data.var is None:
                raise ValueError("ElementData must have either a name or var attribute.")
            if data.name is not None:
                el_root.set("name", data.name)
            if data.var is not None:
                el_root.set("type", data.var)  # IN SPEC 4.0, the type attribute is used for the var attribute
            
            self.meshdata.append(el_root)

            if data.data.ndim == 1 or data.data.ndim == 2:
                for i, elem_data in enumerate(data.data):
                    subel = ET.SubElement(el_root, "elem")
                    subel.set("lid", str(data.ids[i] + 1))  # Convert to one-based indexing
                    if isinstance(elem_data, (str, int, float, np.number)):
                        subel.text = str(elem_data)
                    else:
                        try:
                            subel.text = ",".join(map(str, elem_data))
                        except TypeError:
                            raise ValueError(f"Node data for node set {data.elem_set} is not in the correct format.")
            elif data.data.ndim == 3:
                if data.sub_element_tags is None:
                    raise ValueError("ElementData with 3-dimensional data must have a 'sub_element_tags' attribute.")
                if len(data.sub_element_tags) != data.data.shape[1]:
                    raise ValueError(f"The number of sub-element tags ({len(data.sub_element_tags)}) must "
                                     f"match second axis of the data ({data.data.shape[1]}).")
                
                for i, elem_data in enumerate(data.data):
                    # Create a new element sub-element
                    subel = ET.SubElement(el_root, "elem")
                    subel.set("lid", str(data.ids[i] + 1))  # Convert to one-based indexing
                    # Add the sub-element tags
                    for j, sub_elem_data in enumerate(elem_data):
                        subsubel = ET.SubElement(subel, data.sub_element_tags[j])
                        if isinstance(elem_data, (str, int, float, np.number)):
                            subsubel.text = str(sub_elem_data)
                        else:
                            try:
                                subsubel.text = ",".join(map(str, sub_elem_data))
                            except TypeError:
                                raise ValueError(f"Element data for elment {data.elem_set} is not in the correct format.")
                
                
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

    def remove_node_sets(self, names: List[str]) -> None:
        """
        Removes node sets from Geometry by name.

        Args:
            names (list of str): List of node set names to remove.
        """
        for name in names:
            el = self.mesh.find(f".//NodeSet[@name='{name}']")
            if el is not None:
                self.mesh.remove(el)

    def remove_surface_sets(self, names: List[str]) -> None:
        """
        Removes surface sets from Geometry by name.

        Args:
            names (list of str): List of surface set names to remove.
        """
        for name in names:
            el = self.mesh.find(f".//SurfaceSet[@name='{name}']")
            if el is not None:
                self.mesh.remove(el)

    def remove_element_sets(self, names: List[str]) -> None:
        """
        Removes element sets from Geometry by name.

        Args:
            names (list of str): List of element set names to remove.
        """
        for name in names:
            el = self.mesh.find(f".//ElementSet[@name='{name}']")
            if el is not None:
                self.mesh.remove(el)

    # Mesh Domains
    # ------------------------------

    def remove_mesh_domains(self, identifiers: List[str]) -> None:
        """
        Removes domains from MeshDomains by id, name, or material.

        Args:
            identifiers (list of str): List of ids, names, or materials to remove.
        """
        mesh_domains = self.root.find("MeshDomains")
        if mesh_domains is None:
            return  # If there are no MeshDomains, nothing needs to be removed.

        # Iterate through each identifier to remove domains
        for identifier in identifiers:
            # Remove by name or mat attribute
            els_by_name_or_mat = mesh_domains.findall(f"./*[@name='{identifier}' or @mat='{identifier}']")
            for el in els_by_name_or_mat:
                mesh_domains.remove(el)

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

    def remove_boundary_conditions(self, types: List[str], names: List[str] = None) -> None:
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

    def clear_node_sets(self) -> None:
        """
        Removes all node sets from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.NODESET.value):
            self.mesh.remove(el)

    def clear_surface_sets(self) -> None:
        """
        Removes all surface sets from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.SURFACESET.value):
            self.mesh.remove(el)

    def clear_element_sets(self) -> None:
        """
        Removes all element sets from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.ELEMENTSET.value):
            self.mesh.remove(el)

    def clear_discrete_sets(self) -> None:
        """
        Removes all discrete sets from Geometry.
        """
        for el in self.mesh.findall(self.MAJOR_TAGS.DISCRETESET.value):
            self.mesh.remove(el)
        for el in self.discrete.findall(self.MAJOR_TAGS.DISCRETE.value):
            self.discrete.remove(el)

    def clear_mesh_domains(self) -> None:
        """
        Removes all mesh domains from MeshDomains.
        """
        for el in self.root.findall(self.LEAD_TAGS.MESHDOMAINS.value):
            self.root.remove(el)

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

    # Mesh Domains
    # ------------------------------

    def update_mesh_domains(self, domains: List[Union[GenericDomain, ShellDomain]]) -> None:
        """
        Updates mesh domains in MeshDomains by name, replacing existing mesh domains with the same name.

        Args:
            domains (list of Union[GenericDomain, ShellDomain]): List of domain namedtuples.
        """
        self.remove_mesh_domains([domain.name for domain in domains])
        self.add_mesh_domains(domains)

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

    def update_load_curves(self, load_curves: List[LoadCurve]) -> None:
        """
        Updates load curves in LoadData by ID, replacing existing load curves with the same ID.

        Args:
            load_curves (list of LoadCurve): List of LoadCurve namedtuples, each containing an ID, type, and data.
        """
        self.remove_load_curves([curve.id for curve in load_curves])
        self.add_load_curves(load_curves)

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

    # =========================================================================================================
    # Base control setup
    # =========================================================================================================

    def setup_module(self, module_type="solid"):
        self.module.attrib["type"] = module_type

    def setup_controls(self,
                       analysis="STATIC",
                       time_steps=10,
                       step_size=0.1,
                       plot_zero_state: bool = True,
                       plot_range=(0, 1),
                       plot_level="PLOT_MAJOR_ITRS",
                       output_level="OUTPUT_MAJOR_ITRS",
                       plot_stride=1,
                       output_stride=1,
                       adaptor_re_solve=1,
                       # solver settings
                       solver_type="solid",
                       symmetric_stiffness="symmetric",
                       equation_scheme="staggered",
                       equation_order="default",
                       optimize_bw=0,
                       lstol=0.75,
                       lsmin=0.01,
                       lsiter=5,
                       max_refs=15,
                       check_zero_diagonal=0,
                       zero_diagonal_tol=0,
                       force_partition=0,
                       reform_each_time_step=1,
                       reform_augment=0,
                       diverge_reform=1,
                       min_residual=1e-20,
                       max_residual=0,
                       dtol=0.001,
                       etol=0.01,
                       rtol=0,
                       alpha=1,
                       beta=0.25,
                       gamma=0.5,
                       logSolve=0,
                       arc_length=0,
                       arc_length_scale=0,
                       qn_method_type="BFGS",
                       max_ups=10,
                       max_buffer_size=0,
                       cycle_buffer=1,
                       cmax=100000,
                       qnmethod=0,
                       rhoi=0,
                       # time stepper settings
                       time_stepper_type="default",
                       max_retries=5,
                       opt_iter=10,
                       dtmin=0.01,
                       dtmax=0.1,
                       aggressiveness=0,
                       cutback=0.5,
                       dtforce=0,
                       **control_settings):
        """
        Set up control settings in an FEBio .feb file.

        Args:
            analysis (str): Analysis type.
            time_steps (int): Number of time steps.
            step_size (float): Size of each time step.
            plot_zero_state (bool): Whether to plot zero state.
            plot_range (tuple): Range of plot.
            plot_level (str): Level of plot.
            output_level (str): Level of output.
            plot_stride (int): Plot stride.
            output_stride (int): Output stride.
            adaptor_re_solve (int): Adaptor resolve setting.
            solver_type (str): Type of solver.
            symmetric_stiffness (str): Symmetric stiffness setting.
            equation_scheme (str): Equation scheme.
            equation_order (str): Equation order.
            optimize_bw (int): Optimize bandwidth.
            lstol (float): Line search tolerance.
            lsmin (float): Minimum line search.
            lsiter (int): Line search iterations.
            max_refs (int): Maximum refinements.
            check_zero_diagonal (int): Check zero diagonal.
            zero_diagonal_tol (int): Zero diagonal tolerance.
            force_partition (int): Force partition.
            reform_each_time_step (int): Reform each time step.
            reform_augment (int): Reform augment.
            diverge_reform (int): Diverge reform.
            min_residual (float): Minimum residual.
            max_residual (int): Maximum residual.
            dtol (float): Displacement tolerance.
            etol (float): Energy tolerance.
            rtol (int): Residual tolerance.
            alpha (int): Alpha value.
            beta (float): Beta value.
            gamma (float): Gamma value.
            logSolve (int): Log solve setting.
            arc_length (int): Arc length.
            arc_length_scale (int): Arc length scale.
            qn_method_type (str): Quasi-Newton method type.
            max_ups (int): Maximum ups.
            max_buffer_size (int): Maximum buffer size.
            cycle_buffer (int): Cycle buffer.
            cmax (int): Maximum cycles.
            qnmethod (int): Quasi-Newton method.
            rhoi (int): Rho I.
            time_stepper_type (str): Time stepper type.
            max_retries (int): Maximum retries.
            opt_iter (int): Optimizer iterations.
            dtmin (float): Minimum time step.
            dtmax (float): Maximum time step.
            aggressiveness (int): Aggressiveness level.
            cutback (float): Cutback value.
            dtforce (int): Force time step.
            control_settings (dict): Additional control settings.
        """
        # Clear any existing control settings
        if self.control is not None:
            self.root.remove(self.control)

        # Create new control element
        self.control  # will trigger the creation of the control element

        # Add individual settings
        # Add individual settings
        
        dtmax_lc_id = None
        if isinstance(plot_level, LoadCurve):
            self.add_load_curves([plot_level])
            dtmax_lc_id = plot_level.id
            plot_level = "PLOT_MUST_POINTS"
            
            
        settings = {
            "analysis": analysis,
            "time_steps": time_steps,
            "step_size": step_size,
            "plot_zero_state": int(plot_zero_state),
            "plot_range": f"{plot_range[0]},{plot_range[1]}",
            "plot_level": plot_level,
            "output_level": output_level,
            "plot_stride": plot_stride,
            "output_stride": output_stride,
            "adaptor_re_solve": adaptor_re_solve,
            "solver": {
                "_type": solver_type,
                "symmetric_stiffness": symmetric_stiffness,
                "equation_scheme": equation_scheme,
                "equation_order": equation_order,
                "optimize_bw": optimize_bw,
                "lstol": lstol,
                "lsmin": lsmin,
                "lsiter": lsiter,
                "max_refs": max_refs,
                "check_zero_diagonal": check_zero_diagonal,
                "zero_diagonal_tol": zero_diagonal_tol,
                "force_partition": force_partition,
                "reform_each_time_step": reform_each_time_step,
                "reform_augment": reform_augment,
                "diverge_reform": diverge_reform,
                "min_residual": min_residual,
                "max_residual": max_residual,
                "dtol": dtol,
                "etol": etol,
                "rtol": rtol,
                "rhoi": rhoi,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "logSolve": logSolve,
                "arc_length": arc_length,
                "arc_length_scale": arc_length_scale,
                "qn_method": {
                    "_type": qn_method_type,
                    "max_ups": max_ups,
                    "max_buffer_size": max_buffer_size,
                    "cycle_buffer": cycle_buffer,
                    "cmax": cmax,
                },
            },
            "time_stepper": {
                "_type": time_stepper_type,
                "max_retries": max_retries,
                "opt_iter": opt_iter,
                "dtmin": dtmin,
                "dtmax": dtmax,
                "aggressiveness": aggressiveness,
                "cutback": cutback,
                "dtforce": dtforce,
            }
        }
        settings.update(control_settings)

        for key, value in settings.items():
            if isinstance(value, dict):  # handle nested elements like time_stepper and analysis
                sub_element = ET.SubElement(self.control, key)
                for subkey, subvalue in value.items():
                    if subkey.startswith("_"):
                        sub_element.set(subkey[1:], str(subvalue))
                    else:
                        if isinstance(subvalue, dict):
                            print(subkey, subvalue)
                            subsub_element = ET.SubElement(sub_element, subkey)
                            for subsubkey, subsubvalue in subvalue.items():
                                if subsubkey.startswith("_"):
                                    subsub_element.set(subsubkey[1:], str(subsubvalue))
                                else:
                                    subsubsub_element = ET.SubElement(subsub_element, subsubkey)
                                    subsubsub_element.text = str(subsubvalue)
                                    if dtmax_lc_id is not None and subsubkey == "dtmax":
                                        subsubsub_element.set("lc", str(dtmax_lc_id))
                        else:
                            subsub_element = ET.SubElement(sub_element, subkey)
                            subsub_element.text = str(subvalue)
                            if dtmax_lc_id is not None and subkey == "dtmax":
                                subsub_element.set("lc", str(dtmax_lc_id))
            else:
                element = ET.SubElement(self.control, key)
                element.text = str(value)

    def setup_globals(self, T=0, P=0, R=0, Fc=0):
        """
        Set up or replace the globals settings in an FEBio .feb file.

        Args:
            T (float): Temperature constant.
            P (float): Pressure constant.
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
            "P": P,
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
