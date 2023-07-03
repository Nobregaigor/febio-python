import warnings
import numpy as np
import xml.etree.ElementTree as ET
from .FEBio_xml_handler import FEBio_xml_handler
from .enums import *


class FEBio_feb(FEBio_xml_handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ======================================
    # retrieve data from geometry content

    # ---
    # wrappers for 'get_content_from_repeated_tags'

    def get_nodes(self, dtype: np.dtype = np.float32) -> dict:
        """
          Returns nodes content from .feb file as a dict with keys representing nodes names\
          and values representing [x,y,z] as a numpy array of specified dtype.\
          It 'name' is not found in nodes, it save as the corresponding index.

          Args:
            dtype (np.dtype): Numpy dtype.

            Returns:
                dict: {tag_name: [x,y,z]}

            Example:
                feb.get_nodes()
        """
        return self.get_content_from_repeated_tags(FEB_LEAD_TAGS.GEOMETRY, "Nodes", dtype=dtype)

    def get_elements(self, dtype: np.dtype = np.int64) -> dict:
        """
          Returns elements content from .feb file as a dict with keys representing elements names\
          and values representing node ids as a numpy array of specified dtype.\
          It 'name' is not found in elements, it save as the corresponding index.

          Args:
            dtype (np.dtype): Numpy dtype.

            Returns:
                dict: {tag_name: [node_ids]}

            Example:
                feb.get_elements()
        """
        return self.get_content_from_repeated_tags(FEB_LEAD_TAGS.GEOMETRY, "Elements", dtype=dtype)

    def get_surfaces(self, dtype: np.dtype = np.int64) -> dict:
        """
          Returns surface content from .feb file as a dict with keys representing surface names\
          and values representing node ids as a numpy array of specified dtype.\
          It 'name' is not found in surface, it save as the corresponding index.

          Args:
            dtype (np.dtype): Numpy dtype.

            Returns:
                dict: {tag_name: [node_ids]}

            Example:
                feb.get_surfaces()
        """
        try:
            return self.get_content_from_repeated_tags(FEB_LEAD_TAGS.GEOMETRY, "Surface", dtype=dtype)
        except:
            return {}

    def get_element_data(self, dtype: np.dtype = np.float32) -> dict:
        try:
            return self.get_content_from_repeated_tags(self.LEAD_TAGS.MESHDATA, "ElementData", dtype=dtype)
        except KeyError:
            return {}

    def get_surface_data(self, dtype: np.dtype = np.float32) -> dict:
        try:
            return self.get_content_from_repeated_tags(self.LEAD_TAGS.MESHDATA, "SurfaceData", dtype=dtype)
        except KeyError:
            return {}

    def get_loadcurves(self, dtype: np.dtype = np.float32) -> dict:
        try:
            return self.get_content_from_repeated_tags(self.LEAD_TAGS.LOADDATA, "loadcurve", dtype=dtype)
        except:
            return {}

    # ---
    # wrappers for 'get_ids_from_repeated_tags'

    def get_nodesets(self, dtype=np.int64) -> dict:
        """
          Returns a dict with keys representing node set names and values \
          representing corresponding node ids as a numpy array of specified dtype.\
          It 'name' is not found in nodes, it save as the corresponding index.

          Args:
            dtype (np.dtype): Numpy dtype.

            Returns:
                dict: {tag_name: [node_ids]}

            Example:
                feb.get_nodesets()
        """
        try:
            return self.get_ids_from_repeated_tags(FEB_LEAD_TAGS.GEOMETRY, "NodeSet", dtype=dtype)
        except:
            return {}

    # ---
    # other wrappers
    def get_materials(self) -> dict:
        all_mat_data = {}
        for i, item in enumerate(self.material().findall("material")):
            mat_data = item.attrib
            parameters = {}
            for j, el in enumerate(item.iter()):
                if j == 0:
                    continue
                try:
                    p_val = float(el.text)
                except:
                    p_val = el.text
                parameters[el.tag] = p_val
            mat_data["parameters"] = parameters
            mat_key = mat_data["name"] if "name" in mat_data else mat_data["id"]
            all_mat_data[mat_key] = mat_data
        return all_mat_data

    def get_pressure_loads(self) -> dict:
        press_loads = {}
        for i, load in enumerate(self.loads().findall("surface_load")):
            press = load.find("pressure")
            if press is not None:
                load_info = load.attrib
                press_info = press.attrib
                try:
                    press_mult = float(press.text)
                except:
                    press_mult = press.text
                press_info["multiplier"] = press_mult
                press_loads[load_info["surface"]] = press_info
        return press_loads

    def get_boundary_conditions(self) -> dict:
        if self.boundary() is None:
            return dict()
        else:
            bc_data = dict()
            for elem in self.boundary():
                if elem.tag in bc_data:
                    bc_data[elem.tag].append(elem.attrib)
                else:
                    bc_data[elem.tag] = [elem.attrib]
        return bc_data

    # ======================================
    # export data as dict

    def to_dict(self):
        data = {}
        # get mesh data
        data["NODES"] = self.get_nodes()
        data["ELEMENTS"] = self.get_elements()
        # get mesh elements (nodesets and surfaces)
        data["NODESETS"] = self.get_nodesets()
        data["SURFACES"] = self.get_surfaces()
        # get additional mesh data
        data["ELEMENT_DATA"] = self.get_element_data()
        data["SURFACE_DATA"] = self.get_surface_data()
        # get material values
        data["MATERIALS"] = self.get_materials()
        # get loads (for now we only support pressure loads)
        data["PRESSURE_LOADS"] = self.get_pressure_loads()
        # get load curves from data
        data["LOAD_CURVES"] = self.get_loadcurves()
        # get boundary conditions from data
        data["BOUNDARY_CONDITIONS"] = self.get_boundary_conditions()
        return data

    # ============================
    # Add content to feb file

    def add_nodes(self, nodes: list, initial_el_id: int = 1) -> None:
        """
          Adds nodes to Geometry
        """

        last_initial_id = initial_el_id
        for (node_elem) in nodes:
            if "nodes" not in node_elem:
                raise ValueError(
                    "Nodes not found for one of the node_elem. Each node_elem should have a 'nodes' attribute.")

            el_root = ET.Element("Nodes")
            if "name" in node_elem:
                el_root.set("name", node_elem["name"])

            for i, node_xyz in enumerate(node_elem["nodes"]):
                if len(node_xyz) != 3:  # check for correct node information.
                    raise ValueError(
                        "Node '{}' does not have the correct number of coordinates. It should contain [x,y,z] values.")
                subel = ET.SubElement(el_root, "node")
                subel.set("id", str(i + last_initial_id))
                subel.text = ",".join(map(str, node_xyz))

            last_initial_id = last_initial_id + i + 1

            self.geometry().extend([el_root])

    def add_elements(self, elements: list, initial_el_id: int = 1) -> None:
        """
            Adds elements to Geometry
        """

        last_initial_id = initial_el_id
        for elem_data in elements:
            el_root = ET.Element("Elements")
            el_root.set("name", elem_data["name"])
            if "type" not in elem_data:
                eltype = self.get_element_type(len(elem_data["elems"][0]))
            else:
                eltype = elem_data["type"]
            el_root.set("type", eltype)

            if "mat" in elem_data:
                el_root.set("mat", elem_data["mat"])

            for i, elem in enumerate(elem_data["elems"]):
                subel = ET.SubElement(el_root, "elem")
                subel.set("id", str(i + last_initial_id))
                subel.text = ",".join(map(str, elem))
            last_initial_id = last_initial_id + i + 1

            self.geometry().extend([el_root])

    def add_nodesets(self, nodesets: list) -> None:
        """
          Adds nodesets to Geometry tree element.
          'nodesets' should be a dictionary with keys representing nodeset name
          and values representing nodes of such nodeset.
        """

        for (name, nodes) in nodesets.items():
            nodeset = ET.Element("NodeSet")
            nodeset.set("name", name)
            for node_id in nodes:
                subel = ET.SubElement(nodeset, "node")
                subel.set("id", str(node_id))

            self.geometry().extend([nodeset])

    def add_surfaces(self, nodesets: list, initial_el_id: int = 1) -> None:
        """
          Adds surfaces to Geometry tree element.
        """

        last_initial_id = initial_el_id
        for (name, nodes) in nodesets.items():
            nodeset = ET.Element("Surface")
            nodeset.set("name", name)
            for i, node_ids in enumerate(nodes):
                n_ids = len(node_ids)
                try:
                    subel_type = SURFACE_EL_TYPE(n_ids).name
                except:
                    raise NotImplementedError(
                        "We only SURFACE_EL_TYPE enum values. Check enums for details.")
                subel = ET.SubElement(nodeset, subel_type)
                subel.set("id", str(i+last_initial_id))
                subel.text = ",".join(map(str, node_ids))
            last_initial_id = last_initial_id + i + 1
            self.geometry().extend([nodeset])

    def add_discretesets(self, discretesets: list) -> None:

        for (name, delems) in discretesets.items():
            newset = ET.Element("DiscreteSet")
            newset.set("name", name)
            for nodes_ids in delems:
                subel = ET.SubElement(newset, "delem")
                if isinstance(nodes_ids, str):
                    subel.text = nodes_ids
                else:
                    subel.text = ",".join(map(str, nodes_ids))

            self.geometry().extend([newset])

    def add_boundary_conditions(self, bcs: list) -> None:
        """
          Adds boundary conditions to Boundary tree element.
        """

        for (bc_type, bc, nodeset) in bcs:
            boundary = ET.Element(bc_type)
            boundary.set("bc", bc)
            boundary.set("node_set", nodeset)
            self.boundary().extend([boundary])

    def add_surface_loads(self, surface_loads: list) -> None:
        """Adds surface load to Loads tag

        Args:
            surface_loads (list): _description_
        """
        loads_root = self.loads()
        loads_to_add = []
        for new_load in surface_loads:
            load_element = ET.Element("surface_load")
            load_element.set("type", str(new_load["type"]))
            load_element.set("surface", str(new_load["surface"]))

            subel = ET.SubElement(load_element, str(new_load["type"]))
            subel.set("lc", str(new_load["lc"]))
            if "multiplier" in new_load:
                subel.text = new_load["multiplier"]

            # subel = ET.SubElement(load_element, "linear")
            # subel.text = "0"
            if "surface_data" in new_load:
                # subel = ET.SubElement(load_element, str("value"))
                # subel.set("surface_data", str(new_load["surface_data"]))

                if "multiplier" in new_load:
                    subel.set("type", "math")
                    subel.text = "{}*{}".format(
                        new_load["multiplier"], str(new_load["surface_data"]))
                else:
                    subel.set("type", "map")
                    subel.text = str(new_load["surface_data"])

                # subel.set("map", str(new_load["surface_data"]))

            subel = ET.SubElement(load_element, "symmetric_stiffness")
            subel.text = "1"

            loads_to_add.append(load_element)

        loads_root.extend(loads_to_add)

    # --- this function is ugly -> need to be improved (but works for now)
    def add_meshdata(self, mesh_data: list, initial_el_id: int = 1) -> None:
        """
          Adds meshdata to file
        """

        for elem_data in mesh_data:
            if not "type" in elem_data:  # assume it is element data
                mesh_data_type = "ElementData"
            elif "type" in elem_data:
                mesh_data_type = elem_data["type"]

            if mesh_data_type == "ElementData":
                el_root = ET.Element("ElementData")
                el_root.set("elem_set", elem_data["elem_set"])
                el_root.set("var", elem_data["var"])

                elems = elem_data["elems"]
                el_keys = list(elems.keys())
                n_elems = len(elems[el_keys[0]])
                for i in range(n_elems):
                    subel = ET.SubElement(el_root, "elem")
                    subel.set("lid", str(i + initial_el_id))
                    for k in el_keys:
                        subel_2 = ET.SubElement(subel, k)
                        subel_2.text = ",".join(map(str, elems[k][i]))

            elif mesh_data_type == "SurfaceData":
                el_root = ET.Element("SurfaceData")
                el_root.set("surface", elem_data["surface"])
                el_root.set("name", elem_data["name"])

                elems = elem_data["faces"]

                if isinstance(elems, (dict)):
                    el_keys = list(elems.keys())
                    n_elems = len(elems[el_keys[0]])
                    for i in range(n_elems):
                        subel = ET.SubElement(el_root, "face")
                        subel.set("lid", str(i + initial_el_id))
                        for k in el_keys:
                            subel_2 = ET.SubElement(subel, k)
                            subel_2.text = ",".join(map(str, elems[k][i]))
                elif isinstance(elems, (list, np.ndarray)):
                    n_elems = len(elems)
                    for i in range(n_elems):
                        c_el = elems[i]
                        subel = ET.SubElement(el_root, "face")
                        subel.set("lid", str(i + initial_el_id))
                        subel.text = ",".join(map(str, elems[i]))

            else:
                raise ValueError(
                    "We currently only support ElementData and SurfaceData.")

            self.meshdata().extend([el_root])

    # ===========================
    # Modify content from feb file

    def replace_material_params(self, params: dict) -> None:
        materials = self.material().findall("material")
        for mat_ref, mat_params in params.items():
            # search materials
            for mat in materials:
                # find material reference
                if mat_ref == mat.attrib["name"] or mat_ref == mat.attrib["id"] or mat_ref == mat.attrib["type"]:
                    # modify params
                    for param_key, param_value in mat_params.items():
                        mat_elem = mat.find(param_key)
                        if mat_elem is not None:
                            mat_elem.text = str(param_value)

    def replace_nodes(self, nodes: list, initial_el_id: int = 1):
        """
          Replaces nodes elements from .feb_file with new set of nodes.\
          Check 'add_nodes' for more details.\
          * warning: it does not modify any other part of the feb file,
                     thus, keep in mind the new set of nodes should still
                     represent the same geometry data (nodesets, elementsets, 
                     etc)
        """

        # get nodes tree from existing nodes data in feb
        root = self.geometry().find("Nodes")
        nodes_tree = root.findall("node")

        # raise warning if length of new nodes do not match existing nodes
        if len(nodes_tree) != len(nodes):
            warnings.warn("Replacing nodes data with new amound of nodes.\
                    This can lead to malfunction in feb file exectuion")

        # clear nodes content
        root.clear()

        # add new nodes
        self.add_nodes(nodes, initial_el_id)

        # add new nodes
        # for i, node_xyz in enumerate(nodes):
        #     subel = ET.SubElement(root, "node")
        #     subel.set("id", str(i + 1))
        #     subel.text = ",".join(map(str, node_xyz))

    # ============================
    # Other (not fully tested)

    def create_linear_spring_discresetset(self, ref_node_id, nodes):
        return [(node_id, ref_node_id) for node_id in nodes]

    def add_linear_spring(self, E, name, discreteset):
        # discreteset = self.create_linear_spring_discresetset(ref_node_id, nodes)
        self.add_discretesets([(name, discreteset)])

        for (name, delems) in discreteset.items():
            newset = ET.Element("DiscreteSet")
            newset.set("name", name)
            for nodes_ids in delems:
                subel = ET.SubElement(newset, "delem")
                if isinstance(nodes_ids, str):
                    subel.text = nodes_ids
                else:
                    subel.text = ",".join(map(str, nodes_ids))

            self.geometry().extend([newset])
