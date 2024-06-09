import warnings
import numpy as np
import xml.etree.ElementTree as ET
from .FEBio_xml_handler import FEBio_xml_handler
from febio_python.core.enums import *


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
        for item in self.material().findall("material"):
            # Create a deep copy of the item attributes if necessary
            mat_data = dict(item.attrib)
            
            # Extract parameters
            parameters = {}
            for el in list(item)[1:]:  # Skip the first element which is the material itself
                try:
                    p_val = float(el.text)
                except ValueError:
                    p_val = el.text
                parameters[el.tag] = p_val

            # Add parameters to mat_data, but be cautious about how this is used later
            mat_data["parameters"] = parameters

            # Determine the key for the all_mat_data dictionary
            mat_key = mat_data.get("name", mat_data.get("id"))
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

    # def get_traction_loads(self) -> dict:
    #     press_loads = {}
    #     for i, load in enumerate(self.loads().findall("traction_load")):
    #         press = load.find("pressure")
    #         if press is not None:
    #             load_info = load.attrib
    #             press_info = press.attrib
    #             try:
    #                 press_mult = float(press.text)
    #             except:
    #                 press_mult = press.text
    #             press_info["multiplier"] = press_mult
    #             press_loads[load_info["surface"]] = press_info
    #     return press_loads

    def get_nodal_loads(self) -> list:
        nodal_loads = list()
        for i, load in enumerate(self.loads().findall("nodal_load")):
            scale_data = load.find("scale")
            nodal_loads.append(
                {
                    "bc": load.attrib["bc"],
                    "node_set": load.attrib["node_set"],
                    "scale": scale_data.text,
                    "load_curve": scale_data.attrib["lc"]
                }
            )
        return nodal_loads

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

    def get_mesh_nodal_data(self, dtype=np.float32) -> dict:
        nodal_data = dict()
        for data in self.meshdata().findall("NodeData"):
            _this_data = [x.text for x in data.findall("node")]
            ref = data.attrib["node_set"]
            if ref not in nodal_data.keys():
                nodal_data[ref] = dict()
            name = data.attrib["name"]
            nodal_data[ref][name] = np.array(
                _this_data, dtype=float)
        return nodal_data

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

            # Determine the insertion point based on existing "Nodes"
            insert_point = 0  # Default to beginning
            for idx, element in enumerate(self.geometry()):
                if element.tag == "Nodes":
                    insert_point = idx + 1  # Update to insert after the last "Nodes"
            self.geometry().insert(insert_point, el_root)

    def add_elements(self, elements: list, initial_el_id: int = 1) -> None:
        """
            Adds elements to Geometry
        """

        last_initial_id = initial_el_id
        for elem_data in elements:
            el_root = ET.Element("Elements")

            if "type" not in elem_data:
                eltype = self.get_element_type(len(elem_data["elems"][0]))
            else:
                eltype = str(elem_data["type"])

            if "mat" in elem_data:
                el_mat = str(elem_data["mat"])
            else:
                el_mat = "1"
            if "name" in elem_data:
                el_name = str(elem_data["name"])
            else:
                el_name = "Part1"

            el_root.set("type", eltype)
            el_root.set("mat", el_mat)
            el_root.set("name", el_name)
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

    # - - - - - - - - - - - - -
    # Loads

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

    def add_nodal_loads(self, nodal_loads: list) -> None:

        for load_idx, new_load in enumerate(nodal_loads):
            assert isinstance(new_load, dict), (
                f"Items in nodal_loads must be a dictionary. "
                f"Got {type(new_load)} instead."
                f"Item: {load_idx}."
            )
            assert "bc" in new_load, (
                "Items in nodal_loads must contain a 'bc' key. "
                "'bc' identifies the boundary condition to be applied. "
                "In this case, it is used to indicate the load direction."
                f"Item: {load_idx}."
            )
            assert "node_set" in new_load, (
                "Items in nodal_loads must contain a 'node_set' key. "
                f"Item: {load_idx}."
            )
            assert "scale" in new_load, (
                "Items in nodal_loads must contain a 'scale' key. "
                f"Item: {load_idx}."
            )
            # Optional items
            if "load_curve" not in new_load:
                new_load["load_curve"] = "1"

            load_element = ET.Element("nodal_load")
            load_element.set("bc", new_load["bc"])
            load_element.set("node_set", new_load["node_set"])
            subel = ET.SubElement(load_element, "scale")
            subel.set("lc", new_load["load_curve"])
            subel.text = str(new_load["scale"])

            self.loads().extend([load_element])

    def clear_loads(self):
        loads = self.loads()
        loads[:] = []

    def clear_boundary_conditions(self):
        loads = self.boundary()
        loads[:] = []

    # - - - - - - - - - - - - -
    # Add mesh data to feb file

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

    # Add NodeData to MeshData in feb file
    def add_mesh_node_data(self, node_data: list) -> None:

        for item_idx, item in enumerate(node_data):
            assert isinstance(item, dict), (
                f"Items in node_data must be a dictionary. "
                f"Got {type(item)} instead for item {item_idx}."
            )
            assert "name" in item, (
                "Items in node_data must contain a 'name' key. "
                f"Item: {item_idx}."
            )
            assert "node_set" in item, (
                "Items in node_data must contain a 'node_set' key."
                f"Item: {item_idx}."
            )
            assert "data" in item, (
                "Items in node_data must contain a 'data' key."
                f"Item: {item_idx}."
            )
            # check data types
            assert isinstance(item["name"], str), (
                "Item's 'name' should be a string."
                f"Got {type(item['name'])} instead."
                f"Item: {item_idx}."
            )
            assert isinstance(item["node_set"], str), (
                "Item's 'node_set' should be a string."
                f"Got {type(item['node_set'])} instead."
                f"Item: {item_idx}."
            )
            assert isinstance(item["data"], (list, np.ndarray)), (
                "Item's 'should' be a list or numpy array."
                f"Got {type(item['data'])} instead."
                f"Item: {item_idx}."
            )
            el_root = ET.Element("NodeData")
            el_root.set("name", item["name"])
            el_root.set("node_set", item["node_set"])
            for i, data in enumerate(item["data"]):
                subel = ET.SubElement(el_root, "node")
                subel.set("lid", str(i + 1))
                if isinstance(data, (list, np.ndarray)):
                    subel.text = ",".join(map(str, list(data)))
                else:
                    subel.text = str(data)

            self.meshdata().extend([el_root])

    def add_mesh_element_data(self, elem_data: list) -> None:

        for item_idx, item in enumerate(elem_data):
            assert isinstance(item, dict), (
                f"Items in elem_data must be a dictionary. "
                f"Got {type(item)} instead for item {item_idx}."
            )
            assert "name" in item, (
                "Items in elem_data must contain a 'name' key. "
                f"Item: {item_idx}."
            )
            assert "elem_set" in item, (
                "Items in elem_data must contain a 'elem_set' key."
                f"Item: {item_idx}."
            )
            assert "data" in item, (
                "Items in elem_data must contain a 'data' key."
                f"Item: {item_idx}."
            )
            if "var" not in item:
                assert "datatype" in item, (
                    "Items in elem_data must contain a 'datatype' key "
                    "if 'var' is not specified. FEBio requires user to "
                    "specify the data type for non-standard variables."
                    "e.g. 'scalar', 'vector', 'tensor' or 'mat3d'."
                    f"Item: {item_idx}."
                )
            # check data types
            assert isinstance(item["name"], str), (
                "Item's 'name' should be a string."
                f"Got {type(item['name'])} instead."
                f"Item: {item_idx}."
            )
            assert isinstance(item["elem_set"], str), (
                "Item's 'elem_set' should be a string."
                f"Got {type(item['elem_set'])} instead."
                f"Item: {item_idx}."
            )
            assert isinstance(item["data"], (list, np.ndarray)), (
                "Item's 'should' be a list or numpy array."
                f"Got {type(item['data'])} instead."
                f"Item: {item_idx}."
            )
            if "var" not in item:
                assert isinstance(item["datatype"], str), (
                    "Item's 'datatype' should be a string."
                    f"Got {type(item['datatype'])} instead."
                    f"Item: {item_idx}."
                )
            el_root = ET.Element("ElementData")
            if "var" in item:
                el_root.set("var", item["var"])
            el_root.set("name", item["name"])
            el_root.set("elem_set", item["elem_set"])
            if "datatype" in item:
                el_root.set("datatype", item["datatype"])
            for i, data in enumerate(item["data"]):
                subel = ET.SubElement(el_root, "elem")
                subel.set("lid", str(i + 1))
                if isinstance(data, (list, np.ndarray)):
                    subel.text = ",".join(map(str, list(data)))
                else:
                    subel.text = str(data)

            self.meshdata().extend([el_root])

    # ===========================
    # Modify content from feb file

    def replace_material_params(self, params: dict) -> None:
        materials = self.material().findall("material")
        for mat_ref, mat_params in params.items():
            # Search materials
            for mat in materials:
                # Find material reference
                if mat_ref == mat.get("name") or mat_ref == mat.get("id") or mat_ref == mat.get("type"):
                    # Modify params
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

        # # get nodes tree from existing nodes data in feb
        # root = self.geometry().find("Nodes")
        # nodes_tree = root.findall("node")

        # # raise warning if length of new nodes do not match existing nodes
        # if len(nodes_tree) != len(nodes):
        #     warnings.warn("Replacing nodes data with new amound of nodes.\
        #             This can lead to malfunction in feb file exectuion")

        # # clear nodes content
        # root.clear()

        for new_node_data in nodes:
            data_name = new_node_data["name"]
            data_values = new_node_data["nodes"]

            node_trees = self.geometry().findall("Nodes")
            for tree in node_trees:
                tree_name = node_trees[0].attrib["name"]
                if data_name == tree_name:
                    existing_nodes = tree.findall("node")
                    if len(existing_nodes) != len(data_values):
                        print(
                            "Warning: replacing nodes with different amount of nodes."
                            f"\nExisting nodes: {len(existing_nodes)}, new nodes: {len(data_values)}"
                            f"\nNode name: {data_name}."
                            "\nThis can lead to malfunction in feb file exectuion")
                    self.geometry().remove(tree)

        # add new nodes
        self.add_nodes(nodes, initial_el_id)

    def replace_load_scale(self, loads: list):

        for data in loads:
            load_tag = data["tag"]

            reference_name = None
            reference_attr = None
            # get reference name
            if "bc" in data:
                reference_name = data["bc"]
                reference_attr = "bc"
            elif "node_set" in data:
                reference_name = data["node_set"]
                reference_attr = "node_set"
            elif "surface" in data:
                reference_name = data["surface"]
                reference_attr = "surface"
            elif "elem_set" in data:
                reference_name = data["elem_set"]
                reference_attr = "elem_set"
            elif "name" in data:
                reference_name = data["name"]
                reference_attr = "name"

            load_scale = data["scale"]

            for entry in self.loads().findall(load_tag):
                if reference_attr is not None:
                    if entry.attrib[reference_attr] == reference_name:
                        entry_scale = entry.find("scale")
                        entry_scale.text = str(load_scale)
                # modify all entries
                else:
                    for entry_scale in entry.findall("scale"):
                        entry_scale.text = str(load_scale)

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
