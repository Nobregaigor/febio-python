import warnings
import numpy as np
import xml.etree.ElementTree as ET
from .FEBio_xml_handler import FEBio_xml_handler
from .enums import *


class FEBio_feb(FEBio_xml_handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ===========================
    # retrieve data from feb file

    def get_nodes_array(self, dtype=np.float32):
        """
          Returns nodes from .feb file as a numpy array of specified dtype.
        """
        node_root = self.geometry().find("Nodes")
        nodes_list = node_root.findall("node")
        nodes = np.zeros([len(nodes_list), 3])
        for i, ninfo in enumerate(nodes_list):
            nodes[i, :] = np.fromstring(ninfo.text, sep=",")
        return nodes.astype(dtype)

    def get_elems_array(self):
        """
          Returns elems from .feb file as a numpy array of specified dtype.
        """

        elems_root = self.geometry().find("Elements")
        elems_list = elems_root.findall("elem")
        elems = []
        for ninfo in elems_list:
            elems.append(np.fromstring(ninfo.text, sep=",", dtype=np.int32))
        return np.asarray(elems, dtype=np.object)

    def get_nodesets(self):
        """
          Returns a dictionary with nodesets as {name: [nodes]}
        """

        nodesets = dict()
        nodesets_list = self.geometry().findall("NodeSet")
        for item in nodesets_list:
            nodesets[item.attrib["name"]] = np.array(
                [node.attrib["id"] for node in item.findall("node")],
                dtype=np.int32)
        return nodesets

    # ============================
    # Add content to feb file

    def add_nodes(self, nodes: list, initial_el_id: int = 1) -> None:
        """
          Adds nodes to Geometry
        """

        for (node_elem) in nodes:
            if "nodes" not in node_elem:
                raise ValueError("Nodes not found for one of the nodes.")

            el_root = ET.Element("Nodes")
            if "name" in node_elem:
                el_root.set("name", node_elem["name"])

            for i, node_xyz in enumerate(node_elem["nodes"]):
                if len(node_xyz) != 3:  # check for correct node information.
                    raise ValueError(
                        "Node '{}' does not have the correct number of coordinates. It should contain [x,y,z] values.")
                subel = ET.SubElement(el_root, "node")
                subel.set("id", str(i + initial_el_id))
                subel.text = ",".join(map(str, node_xyz))
            initial_el_id = i + 1

            self.geometry().extend([el_root])

    def add_elements(self, elements: list, initial_el_id: int = 1) -> None:
        """
          Adds elements to Geometry
        """

        for elem_data in elements:
            el_root = ET.Element("Elements")
            el_root.set("name", elem_data["name"])
            el_root.set("type", elem_data["type"])
            if "mat" in elem_data:
                el_root.set("mat", elem_data["mat"])

            for i, elem in enumerate(elem_data["elems"]):
                subel = ET.SubElement(el_root, "elem")
                subel.set("id", str(i + initial_el_id))
                subel.text = ",".join(map(str, elem))
            initial_el_id = i + 1

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
                subel.set("id", str(i+initial_el_id))
                subel.text = ",".join(map(str, node_ids))

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

    # --- this function is ugly -> need to be improved (but works for now)
    def add_meshdata(self, mesh_data: list, initial_el_id: int = 1) -> None:
        """
          Adds meshdata to file
        """

        for elem_data in mesh_data:
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

            self.meshdata().extend([el_root])

    # ===========================
    # Modify content from feb file

    def replace_nodes(self, nodes, name="Object"):
        """
          Replaces nodes elements from .feb_file with new set of nodes.
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
        for i, node_xyz in enumerate(nodes):
            subel = ET.SubElement(root, "node")
            subel.set("id", str(i + 1))
            subel.text = ",".join(map(str, node_xyz))

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
