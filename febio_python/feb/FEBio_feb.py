import warnings
import numpy as np
import xml.etree.ElementTree as ET
from .FEBio_xml_handler import FEBio_xml_handler


class FEBio_feb(FEBio_xml_handler):
    def __init__(self, path_to_feb_file):
        super().__init__(path_to_feb_file)

    # ===========================
    # retrieve data from feb file

    def get_nodes_array(self, dtype=np.float32):
        """
          Returns nodes from .feb file as a numpy array of specified dtype.
        """
        if not self.has_tag("Geometry"):
            print("Could not find Geometry tag in .feb file.")
            return np.empty([])

        node_root = self.Geometry.find("Nodes")
        nodes_list = node_root.findall("node")
        nodes = np.zeros([len(nodes_list), 3])
        for i, ninfo in enumerate(nodes_list):
            nodes[i, :] = np.fromstring(ninfo.text, sep=",")
        return nodes.astype(dtype)

    def get_elems_array(self):
        """
          Returns elems from .feb file as a numpy array of specified dtype.
        """
        if not self.has_tag("Geometry"):
            print("Could not find Geometry tag in .feb file.")
            return np.empty([])

        elems_root = self.Geometry.find("Elements")
        elems_list = elems_root.findall("elem")
        elems = []
        for ninfo in elems_list:
            elems.append(np.fromstring(ninfo.text, sep=",", dtype=np.int32))
        return np.asarray(elems, dtype=np.object)

    def get_nodesets(self):
        """
          Returns a dictionary with nodesets as {name: [nodes]}
        """
        if not self.has_tag("Geometry"):
            print("Could not find Geometry tag in .feb file.")
            return np.empty([])

        nodesets = dict()
        nodesets_list = self.Geometry.findall("NodeSet")
        for item in nodesets_list:
            nodesets[item.attrib["name"]] = np.array(
                [node.attrib["id"] for node in item.findall("node")],
                dtype=np.int32)
        return nodesets

    # ============================
    # Add content to feb file

    def add_nodesets(self, nodesets):
        """
          Adds nodesets to Geometry tree element.
          'nodesets' should be a dictionary with keys representing nodeset name
          and values representing nodes of such nodeset.
        """
        if not self.has_tag("Geometry"):
            print("Could not find Geometry tag in .feb file.")
            return None

        for (name, nodes) in nodesets.items():
            nodeset = ET.Element("NodeSet")
            nodeset.set("name", name)
            for node_id in nodes:
                subel = ET.SubElement(nodeset, "node")
                subel.set("id", str(node_id))

            self.Geometry.extend([nodeset])

    def add_boundaries(self, bcs):
        """
          Adds boundary conditions to Boundary tree element.
        """
        if not self.has_tag("Boundary"):
            print("Could not find Boundary tag in .feb file.")
            return None

        for (bc_type, bc, nodeset) in bcs:
            boundary = ET.Element(bc_type)
            boundary.set("bc", bc)
            boundary.set("node_set", nodeset)
            self.Boundary.extend([boundary])

    def add_discretesets(self, discretesets):
        if not self.has_tag("Geometry"):
            print("Could not find Geometry tag in .feb file.")
            return None

        for (name, delems) in discretesets.items():
            newset = ET.Element("DiscreteSet")
            newset.set("name", name)
            for nodes_ids in delems:
                subel = ET.SubElement(newset, "delem")
                if isinstance(nodes_ids, str):
                    subel.text = nodes_ids
                else:
                    subel.text = ",".join(map(str, nodes_ids))

            self.Geometry.extend([newset])

    def add_boundaries(self, bcs):
        """
          Adds boundary conditions to Boundary tree element.
        """
        if not self.has_tag("Boundary"):
            print("Could not find Boundary tag in .feb file.")
            return None

        for (bc_type, bc, nodeset) in bcs:
            boundary = ET.Element(bc_type)
            boundary.set("bc", bc)
            boundary.set("node_set", nodeset)
            self.Boundary.extend([boundary])

    def add_elements(self, elements, initial_el_id=1, eltype="quad4"):
        """
          Adds elements to Geometry
        """
        if not self.has_tag("Geometry"):
            print("Could not find Geometry tag in .feb file.")
            return None

        for (name, elemens) in elements.items():
            el_root = ET.Element("Elements")
            el_root.set("name", name)
            el_root.set("type", eltype)

            for i, elem in enumerate(elemens):
                subel = ET.SubElement(el_root, "elem")
                subel.set("id", str(i + initial_el_id))
                subel.text = ",".join(map(str, elem))
            initial_el_id = i + 1

            self.Geometry.extend([el_root])

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
        if not self.has_tag("Geometry"):
            print("Could not find Geometry tag in .feb file.")
            return None

        # get nodes tree from existing nodes data in feb
        root = self.Geometry.find("Nodes")
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
    # Other

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

            self.Geometry.extend([newset])
