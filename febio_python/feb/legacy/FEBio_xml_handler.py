from tkinter import E
import xml.etree.ElementTree as ET
from os.path import isfile, join
# from .. logger import console_log as log
from febio_python.core.enums import *
from pathlib import Path
import numpy as np

from febio_python.utils.enum_utils import *

class FEBio_xml_handler():
    def __init__(self, tree, root=None, filepath=None):
        self.path_to_file = filepath

        # ET tree reference for paesed xml file
        self.tree = tree
        # Tree root reference for paesed xml file
        if root is not None:
            self.root = root
        else:
            self.root = tree.getroot()

        # Order in which outer tags should be placed
        self.lead_tag_order = [item.value for item in list(FEB_LEAD_TAGS)]

        self.LEAD_TAGS = FEB_LEAD_TAGS
        self.ELEM_TYPES = ELEM_TYPES
        self.N_PTS_IN_ELEMENT = N_PTS_IN_ELEMENT
        self.SURFACE_EL_TYPE = SURFACE_EL_TYPE

    def __repr__(self):
        to_print = "{}:\n".format(self.__class__.__name__)
        for el in list(self.root):
            to_print += "-> {}: {}\n".format(el.tag, len(el))

            # print material details
            if str(el.tag).lower() == str(FEB_LEAD_TAGS.MATERIAL.value).lower() and len(el) > 0:
                for geo_el in list(el):
                    if "type" in geo_el.keys():
                        to_print += "--> {} '{}': {}\n".format(
                            geo_el.tag, geo_el.attrib["type"], len(geo_el))
                    else:
                        to_print += "--> {}: {}\n".format(
                            geo_el.tag, len(geo_el))

            # print geometry details (nodes and elements)
            if str(el.tag).lower() == str(FEB_LEAD_TAGS.GEOMETRY.value).lower() and len(el) > 0:
                for geo_el in list(el):
                    if geo_el.tag == "Nodes" or geo_el.tag == "Elements":
                        if "name" in geo_el.keys():
                            to_print += "--> {} '{}': {}\n".format(
                                geo_el.tag, geo_el.attrib["name"], len(geo_el))
                        else:
                            to_print += "--> {}: {}\n".format(
                                geo_el.tag, len(geo_el))
        return to_print

    def __len__(self):
        return len(self.root)

    @staticmethod
    def parse(content) -> tuple:
        """Parse some content into 'ET' tree and root elements.

        Args:
            content (str, pathlike, ElementTree or Element): Content to be parsed.

        Returns:
            tuple: (tree, root)
        """
        if isinstance(content, str) or isinstance(content, Path):
            if isfile(str(content)):
                tree = ET.parse(content)
                root = tree.getroot()
            else:
                try:
                    tree = ET.fromstring(content)
                    root = tree.getroot()
                except:
                    raise(ValueError(
                        "Content was identified as string, but could not be parsed. Please, verify."))
        elif isinstance(content, ET.ElementTree):
            try:
                tree = content
                root = tree.getroot()
            except:
                raise(ValueError(
                    "Content was identified as ElementTree object, but could not get its root. Please, verify."))
        elif isinstance(content, ET.Element):
            tree = None
            root = content
        else:
            raise(ValueError("Content is not file, string or xml tree. Please, verify."))

        return tree, root

    @classmethod
    def from_file(cls, filename) -> object:
        if not isfile(filename):
            raise ValueError("Input file does not exist.")
        tree, root = FEBio_xml_handler.parse(filename)
        return cls(tree, root=root, filepath=filename)

    @classmethod
    def from_parse(cls, content) -> object:
        tree, root = FEBio_xml_handler.parse(content)
        return cls(tree, root=root)

    @classmethod
    def from_new(cls) -> object:
        # tree = ET.ElementTree(ET.Element(FEB_ROOT.ROOT.value))
        root = ET.Element(FEB_ROOT.ROOT.value)
        for item in FEB_LEAD_TAGS:
            _ = ET.SubElement(root, item.value)
        tree = ET.ElementTree(root)
        return cls(tree)

    def clean(self) -> None:
        """Remore all root children that are empty."
        """
        for child in list(self.root):
            if child.tag != FEB_LEAD_TAGS.MODULE:
                if len(child) == 0:
                    self.root.remove(child)

    def write(self, filepath, clean=False) -> None:
        # self.indent(self.root)
        if clean:
            self.clean()
        ET.indent(self.tree, space="\t", level=0)
        import os
        out_dir = os.path.dirname(os.path.abspath(filepath))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        self.tree.write(filepath, encoding="ISO-8859-1")

    def clone(self) -> object:
        """Return a copy of itself."""
        from copy import deepcopy
        return deepcopy(self)

    # ----------------------------------------------------------------
    # old methods for extracting some geometry data (not fully tested and not optimized)

    def get_elemsets(self, ids_only=True):
        """
                Return a dictionary with Elementset name as key and 
                element data as values.
                If ids_only is True, element data is a set with element ids
                Else, element data is a list with all nodes from that elem.
        """
        elsets = {}
        # iterate through all Elements tags
        for elem_set in self.Geometry.findall("Elements"): 				# pylint: disable=no-member
            # iterate through all elements and save all elements
            if ids_only:
                elems = set()
                for elem in elem_set.findall("elem"):
                    elems.add(int(float(elem.get("id"))))
            else:
                elems = []
                for elem in elem_set.findall("elem"):
                    a = [elem.get("id")]
                    a.extend([int(float(b))
                             for b in str(elem.text).split(",")])
                    elems.append(a)
            # get name of element set and save eldata in dict
            elsets[elem_set.get("name")] = elems
        return elsets

    def get_geometry_data(self, what=[], return_nodes=True, return_elems=True):
        nodes = []
        elems = []

        return_selected = list()
        if self.has_tag("Geometry"):
            for node in self.Geometry.find("Nodes").findall("node"): 		# pylint: disable=no-member
                a = [node.get('id')]
                a.extend([float(b) for b in str(node.text).split(",")])
                nodes.append(a)

            for all_elems in self.Geometry.findall("Elements"): 				# pylint: disable=no-member
                for elem in all_elems.findall("elem"):
                    a = [elem.get("id")]
                    a.extend([float(b) for b in str(elem.text).split(",")])
                    elems.append(a)

            if len(what) > 0:
                # selected_items_list = []
                selected_items = {}
                for item in what:

                    for catg in self.Geometry.findall(item[0]):
                        if catg.attrib["name"] == item[1]:
                            selected_items[item[1]] = []
                            cat_list = list(
                                catg) if item[2] == "any" else catg.findall(item[2])
                            for c in cat_list:
                                a = [c.get("id")]
                                if c.text != None:
                                    a.extend([float(b)
                                             for b in str(c.text).split(",")])
                                selected_items[item[1]].append(a)
                    # selected_items_list.append(selected_items)
                return_selected.append(selected_items)

        to_return = list()
        if return_nodes == True:
            to_return.append(nodes)
        if return_elems == True:
            to_return.append(elems)
        if len(return_selected) > 0:
            to_return.extend(return_selected)

        return to_return

    # ===========================
    # NEW METHODS:

    def check_enum(self, value: Enum) -> tuple:
        if isinstance(value, Enum):
            return value.name, value.value
        else:
            return value, value

    def get_lead_tag(self, tag: str) -> ET.Element:
        """Return ET element corresponding to given tag. Lead tags are defined \
            as 'root' tags contained within 'febio_spec'.

        Args:
            tag (str or enum): Name of tag.

        Raises:
            KeyError: If tag is not found in 'febio_spec'.

        Returns:
            ET.Element: Pointer to tag element.
        """
        tag_name, tag_val = self.check_enum(tag)
        el = self.root.find(tag_val)
        if el is None:
            raise KeyError(
                "Tag '%s' not found. Are you sure it is valid? Check FEB_LEAD_TAGS enums for details." % tag_name)
        return el

    def get_tag(self, lead_tag: str, tag: str) -> ET.Element:
        """Return ET element corresponding to given tag contained within 'lead_tag'. \
            Lead tags are defined as 'root' tags contained within 'febio_spec'. \
            This method will return only the first tag found. If you wish to\
                find multiple repeated tags, use 'get_repeated_tags'.

        Args:
            lead_tag (str or enum): Name of lead tag.
            tag (str or enum): Name of tag.

        Raises:
            KeyError: If tag is not found in 'lead_tag'.

        Returns:
            ET.Element: Pointer to tag element.

        Example:
            feb.get_tag(FEB_LEAD_TAGS.GEOMETRY, "Nodes")
        """
        el = self.get_lead_tag(lead_tag)
        tag_name, tag_val = self.check_enum(tag)
        el = el.find(tag_val)
        if el is None:
            lead_tag_name, _ = self.check_enum(lead_tag)
            raise KeyError(
                "Tag '{}' not found within {}.".format(tag_name, lead_tag_name))
        return el

    def get_repeated_tags(self, lead_tag: str, tag: str) -> ET.Element:
        """Return ET element corresponding to given tag contained within 'lead_tag'. \
            Lead tags are defined as 'root' tags contained within 'febio_spec'. \

        Args:
            lead_tag (str or enum): Name of lead tag.
            tag (str or enum): Name of tag.

        Raises:
            KeyError: If tag is not found in 'lead_tag'.

        Returns:
            ET.Element: Pointer to tag element.

        Example:
            feb.get_repeated_tags(FEB_LEAD_TAGS.GEOMETRY, "NodeSet")

        """
        el = self.get_lead_tag(lead_tag)
        tag_name, tag_val = self.check_enum(tag)
        el = el.findall(tag_val)
        if len(el) == 0:
            lead_tag_name, _ = self.check_enum(lead_tag)
            raise KeyError(
                "Tag '{}' not found within {}.".format(tag_name, lead_tag_name))
        return el

    def add_lead_tag(self, lead_tag: str) -> None:
        """Tries to add lead tag at proper position according to 'lead_tag_order'.

        Args:
            lead_tag (str): Lead tag to add. Refer to FEB_LEAD_TAGS enum for details.

        Raises:
            ValueError: If lead_tag is invalid.
        """
        try:
            self.get_lead_tag(lead_tag)
        except KeyError:
            tag_name, tag_val = self.check_enum(lead_tag)
            if tag_val not in self.lead_tag_order:
                raise ValueError(
                    "Invalid value for 'lead_tag': {}. Check FEB_LEAD_TAGS enum for details.".format(tag_name))
            idx = self.lead_tag_order.index(tag_val)  # where to insert

            if len(self.root) == 0 or idx == 0:
                self.root.insert(0, ET.Element(tag_val))
            elif idx == len(self.root):
                self.root.insert(idx-1, ET.Element(tag_val))
            elif idx < len(self.root):
                self.root.insert(idx, ET.Element(tag_val))
            else:
                added = False
                for child in list(self.root):
                    child_idx = self.lead_tag_order.index(child.tag)
                    if idx < child_idx:
                        self.root.insert(-1, ET.Element(tag_val))
                        added = True
                        break
                if added == False:
                    self.root.insert(idx, ET.Element(tag_val))

    def get_content_from_repeated_tags(self, lead_tag: str, tag: str, dtype=np.float32) -> dict:
        """
            Returns a dictionary with keys corresponding to tag names (if no name is found, \
            it will replace with corresponding index), and values from each tag text as a np.ndarray. \
            Lead tags are defined as 'root' tags contained within 'febio_spec'. \
            Tag is defined as the sub-tag of a lead tag.

        Args:
            lead_tag (str or enum): Name of lead tag.
            tag (str or enum): Name of tag.

        Returns:
            Dict with tag name as keys and values as np.ndarray from tag content.

        Example:
            feb.get_content_from_repeated_tags(FEB_LEAD_TAGS.GEOMETRY, "Surface")
        """

        data = dict()
        for i, item in enumerate(self.get_repeated_tags(lead_tag, tag)):
            if "name" in item.attrib:
                name = item.attrib["name"]
            else:
                name = i
            data[name] = np.array(
                [np.fromstring(sub_el.text, sep=",") for sub_el in list(item)],
                dtype=dtype)
        return data

    def get_ids_from_repeated_tags(self, lead_tag: str, tag: str, dtype=np.float32) -> dict:
        """
            Returns a dictionary with keys corresponding to tag names (if no name is found, \
            it will replace with corresponding index), and values from each tag text as a np.ndarray. \
            Lead tags are defined as 'root' tags contained within 'febio_spec'. \
            Tag is defined as the sub-tag of a lead tag.

        Args:
            lead_tag (str or enum): Name of lead tag.
            tag (str or enum): Name of tag.

        Returns:
            Dict with tag name as keys and values as np.ndarray from tag ids.

        Example:
            feb.get_ids_from_repeated_tags(FEB_LEAD_TAGS.GEOMETRY, "NodeSet")
        """

        data = dict()
        for i, item in enumerate(self.get_repeated_tags(lead_tag, tag)):
            if "name" in item.attrib:
                name = item.attrib["name"]
            else:
                name = i
            data[name] = np.array(
                [sub_el.attrib["id"] for sub_el in list(item)],
                dtype=dtype)
        return data

    # ----------------------------------------------------------------
    # lead tag shortcuts

    def module(self) -> ET.Element:
        """Returns pointer to 'MODULE' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.MODULE)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.MODULE)
            return self.get_lead_tag(FEB_LEAD_TAGS.MODULE)

    def control(self) -> ET.Element:
        """Returns pointer to 'CONTROL' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.CONTROL)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.CONTROL)
            return self.get_lead_tag(FEB_LEAD_TAGS.CONTROL)

    def material(self) -> ET.Element:
        """Returns pointer to 'MATERIAL' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.MATERIAL)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.MATERIAL)
            return self.get_lead_tag(FEB_LEAD_TAGS.MATERIAL)

    def globals(self) -> ET.Element:
        """Returns pointer to 'GLOBALS' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.GLOBALS)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.GLOBALS)
            return self.get_lead_tag(FEB_LEAD_TAGS.GLOBALS)

    def geometry(self) -> ET.Element:
        """Returns pointer to 'GEOMETRY' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.GEOMETRY)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.GEOMETRY)
            return self.get_lead_tag(FEB_LEAD_TAGS.GEOMETRY)

    def boundary(self) -> ET.Element:
        """Returns pointer to 'BOUNDARY' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.BOUNDARY)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.BOUNDARY)
            return self.get_lead_tag(FEB_LEAD_TAGS.BOUNDARY)

    def loads(self) -> ET.Element:
        """Returns pointer to 'LOADS' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.LOADS)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.LOADS)
            return self.get_lead_tag(FEB_LEAD_TAGS.LOADS)

    def discrete(self) -> ET.Element:
        """Returns pointer to 'DISCRETE' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.DISCRETE)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.DISCRETE)
            return self.get_lead_tag(FEB_LEAD_TAGS.DISCRETE)

    def loaddata(self) -> ET.Element:
        """Returns pointer to 'LOADDATA' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.LOADDATA)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.LOADDATA)
            return self.get_lead_tag(FEB_LEAD_TAGS.LOADDATA)

    def output(self) -> ET.Element:
        """Returns pointer to 'OUTPUT' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.OUTPUT)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.OUTPUT)
            return self.get_lead_tag(FEB_LEAD_TAGS.OUTPUT)

    def meshdata(self) -> ET.Element:
        """Returns pointer to 'MESHDATA' element within 'febio_spec'."""
        try:
            return self.get_lead_tag(FEB_LEAD_TAGS.MESHDATA)
        except KeyError:
            self.add_lead_tag(FEB_LEAD_TAGS.MESHDATA)
            return self.get_lead_tag(FEB_LEAD_TAGS.MESHDATA)


    def get_element_type(self, n_nodes_per_cell) -> str:
        """ Returns FEBio element type representation """
        try:
            assert_value(self.N_PTS_IN_ELEMENT, n_nodes_per_cell)
            eltype = self.N_PTS_IN_ELEMENT(n_nodes_per_cell).name
            assert_member(self.ELEM_TYPES, eltype)
            return self.ELEM_TYPES[eltype].value
        except AssertionError:
            raise AssertionError(
                "Unable to identify element type. Are you sure "
                "that element type is a valid FEBio element? "
                "If so, check implemented types avaiable at "
                "{} and corresponding number of nodes per cell "
                "at {}. Maybe you can improve our library with "
                "additional members. Current implementations are:"
                "\n'ELEM_TYPES':{}\n'N_PTS_IN_ELEMENT':{}"
                "".format(self.ELEM_TYPES, self.N_PTS_IN_ELEMENT,
                    enum_to_dict(self.ELEM_TYPES),
                    enum_to_dict(self.N_PTS_IN_ELEMENT)
                )
            )