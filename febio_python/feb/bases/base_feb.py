# from tkinter import E
import xml.etree.ElementTree as ET
from os.path import isfile
# from .. logger import console_log as log
from febio_python.core.enums import FEB_2_5_LEAD_TAGS, FEB_3_0_LEAD_TAGS, FEB_4_0_LEAD_TAGS
from febio_python.core.enums import FEB_ROOT, FEB_MAJOR_TAGS
from pathlib import Path
import numpy as np
from typing import Union, Tuple, List
from febio_python.utils import enum_utils as eu
from collections import OrderedDict


class FebBaseObject():
    def __init__(self,
                 tree: Union[ET.ElementTree, None] = None,
                 root: Union[ET.Element, None] = None,
                 filepath: Union[str, Path] = None):
        self.path_to_file = filepath

        # Handle initialization of tree and root
        tree, root = self._handle_initialization(tree, root, filepath)

        # Set tree and root attributes
        self.tree: ET.ElementTree = tree
        self.root: ET.Element = root
        # Set some enums as attributes
        if self.version == 4.0:
            LEAD_TAGS = FEB_4_0_LEAD_TAGS
        elif self.version == 3.0:
            LEAD_TAGS = FEB_3_0_LEAD_TAGS
        else:
            LEAD_TAGS = FEB_2_5_LEAD_TAGS
        # Order in which outer tags should be placed
        self.LEAD_TAGS: FEB_2_5_LEAD_TAGS | FEB_3_0_LEAD_TAGS | FEB_4_0_LEAD_TAGS = LEAD_TAGS
        self.MAJOR_TAGS = FEB_MAJOR_TAGS

        self.lead_tag_order = [item.value for item in list(LEAD_TAGS)]

    def __repr__(self):
        to_print = f"{self.__class__.__name__}({self.version}):\n"
        for el in list(self.root):
            if str(el.tag).lower() == str(self.LEAD_TAGS.MODULE.value).lower():
                to_print += f"-> {el.tag}: {el.attrib['type'] if 'type' in el.attrib else 'Unknown'}\n"
            else:
                to_print += f"-> {el.tag}: {len(el)}\n"

        return to_print

    def __len__(self):
        return len(self.root)

    # ====================================================================================================== #
    # Initialization methods
    # ====================================================================================================== #

    def _handle_initialization(self,
                               tree: Union[ET.ElementTree, None] = None,
                               root: Union[ET.Element, None] = None,
                               filepath: Union[str, Path] = None) -> Tuple[ET.ElementTree, ET.Element]:
        # Handle the case in which a filepath is provided -> parse and get tree and root
        if filepath is not None:
            tree, root = FebBaseObject.parse(filepath)

        # If no filepath is provided, proceed with other cases
        else:
            # Handle the case in which a tree what provided
            if tree is not None:
                # ET tree reference for paesed xml file
                if not isinstance(tree, ET.ElementTree):
                    raise ValueError("Tree must be an ElementTree.ElementTree object."
                                    f"Received: {type(tree)}")

                # Tree root reference for paesed xml file
                if root is not None:
                    if not isinstance(root, ET.ElementTree):
                        raise ValueError("Root must be an ElementTree.ElementTree object."
                                        f"Received: {type(root)}")
                else:
                    root: ET.Element = tree.getroot()

            # Handle the case in which a tree was not provided: Plant a new tree!
            else:
                if self.__default_version == 2.5:
                    LEAD_TAGS = FEB_2_5_LEAD_TAGS
                elif self.__default_version == 3.0:
                    LEAD_TAGS = FEB_3_0_LEAD_TAGS
                else:
                    LEAD_TAGS = FEB_4_0_LEAD_TAGS
                root = ET.Element(FEB_ROOT.ROOT.value)
                for item in LEAD_TAGS:
                    _ = ET.SubElement(root, item.value)
                tree = ET.ElementTree(root)

        # If for some reason tree and root are still None, raise an error
        # Should not happen, but just in case...
        if tree is None or root is None:
            raise ValueError("Tree and root were not found, please check input parameters.")

        # Now, if everything is fine, we need to check if the root has the proper version attribute
        if "version" not in root.attrib:
            root.attrib["version"] = str(self.__default_version)
        return tree, root

    # ====================================================================================================== #
    # Static methods
    # ====================================================================================================== #

    @staticmethod
    def parse(content: Union[str, Path, ET.ElementTree, ET.Element]) -> tuple:
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
                except Exception as e:
                    raise ValueError("Content was identified as string, but could not be parsed. "
                                     "Please, verify.") from e
        elif isinstance(content, ET.ElementTree):
            try:
                tree = content
                root = tree.getroot()
            except Exception as e:
                raise ValueError("Content was identified as ElementTree object, "
                                 "but could not get its root. Please, verify.") from e
        elif isinstance(content, ET.Element):
            tree = None
            root = content
        else:
            raise ValueError("Content is not file, string or xml tree. Please, verify.")

        return tree, root

    @classmethod
    def from_file(cls, filename) -> object:
        if not isfile(filename):
            raise ValueError("Input file does not exist.")
        tree, root = cls.parse(filename)
        return cls(tree, root=root, filepath=filename)

    # ====================================================================================================== #
    # Helper methods
    # ====================================================================================================== #

    def clean(self) -> None:
        """Remore all root children that are empty."
        """
        for child in list(self.root):
            if child.tag != self.LEAD_TAGS.MODULE:
                if len(child) == 0:
                    self.root.remove(child)

    def clone(self) -> object:
        """Return a copy of itself."""
        from copy import deepcopy
        return deepcopy(self)

    # ====================================================================================================== #
    # Writing methods
    # ====================================================================================================== #

    def write(self, filepath, clean=False, encoding="ISO-8859-1") -> None:
        if clean:
            self.clean()
        ET.indent(self.tree, space="\t", level=0)
        import os
        out_dir = os.path.dirname(os.path.abspath(filepath))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.tree.write(filepath, encoding=encoding)

    # ====================================================================================================== #
    # Helper Methods to handle tags
    # ====================================================================================================== #

    # Lead tags
    # ----------------

    def add_lead_tag(self, lead_tag: str) -> ET.Element:
        """Tries to add lead tag at proper position according to 'lead_tag_order'.

        Args:
            lead_tag (str): Lead tag to add. Refer to self.LEAD_TAGS enum for details.

        Raises:
            ValueError: If lead_tag is invalid.

        Returns:
            ET.Element: Element added.
        """
        try:
            self.get_lead_tag(lead_tag)
        except KeyError:
            tag_name, tag_val = eu.check_enum(lead_tag)
            if tag_val not in self.lead_tag_order:
                raise ValueError(
                    f"Invalid value for 'lead_tag': {tag_name}. Check self.LEAD_TAGS enum for details.")
            idx = self.lead_tag_order.index(tag_val)  # where to insert

            if len(self.root) == 0 or idx == 0:
                self.root.insert(0, ET.Element(tag_val))
            elif idx == len(self.root):
                self.root.insert(idx - 1, ET.Element(tag_val))
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
                if added is False:
                    self.root.insert(idx, ET.Element(tag_val))

        # return element added
        return self.get_lead_tag(lead_tag)

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
        tag_name, tag_val = eu.check_enum(tag)  # simply check if input is an enum or not. If not, it returns the same value.
        el = self.root.find(tag_val)
        if el is None:
            raise KeyError(
                f"Tag '{tag_name}' not found. Are you sure it is valid? Check self.LEAD_TAGS enums for details.")
        return el

    # Sub-tags
    # ----------------

    def get_tag(self, lead_tag: str, tag: str) -> ET.Element:
        """Return ET element corresponding to given tag contained within 'lead_tag'. \
            Lead tags are defined as 'root' tags contained within 'febio_spec'. \
            This method will return only the first tag found. If you wish to\
                find multiple repeated tags, use 'find_tags'.

        Args:
            lead_tag (str or enum): Name of lead tag.
            tag (str or enum): Name of tag.

        Raises:
            KeyError: If tag is not found in 'lead_tag'.

        Returns:
            ET.Element: Pointer to tag element.

        Example:
            feb.get_tag(self.LEAD_TAGS.GEOMETRY, "Nodes")
        """
        el = self.get_lead_tag(lead_tag)
        tag_name, tag_val = eu.check_enum(tag)
        el = el.find(tag_val)
        if el is None:
            lead_tag_name, _ = eu.check_enum(lead_tag)
            raise KeyError(
                f"Tag '{tag_name}' not found within {lead_tag_name}.")
        return el

    def find_tags(self, lead_tag: str, tag: str) -> List[ET.Element]:
        """Helper function to find all tags within a lead_tag and handle exceptions if not found."""
        lead_element = self.get_lead_tag(lead_tag)
        tag_name, tag_val = eu.check_enum(tag)
        tags = lead_element.findall(tag_val)
        if not tags:
            # raise KeyError(f"Tag '{tag_name}' not found within '{eu.check_enum(lead_tag)[0]}'.")
            return []
        return tags

    def get_tag_data(self, lead_tag: str, tag: str, content_type='text', dtype=np.float32) -> OrderedDict:
        """
        General function to extract data from repeated tags based on content type (text or attribute).

        Args:
            lead_tag (str or enum): Name of the lead tag.
            tag (str or enum): Name of the sub-tag.
            content_type (str): Specifies whether to extract text or a specific attribute ('id' for example).
            dtype (data-type): The desired data-type for the numpy array.

        Returns:
            dict: Data extracted from tags, with keys as tag names or indices.
        """
        data = OrderedDict()
        for i, item in enumerate(self.find_tags(lead_tag, tag)):
            name = item.attrib.get('name', i)
            if content_type == 'text':
                data[name] = np.array([np.fromstring(el.text, sep=",") for el in item], dtype=dtype)
            else:
                data[name] = np.array([sub_el.attrib[content_type] for sub_el in item], dtype=dtype)
        return data

    # ====================================================================================================== #
    # Properties
    # ====================================================================================================== #

    @property
    def __default_version(self) -> float:
        """Return default version of FEBio file."""
        if not hasattr(self, "_default_version"):
            raise ValueError("Default version is not set."
                             "._default_version should be set in the derived class "
                             "before any initialization is done. "
                             "e.g. self._default_version = 3.0")
        return self._default_version

    @property
    def version(self) -> float:
        """Return version of FEBio file."""
        root_version = self.root.attrib.get("version", None)
        if root_version is None:
            root_version = self.__default_version
        return float(root_version)

    @property
    def module(self) -> ET.Element:
        """Returns 'MODULE' tree element within 'febio_spec'."""
        try:
            return self.get_lead_tag(self.LEAD_TAGS.MODULE)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.MODULE)

    @property
    def control(self) -> ET.Element:
        """Returns 'CONTROL' tree element within 'febio_spec'."""
        try:
            return self.get_lead_tag(self.LEAD_TAGS.CONTROL)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.CONTROL)

    @property
    def material(self) -> ET.Element:
        """Returns 'MATERIAL' tree element within 'febio_spec'."""
        try:
            return self.get_lead_tag(self.LEAD_TAGS.MATERIAL)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.MATERIAL)

    @property
    def globals(self) -> ET.Element:
        """Returns 'GLOBALS' tree element within 'febio_spec'."""
        try:
            return self.get_lead_tag(self.LEAD_TAGS.GLOBALS)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.GLOBALS)

    @property
    def geometry(self) -> ET.Element:
        """Returns 'GEOMETRY' tree element within 'febio_spec'."""
        if self.version > 2.5:
            raise ValueError("Geometry tag is not available for FEBio versions greater than 2.5."
                             "Please, use 'Mesh' tag instead.")
        try:
            return self.get_lead_tag(self.LEAD_TAGS.GEOMETRY)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.GEOMETRY)

    @property
    def mesh(self) -> ET.Element:
        """Returns 'MESH' tree element within 'febio_spec'."""
        if self.version < 3.0:
            raise ValueError("Mesh tag is not available for FEBio versions less than 3.0."
                             "Please, use 'Geometry' tag instead.")
        try:
            return self.get_lead_tag(self.LEAD_TAGS.MESH)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.MESH)

    @property
    def meshdomains(self) -> ET.Element:
        """Returns 'MESHDOMAINS' tree element within 'febio_spec'."""
        if self.version < 3.0:
            raise ValueError("MeshDomains tag is not available for FEBio versions less than 3.0.")
        try:
            return self.get_lead_tag(self.LEAD_TAGS.MESHDOMAINS)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.MESHDOMAINS)

    @property
    def boundary(self) -> ET.Element:
        """Returns 'BOUNDARY' tree element within 'febio_spec'."""
        try:
            return self.get_lead_tag(self.LEAD_TAGS.BOUNDARY)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.BOUNDARY)

    @property
    def loads(self) -> ET.Element:
        """Returns 'LOADS' tree element within 'febio_spec'."""
        try:
            return self.get_lead_tag(self.LEAD_TAGS.LOADS)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.LOADS)

    @property
    def discrete(self) -> ET.Element:
        """Returns 'DISCRETE' tree element within 'febio_spec'."""
        try:
            return self.get_lead_tag(self.LEAD_TAGS.DISCRETE)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.DISCRETE)

    @property
    def loaddata(self) -> ET.Element:
        """Returns 'LOADDATA' tree element within 'febio_spec'."""
        try:
            return self.get_lead_tag(self.LEAD_TAGS.LOADDATA)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.LOADDATA)

    @property
    def output(self) -> ET.Element:
        """Returns 'OUTPUT' tree element within 'febio_spec'."""
        try:
            return self.get_lead_tag(self.LEAD_TAGS.OUTPUT)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.OUTPUT)

    @property
    def meshdata(self) -> ET.Element:
        """Returns 'MESHDATA' tree element within 'febio_spec'."""
        try:
            return self.get_lead_tag(self.LEAD_TAGS.MESHDATA)
        except KeyError:
            return self.add_lead_tag(self.LEAD_TAGS.MESHDATA)

    # ====================================================================================================== #
    # Other utility methods
    # ====================================================================================================== #

    def inspect_nodes_and_elements(self) -> None:
        """Inspect geometry details."""
        to_print = ""
        el = self.geometry if self.version < 3.0 else self.mesh
        for geo_el in list(el):
            if geo_el.tag == "Nodes" or geo_el.tag == "Elements":
                if "name" in geo_el.keys():
                    to_print += "--> {} '{}': {}\n".format(
                        geo_el.tag, geo_el.attrib["name"], len(geo_el))
                else:
                    to_print += "--> {}: {}\n".format(
                        geo_el.tag, len(geo_el))
        if len(to_print) == 0:
            to_print = "No geometry details found. Is the geometry defined?"
        print(to_print)
