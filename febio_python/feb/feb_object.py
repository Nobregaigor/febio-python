from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree
from .bases import FebBaseObject
import numpy as np
from typing import Union, Dict, List

from .meta_data import Material
from .meta_data import NodalLoad, PressureLoad
from .meta_data import BoundaryCondition, FixCondition, FixedAxis, RigidBodyCondition
from .meta_data import NodalData, SurfaceData, ElementData


class Feb(FebBaseObject):
    def __init__(self, tree: ElementTree | None = None, root: Element | None = None, filepath: str | Path = None):
        super().__init__(tree, root, filepath)
        
    # =========================================================================================================
    # Retrieve methods
    # =========================================================================================================
    
    # Main geometry data
    # ------------------------------

    # Get full geometry data (as a dict)
    
    def get_nodes_dict(self, dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
        """
        Returns nodes content from .feb file as a dict with keys representing nodes names\
        and values representing [x,y,z] as a numpy array of specified dtype.\

        Args:
        dtype (np.dtype): Numpy dtype.

        Returns:
            dict: {tag_name: [x,y,z]}

        Example:
            feb.get_nodes()
        """
        return self.get_tag_data(self.LEAD_TAGS.GEOMETRY, self.MAJOR_TAGS.NODES, content_type="text",  dtype=dtype)

    def get_elements_dict(self, dtype: np.dtype = np.int64) -> Dict[str, np.ndarray]:
        """
        Returns elements content from .feb file as a dict with keys representing elements names\
        and values representing node ids as a numpy array of specified dtype.\

        Args:
        dtype (np.dtype): Numpy dtype.

        Returns:
            dict: {tag_name: [node_ids]}

        Example:
            feb.get_elements()
        """
        return self.get_tag_data(self.LEAD_TAGS.GEOMETRY, self.MAJOR_TAGS.ELEMENTS, content_type="text",  dtype=dtype)

    def get_surfaces_dict(self, dtype: np.dtype = np.int64) -> Dict[str, np.ndarray]:
        """
        Returns surface content from .feb file as a dict with keys representing surface names\
        and values representing node ids as a numpy array of specified dtype.\

        Args:
        dtype (np.dtype): Numpy dtype.

        Returns:
            dict: {tag_name: [node_ids]}

        Example:
            feb.get_surfaces()
        """
        try:
            return self.get_tag_data(self.LEAD_TAGS.GEOMETRY, self.MAJOR_TAGS.SURFACE, content_type="text",  dtype=dtype)
        except:
            return {}

    # Faster access to specific geometry data (directly providing key or using first index as default)
    
    def get_nodes(self, key=0, dtype=np.float32) -> np.ndarray:
        nodes_dict = self.get_nodes_dict(dtype=dtype)
        if key in nodes_dict.keys():
            return nodes_dict[key]
        elif isinstance(key, int):
            return list(nodes_dict.values())[key]
        else:
            raise KeyError(f"Key '{key}' not found in nodes_dict. Available keys: {list(nodes_dict.keys())}")
        
    def get_elements(self, key=0, dtype=np.int64) -> np.ndarray:
        elements_dict = self.get_elements_dict(dtype=dtype)
        if key in elements_dict.keys():
            return elements_dict[key]
        elif isinstance(key, int):
            return list(elements_dict.values())[key]
        else:
            raise KeyError(f"Key '{key}' not found in elements_dict. Available keys: {list(elements_dict.keys())}")
    
    def get_surfaces(self, key=0, dtype=np.int64) -> np.ndarray:
        surfaces_dict = self.get_surfaces_dict(dtype=dtype)
        if key in surfaces_dict.keys():
            return surfaces_dict[key]
        elif isinstance(key, int):
            return list(surfaces_dict.values())[key]
        else:
            raise KeyError(f"Key '{key}' not found in surfaces_dict. Available keys: {list(surfaces_dict.keys())}")
    
    # ID data from tags
    # ------------------------------

    def get_nodesets(self, dtype=np.int64) -> Dict[str, np.ndarray]:
        """
          Returns a dict with keys representing node set names and values \
          representing corresponding node ids as a numpy array of specified dtype.\

          Args:
            dtype (np.dtype): Numpy dtype.

            Returns:
                dict: {tag_name: [node_ids]}

            Example:
                feb.get_nodesets()
        """
        try:
            return self.get_tag_data(self.LEAD_TAGS.GEOMETRY, self.MAJOR_TAGS.NODESET, content_type="id", dtype=dtype)
        except:
            return {}
    
    def get_materials(self) -> Dict[str, Material]:
        all_mat_data = {}
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
            mat_name = mat_attrib.pop("name", "Unnamed Material")

            # Create a Material named tuple for the current material
            current_material = Material(id=mat_id, name=mat_name, parameters=parameters, attributes=mat_attrib)

            # Use material name or ID as the key in the dictionary
            mat_key = mat_name if mat_name else mat_id
            all_mat_data[mat_key] = current_material

        return all_mat_data

    # Loads
    # ------------------------------
    
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
            current_load = NodalLoad(
                bc=load.attrib.get("bc", "UndefinedBC"),  # Default to 'UndefinedBC' if not specified
                node_set=load.attrib.get("node_set", f"UnnamedNodeSet{i}"),  # Default to an indexed name if not specified
                scale=scale_value,
                load_curve=scale_data.attrib.get("lc", "NoCurve")  # Default to 'NoCurve' if not specified
            )

            # Add the new NodalLoad to the list
            nodal_loads.append(current_load)

        return nodal_loads
        
    def get_pressure_loads(self) -> Dict[str, PressureLoad]:
        press_loads = {}
        for i, load in enumerate(self.loads.findall("surface_load")):
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

                # Use the surface name as the key in the dictionary
                press_loads[current_load.surface] = current_load

        return press_loads
    
    def get_loadcurves(self, dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
        """
        Returns loadcurve content from .feb file as a dict with keys representing loadcurve names\
        and values representing data as a numpy array of specified dtype.\
        Loadcurve data is stored in the LoadData tag.\

        Args:
            dtype (np.dtype, optional): _description_. Defaults to np.float32.

        Returns:
            Dict[str, np.ndarray]: _description_
        """
        try:
            return self.get_tag_data(self.LEAD_TAGS.LOADDATA, self.MAJOR_TAGS.LOADCURVE, content_type="text", dtype=dtype)
        except:
            return {}
        
    # Boundary conditions
    # ------------------------------
    
    def get_boundary_conditions(self) -> List[Union[FixCondition, RigidBodyCondition, BoundaryCondition]]:
        if self.boundary() is None:
            return []
        
        boundary_conditions_list = []
        for elem in self.boundary:
            if elem.tag == 'fix':
                # Create an instance of FixCondition for each 'fix' element
                fix_condition = FixCondition(bc=elem.attrib['bc'], node_set=elem.attrib['node_set'])
                boundary_conditions_list.append(fix_condition)
            
            elif elem.tag == 'rigid_body':
                # Gather all 'fixed' sub-elements for a 'rigid_body'
                fixed_axes = [FixedAxis(bc=fixed.attrib['bc']) for fixed in elem.findall('fixed')]
                # Create an instance of RigidBodyCondition for each 'rigid_body' element
                rigid_body_condition = RigidBodyCondition(material=elem.attrib['mat'], fixed_axes=fixed_axes)
                boundary_conditions_list.append(rigid_body_condition)
            
            else:
                # Fallback to a generic BoundaryCondition for unrecognized types
                generic_condition = BoundaryCondition(type=elem.tag, attributes=elem.attrib)
                boundary_conditions_list.append(generic_condition)

        return boundary_conditions_list
        
    def get_nodal_data(self, dtype=np.float32) -> List[NodalData]:
        nodal_data_list = []
        for data in self.meshdata.findall(self.MAJOR_TAGS.NODE_DATA.value):
            _this_data = [float(x.text) if x.text.isdigit() else x.text for x in data.findall("node")]
            ref = data.attrib["node_set"]
            name = data.attrib["name"]
            
            # Create a NodalData instance
            current_data = NodalData(
                node_set=ref,
                name=name,
                data=np.array(_this_data, dtype=dtype)  # Ensure data is in the correct dtype
            )

            # Add the NodalData instance to the list
            nodal_data_list.append(current_data)

        return nodal_data_list
    
    def get_surface_data(self, dtype=np.float32) -> List[SurfaceData]:
        surf_data_list = []
        for data in self.meshdata.findall(self.MAJOR_TAGS.SURFACE_DATA.value):
            _this_data = [float(x.text) if x.text.isdigit() else x.text for x in data.findall("surf")]
            ref = data.attrib["surf_set"]
            name = data.attrib["name"]
            
            # Create a NodalData instance
            current_data = NodalData(
                node_set=ref,
                name=name,
                data=np.array(_this_data, dtype=dtype)  # Ensure data is in the correct dtype
            )

            # Add the NodalData instance to the list
            surf_data_list.append(current_data)

        return surf_data_list
    
    def get_element_data(self, dtype=np.float32) -> List[ElementData]:
        elem_data_list = []
        for data in self.meshdata.findall(self.MAJOR_TAGS.ELEMENT_DATA.value):
            _this_data = [float(x.text) if x.text.isdigit() else x.text for x in data.findall("elem")]
            ref = data.attrib["elem_set"]
            name = data.attrib["name"]
            
            # Create a NodalData instance
            current_data = NodalData(
                node_set=ref,
                name=name,
                data=np.array(_this_data, dtype=dtype)  # Ensure data is in the correct dtype
            )

            # Add the NodalData instance to the list
            elem_data_list.append(current_data)

        return elem_data_list


    # =========================================================================================================
    # Add methods
    # =========================================================================================================
    
    # Main geometry data
    # ------------------------------
    
    