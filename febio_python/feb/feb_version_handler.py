from ._feb_25 import Feb25
from ._feb_30 import Feb30
from .bases import FebBaseObject
from typing import Union
from xml.etree import ElementTree as ET
from pathlib import Path

# from typing import NewType

# FebTypes = NewType('FebTypes', Union[Feb25, Feb30])

def Feb(filepath: Union[str, Path]=None, 
        tree: Union[ET.ElementTree, None] = None, 
        root: Union[ET.Element, None] = None, 
        version: float = None) -> Union[Feb25, Feb30]:
    """Create a FEB object based on the version of the FEB.
    """
    # version = determine_version(filepath)
    if version is None:
        version = FebBaseObject(tree=tree, root=root, filepath=filepath).version
    
    if isinstance(version, (str)):
        try:
            version = float(str)
        except Exception as e:
            raise RuntimeError(f"Version should be a float, not {type(version)}") from e                   
    
    if version == 2.5:
        return Feb25(tree=tree, root=root, filepath=filepath)
    elif version == 3.0:
        return Feb30(tree=tree, root=root, filepath=filepath)
    else:
        raise ValueError(f"Unsupported version: {version}. Supported versions are 2.5 and 3.0.")
