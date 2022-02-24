from .functions import read_json
from os import path

PATH_DATABASE = "database"
FILEPATH_TAGSDATA  = path.join(path.dirname(__file__), PATH_DATABASE, "tags.json")
FILEPATH_ELEMTYPES = path.join(path.dirname(__file__), PATH_DATABASE, "element_types.json")
FILEPATH_NDSPERELEM = path.join(path.dirname(__file__), PATH_DATABASE, "nodes_per_elem.json")


XPLT_TAGS = read_json(FILEPATH_TAGSDATA)
XPLT_ELEM_TYPES = read_json(FILEPATH_ELEMTYPES)
XPLT_NODES_PER_ELEM = read_json(FILEPATH_NDSPERELEM)
