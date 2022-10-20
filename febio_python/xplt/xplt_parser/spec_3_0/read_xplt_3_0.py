# import struct
# import json
from os import path, curdir
import numpy as np
# import sys
import tempfile
import pathlib
from collections import deque

from ..common.utils import *  # pylint: disable=unused-wildcard-import
from .utils import *

from ..common.database import TAGS, ELEM_TYPES, NODES_PER_ELEM, R_KEYS

# ==  MACROS == #
PATH_DATABASE = pathlib.Path(path.dirname(__file__)).parent/"common"/"database"

# FILEPATH_TAGSDATA = PATH_DATABASE/"tags.json"
# FILEPATH_ELEMTYPES = PATH_DATABASE/"element_types.json"
# FILEPATH_NDSPERELEM = PATH_DATABASE/"nodes_per_elem.json"

# read database
# TAGS = read_json(FILEPATH_TAGSDATA)
# ELEMS_TYPES = read_json(FILEPATH_ELEMTYPES)
# NODES_PER_ELEM = read_json(FILEPATH_NDSPERELEM)

# INV_TAGS = dict(zip(TAGS.values(), TAGS.keys()))


# def get_filesize(bf):
#     bf.seek(0, 2)
#     filesize = bf.tell()
#     bf.seek(0, 0)

#     if filesize == 0:
#         raise(ValueError("Input xplit file size is zero. Check file."))
#     return filesize


def check_fileformat(bf, verbose):
    if(TAGS['FEBIO'] == read_bytes(bf)):
        console_log('Correct FEBio format', 2, verbose)
    else:
        raise(ValueError("Input XPLIT file does not have the correct format."))


def check_fileversion(bf, verbose):
    version = read_bytes(bf)
    if(version == TAGS['VERSION_3_0']):
        console_log('Current version is: %d' % version, 2, verbose)
    elif (version == 49):
        console_log('Current spec version is: 3.0 -> %d | WARNING: Docs say version should be 8, but it is 49.' % version, 2, verbose)
    else:
        raise(ValueError("Incorrect XPLIT file version: {}".format(version)))


def read_xplt_3_0(xplit_filepath, nstate=-1, verbose=0):
    
    # open binary file
    bf = open(xplit_filepath, "rb")

    # get file size and check if its not empty
    filesize = get_file_size(bf)

    # check if file format meets requirement
    check_fileformat(bf, verbose)

    console_log("\n___Begin read header___\n", 1, verbose)

    # move cursor to "ROOT"
    search_block(bf, TAGS, 'ROOT', verbose=verbose)
    # move cursor to "HEADER"
    search_block(bf, TAGS, 'HEADER', verbose=verbose)
    # move cursor to "HDR_VERSION" -> feb file version
    search_block(bf, TAGS, 'HDR_VERSION', verbose=verbose)
    check_fileversion(bf, verbose)

    # move cursor to "HDR_COMPRESSION" and check wheter states are compressed or not
    a = search_block(bf, TAGS, 'HDR_COMPRESSION', verbose=verbose)
    file_is_compressed = read_bytes(bf, nb=a)
    console_log("Compression: %d" % file_is_compressed, 2, verbose)

    # -- Explore Dictionary --
    console_log("\n___Begin read dictionary___\n", 1, verbose)
    states_dict = read_dictionary(bf, TAGS, verbose=verbose)
    # console_log("Data: {}".format(item_names), 1, verbose)
    
    # -- Explore MESH section --
    console_log("\n___Begin read mesh___\n", 1, verbose)
    search_block(bf, TAGS, 'MESH', verbose=verbose)
    
    # --> explore node section
    console_log("__node section__\n", 1, verbose)
    nodes_coords = read_nodes_coords(bf, TAGS, verbose=verbose)
    
    # -- Explore DOMAIN --
    dom_data = read_domain(bf, TAGS, ELEM_TYPES, NODES_PER_ELEM, verbose=verbose)
    
    # console_log("Number of elems: {}".format(dom_n_elems), 1, verbose)
    # console_log("Element types: {}".format(dom_elem_types), 1, verbose)
    # console_log("Element materials: {}".format(dom_mat_ids), 1, verbose)

    # -- Explore SURFACE_SECTION --
    surface_data = read_surface_section(bf, TAGS, filesize=filesize, verbose=verbose)
    # console_log("Surface names: {}".format(surface_names), 1, verbose)

    # -- Explore NODESET_SECTION --
    nodeset_data = read_nodeset_section(bf, TAGS, verbose=verbose)
    # console_log("Nodeset names: {}".format(nodeset_names), 1, verbose)

    # -- Explore Parts section --
    # material = read_materials(bf, TAGS, verbose=verbose)
    
    # -- Read states --
    console_log("\n____Begin read states____", 1, verbose)
    state_data = read_state(bf, TAGS, states_dict, decompress=file_is_compressed, verbose=verbose)
    
    console_log('\nData extraction done.', 1, verbose)

    # close binary file
    bf.close()
    
    return {
            R_KEYS.SPEC_VERSION.value: 3.0,                     # spec version
        
            # quick references
            R_KEYS.N_NODES.value: nodes_coords.shape[0],        # number of nodes
            R_KEYS.N_DOMS.value: dom_data["n_doms"],            # number of domains
            R_KEYS.N_ELEMS.value: dom_data["n_elems"],          # list of number of elements per domain
            R_KEYS.N_STATES.value: state_data["n"],             # number of states
                
            # mesh reference data 
            R_KEYS.NODES.value: nodes_coords,                   # nodes coordinats
            R_KEYS.ELEMENTS.value: dom_data["elems"],           # elements (per domain)
            
            # surface data
            R_KEYS.FACETS.value: surface_data["facets"],             # facets as array
            R_KEYS.FACETS_BY_NAME.value: surface_data["by_name"],    # facett as {name: facet}
            R_KEYS.FACETS_BY_ID.value: surface_data["by_id"],        # facett as {id: facet}
            
            # nodes data
            R_KEYS.NODESETS.value: nodeset_data["nodesets"],         # nodesets
            R_KEYS.NODESETS_BY_NAME.value: nodeset_data["by_name"],  # nodesett as {name: nodeset}
            R_KEYS.NODESETS_BY_ID.value: nodeset_data["by_id"],      # nodesett as {name: nodeset}
            
            # states data
            R_KEYS.TIMESTEPS.value: state_data["time"],             # timesteps as array
            R_KEYS.STATES.value: state_data["data"],                # state data by timestep
        }   