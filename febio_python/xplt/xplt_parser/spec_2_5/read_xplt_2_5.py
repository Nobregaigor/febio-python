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


def check_fileversion(bf, verbose):
    version = read_bytes(bf)
    if(version == TAGS['VERSION_2_5']):
        console_log('Current version is: %d' % version, 2, verbose)
    else:
        raise(ValueError("Incorrect XPLIT file version: {}".format(version)))


def read_xplt_2_5(xplit_filepath, nstate=-1, verbose=0):

    # open binary file
    bf = open(xplit_filepath, "rb")

    # get file size and check if its not empty
    filesize = get_file_size(bf)

    # check if file format meets requirement
    check_fileformat(bf, TAGS, verbose=verbose)

    console_log("\n___Begin read header___\n", 1, verbose)

    # move cursor to "ROOT"
    search_block(bf, TAGS, 'ROOT', verbose=verbose)
    # move cursor to "HEADER"
    search_block(bf, TAGS, 'HEADER', verbose=verbose)
    # move cursor to "HDR_VERSION" -> feb file version
    search_block(bf, TAGS, 'HDR_VERSION', verbose=verbose)
    check_fileversion(bf, verbose)

    # move cursor to "HDR_NODES" -> get number of nodes
    search_block(bf, TAGS, 'HDR_NODES', verbose=verbose)
    n_nodes = read_bytes(bf)
    console_log("Number of nodes: %d" % n_nodes, 1, verbose)

    # move cursor to "HDR_COMPRESSION" and check wheter states are compressed or not
    a = search_block(bf, TAGS, 'HDR_COMPRESSION', verbose=verbose)
    file_is_compressed = read_bytes(bf, nb=a)
    console_log("Compression: %d" % file_is_compressed, 2, verbose)

    # -- Explore Dictionary --
    states_dict = read_dictionary(bf, TAGS, verbose=verbose)
    # console_log("Data: {}".format(item_names), 1, verbose)

    # -- Explore MATERIALS --
    mat_data = read_materials(bf, TAGS, verbose=verbose)
    # console_log("Materials: {}".format(mat_names), 1, verbose)

    console_log("\n___Begin read mesh___\n", 1, verbose)

    # -- Explore MESH-> NODESCOORDS --
    nodes_coords = read_nodes_coords(bf, TAGS, verbose=verbose)

    # -- Explore DOMAIN --
    dom_data = read_domain(bf, TAGS, ELEM_TYPES, NODES_PER_ELEM, verbose=verbose)
    # console_log("Number of elems: {}".format(dom_n_elems), 1, verbose)
    # console_log("Element types: {}".format(dom_elem_types), 1, verbose)
    # console_log("Element materials: {}".format(dom_mat_ids), 1, verbose)

    # -- Explore SURFACE_SECTION --
    surface_data = read_surface_section(bf, TAGS, filesize=filesize, verbose=verbose)
    # console_log("Surface names: {}".format(surface_names), 1, verbose)
    # console_log("Faces: {}".format(faces), 1, verbose)
    # if len(faces) > 0:
    #     console_log("face sample [0]: {}".format(faces[0]), 2, verbose)

    # -- Explore NODESET_SECTION --
    nodeset_data = read_nodeset_section(bf, TAGS, verbose=verbose)
    # console_log("Nodeset names: {}".format(nodeset_names), 1, verbose)
    
    console_log("\n____Begin read states____", 1, verbose)
    state_data = read_state(bf, TAGS, states_dict, decompress=file_is_compressed, verbose=verbose)
    
    console_log('\nData extraction done.', 1, verbose)
    
    # close binary file
    bf.close()
    
    return {
            R_KEYS.SPEC_VERSION.value: 2.5,                     # spec version
        
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
    

    # # -- Decompress STATE data, if needed --
    # if file_is_compressed:
    #     console_log("\n_____Begin decompressing states_____", 1, verbose)
    #     alldata = bf.read(-1)
    #     bf.close()
    #     decompressed = decompress(alldata, verbose=verbose)
    #     bf = tempfile.TemporaryFile()
    #     bf.write(decompressed)
    #     bf.seek(0)
    #     filesize = len(decompressed)

    # # -- Read STATES --
    # console_log("\n____Begin read states____", 1, verbose)
    # states_data = deque()
    # states_time = deque()
    # cur_state = 0
    # if nstate != -1:  # skip the first nstate states if a specific nstate is given
    #     while check_block(bf, TAGS, 'STATE') and (cur_state < nstate):
    #         read_state(bf, TAGS, exit_at_next_state=True)
    #         cur_state += 1

    #     if cur_state != nstate:  # check if it could reach desired state
    #         console_log("State %d does not exist!" % nstate, 1, verbose)
    #         console_log("Current state is, {}".format(cur_state), 1, verbose)
    #         return -1

    # # extract information from states
    # while check_block(bf, TAGS, 'STATE'):
    #     state_time = read_state(bf, TAGS, verbose=verbose)
    #     cur_state += 1
    #     (n_node_data, item_def_doms, item_data) = read_nodes_data(
    #         bf, TAGS, item_names, item_types, verbose=verbose)
    #     (item_def_doms, _, item_data) = read_elems_data(bf, TAGS, item_names, item_types, n_node_data,
    #                                                     item_def_doms, filesize, item_data, verbose=verbose)
    #     states_data.append(item_data)
    #     states_time.append(state_time)
    #     if bf.tell() == filesize or nstate != -1:  # exit if reached end or desired nstate
    #         break

    # if nstate != -1:  # skip rest of states
    #     cur_state = nstate
    #     if bf.tell() < filesize:
    #         while check_block(bf, TAGS, 'STATE'):
    #             read_state(bf, TAGS, exit_at_next_state=True, verbose=verbose)
    #             cur_state += 1
    #             if bf.tell() == filesize:
    #                 break

    # console_log('\nData extraction done.', 1, verbose)
    # console_log("Number of states read: {}".format(
    #     len(states_data)), 1, verbose)

    # if bf.tell() == filesize:
    #     console_log('\n== EOF reached. ==', 1, verbose)

    # # close binary file
    # bf.close()

    # if len(states_data) == 0:
    #     return -1

    # return {
    #     # nodes data
    #     "n_nodes": n_nodes,
    #     "nodes": nodes_coords,
    #     # elems data
    #     "n_elems": dom_n_elems,
    #     "elem_types": dom_elem_types,
    #     "elem_mats": dom_mat_ids,
    #     "elems": dom_elements,
    #     # materials data
    #     "materials": mat_names,
    #     "materials_id": mat_ids,
    #     # surface data
    #     "surface_ids": surface_ids,
    #     "surface_faces": surface_faces,
    #     "surface_names": surface_names,
    #     "surface_nodes": surface_nodes,
    #     "faces": faces,
    #     "face_ids": face_ids,
    #     # nodeset data
    #     "nodeset_ids": nodeset_ids,
    #     "nodeset_names": nodeset_names,
    #     "nodeset_nodes": nodeset_nodes,
    #     # states data
    #     "data": np.array(states_data, dtype="object"),
    #     "data_keys": item_names,
    #     "data_format": item_formats,
    #     "data_map": item_def_doms,
    #     "n_states": len(states_data),
    #     "timesteps": states_time,
    # }
