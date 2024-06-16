from enum import IntEnum
from collections import deque, defaultdict
import numpy as np
import tempfile
from typing import List, Tuple

from febio_python.core.enums import XPLT_TAGS as TAGS
from febio_python.core.enums import XPLT_DATA_TYPES as DATA_TYPES
from febio_python.core.enums import XPLT_DATA_DIMS as DATA_DIMS

from febio_python.core.element_types import FEBioElementValue
from febio_python.core.element_types import NumberOfNodesPerElement

from febio_python.utils.log import console_log

from .._binary_file_reader_helpers import (
    read_bytes,
    search_block,
    check_block,
    get_file_size)

from .._decompress import decompress

from febio_python.core import (
    Nodes,
    Elements,
    NodeSet,
    # Xplt data
    XpltMesh,
    XpltMeshPart,
    StatesDict,
    StateVariable,
    StateData,
    States
)


# =============================================================================
# CHECKS
# =============================================================================

def check_fileformat(bf, root_tag: IntEnum, verbose: int = 0):
    """
    Check if the binary file starts with a specific tag indicating the correct file format.

    Args:
        bf: Binary file object.
        root_tag: IntEnum with the value for the root tag (e.g. TAGS.FEBIO)
        verbose: Verbosity level for logging.

    Throws:
        ValueError: If the file does not start with the expected format tag.
    """
    # Attempt to read the format tag from the file
    try:
        file_format_tag = read_bytes(bf)
    except Exception as e:
        console_log(f"Failed to read file format: {e}", 2, verbose)
        raise ValueError("Failed to read from the file, possibly due to an IO error.")

    # Check if the read tag matches the expected 'FEBIO' tag
    if root_tag.value == file_format_tag:
        console_log('Correct FEBio format', 2, verbose)
    else:
        raise ValueError("Input XPLIT file does not have the correct format.")


# =============================================================================
# READ DICTIONARY
# =============================================================================

def _read_dictionary_items(bf, verbose):
    item_types = deque()
    item_formats = deque()
    item_names = deque()

    while check_block(bf, TAGS.DIC_ITEM):
        search_block(bf, TAGS.DIC_ITEM, verbose=verbose)

        # Read item type
        search_block(bf, TAGS.DIC_ITEM_TYPE, verbose=verbose)
        item_types.append(int(read_bytes(bf)))

        # Read item format
        search_block(bf, TAGS.DIC_ITEM_FMT, verbose=verbose)
        item_formats.append(int(read_bytes(bf)))

        # Read item name
        search_block(bf, TAGS.DIC_ITEM_NAME, verbose=verbose)
        i_name = bf.read(64)
        try:
            item_name = i_name.decode('ascii').split('\x00')[0]
        except UnicodeDecodeError:
            item_name = i_name.split(b'\x00')[0].decode('ascii')
        item_names.append(item_name)

    return item_types, item_formats, item_names


def read_dictionary(bf, verbose: int = 0) -> StatesDict:
    # Navigate to "DICTIONARY" and "DIC_NODAL"
    search_block(bf, TAGS.DICTIONARY, verbose=verbose)
    search_block(bf, TAGS.DIC_NODAL, verbose=verbose)

    # Read nodal items
    item_types, item_formats, item_names = _read_dictionary_items(bf, verbose)

    # Navigate to "DIC_DOMAIN" and read domain items
    search_block(bf, TAGS.DIC_DOMAIN, verbose=verbose)
    domain_types, domain_formats, domain_names = _read_dictionary_items(bf, verbose)

    # Append domain items to nodal items
    item_types.extend(domain_types)
    item_formats.extend(domain_formats)
    item_names.extend(domain_names)

    console_log("read_dictionary:", 2, verbose, indent=1)
    console_log("items: [item_types, item_formats, item_names]", 2, verbose, indent=2)
    console_log([item_types, item_formats, item_names], 2, verbose, indent=2)

    return StatesDict(types=item_types, formats=item_formats, names=item_names)

# =============================================================================
# READ MESH
# =============================================================================


def _read_nodes_section(bf, verbose=0) -> List[Nodes]:
    search_block(bf, TAGS.NODE_SECTION, verbose=verbose)
    search_block(bf, TAGS.NODE_HEADER, verbose=verbose)

    n_nodes = read_bytes(bf, search_block(bf, TAGS.NODE_N_NODES, verbose=verbose))
    n_dims = read_bytes(bf, search_block(bf, TAGS.NODE_DIM, verbose=verbose))
    nodes_shape = (n_nodes, n_dims)
    console_log(f"n_nodes: {n_nodes} | n_dim: {n_dims} -> shape {nodes_shape}", 3, verbose=verbose)

    # Prepare format string and read node data
    f_format = "i" + "f" * n_dims
    node_data = np.array(read_bytes(bf, search_block(bf, TAGS.NODE_COORDS_3_0, verbose=verbose), format=f_format * n_nodes), dtype=float).reshape((n_nodes, n_dims + 1))

    # Extract node IDs and coordinates
    node_ids = node_data[:, 0].astype(int)  # Assuming node IDs are the first column
    node_coords = node_data[:, 1:]  # Assuming subsequent columns are coordinates

    console_log("---read_nodes_section", 2, verbose)
    console_log(f"->number of node coords: {n_nodes}", 2, verbose)
    console_log(f"->number of node dims: {n_dims}", 2, verbose)
    console_log(f"->extracted coords shape: {node_coords.shape}", 2, verbose)
    console_log(node_coords, 3, verbose)

    # In FEBio-XPLT data, nodes are grouped into one single array. They do NOT have "names"
    # -- This is now done using "parts" and "Nodesets"
    nodes = Nodes(name=None, coordinates=node_coords, ids=node_ids)
    return [nodes]


def _read_domain_section(bf, verbose=0) -> List[Elements]:
    console_log("---read_domain", 2, verbose)

    domains = list()

    # move pointer to domain section
    search_block(bf, TAGS.DOMAIN_SECTION, verbose=verbose)

    previous_id = 0
    while check_block(bf, TAGS.DOMAIN):
        console_log("--reading domain content", 3, verbose)

        # move pointer to domain content and header
        search_block(bf, TAGS.DOMAIN, verbose=verbose)

        # -- Explore domain HEADER --
        search_block(bf, TAGS.DOMAIN_HEADER, verbose=verbose)
        # Read element type, part ID, and number of elements
        search_block(bf, TAGS.DOM_ELEM_TYPE, verbose=verbose)
        elem_type = int(read_bytes(bf))
        search_block(bf, TAGS.DOM_PART_ID, verbose=verbose)
        part_id = int(read_bytes(bf))
        search_block(bf, TAGS.DOM_N_ELEMS, verbose=verbose)
        num_elems = int(read_bytes(bf))

        # Get element type name and number of nodes per element
        elem_type_name = FEBioElementValue(elem_type).name
        n_nodes_per_element = NumberOfNodesPerElement[elem_type_name].value

        if verbose >= 3:
            console_log("--elem_type: {}".format(elem_type), 3, verbose)
            console_log("--elem_type_name: {}".format(elem_type_name), 3, verbose)
            console_log("--n_nodes_per_element: {}".format(n_nodes_per_element), 3, verbose)

        # Prepare for reading elements list
        search_block(bf, TAGS.DOM_ELEM_LIST, verbose=verbose)
        elements = deque()
        for _ in range(num_elems):
            a = search_block(bf, TAGS.ELEMENT, verbose=verbose)
            element = read_bytes(bf, nb=a, format="I" * (n_nodes_per_element + 1))
            elements.append(element)

        # Convert elements to numpy array
        elements = np.array(elements, dtype=int)
        # domains.append(domain_info)
        console_log("--elements shape: {}".format(elements.shape), 3, verbose)

        new_elements = Elements(
            name=None,
            mat=None,
            part=part_id,
            type=elem_type_name,
            connectivity=elements,
            ids=np.arange(previous_id, previous_id + num_elems)
        )
        previous_id += num_elems
        domains.append(new_elements)

    console_log(f"->number of domains: {len(domains)}", 2, verbose)
    console_log(f"->extracted domains: {domains}", 3, verbose)

    return domains


def _read_surface_section(bf, verbose=0) -> List[Elements]:

    console_log("---read_surface_section", 2, verbose)

    all_surfaces = list()

    # surfaces are optional.
    # So we need to check if there are surfaces in our file
    if search_block(bf, TAGS.SURFACE_SECTION, verbose=verbose) > 0:

        previous_el_id = 0
        while check_block(bf, TAGS.SURFACE):
            # move poiter to next surface
            search_block(bf, TAGS.SURFACE, verbose=verbose)

            # explore surface header
            search_block(bf, TAGS.SURFACE_HEADER, verbose=verbose)
            # get surface id
            a = search_block(bf, TAGS.SURFACE_ID, verbose=verbose)
            surf_id = read_bytes(bf, nb=a)
            # get number of facets in surface
            search_block(bf, TAGS.SURFACE_N_FACETS, verbose=verbose)
            n_facets = int(read_bytes(bf, nb=a))
            # get surface name (if it exists)
            a = search_block(bf, TAGS.SURFACE_NAME, verbose=verbose)
            name = None
            if a > 0:
                i_name = bf.read(a)  # from docs, here should be CHAR64, but its DWORD from above
                try:
                    name = i_name.decode('ascii').split('\x00')[-1]
                except UnicodeDecodeError:
                    name = str(i_name).split('\\x00')[-1]

            # get max facet nodes
            a = search_block(bf, TAGS.MAX_FACET_NODES, verbose=verbose)
            nodes_per_facet = int(read_bytes(bf))

            # Get element type based on number of nodes per facet
            if nodes_per_facet == 4:
                surf_type = "QUAD"  # to aviod identifying it as TETRA
            else:
                surf_type = NumberOfNodesPerElement[nodes_per_facet].name

            if verbose >= 3:
                console_log("--surf_id: {}".format(surf_id), 3, verbose)
                console_log("--n_facets: {}".format(n_facets), 3, verbose)
                console_log("--name: {}".format(name), 3, verbose)
                console_log("--nodes_per_facet: {}".format(nodes_per_facet), 3, verbose)

            # move pointer to facet list
            search_block(bf, TAGS.FACET_LIST, verbose=verbose)

            # read facets
            surf_facets = deque()
            for _ in range(n_facets):
                # move pointer to next facet
                a = search_block(bf, TAGS.FACET, verbose=verbose)
                # read facet nodes
                facet = np.array(read_bytes(bf, nb=a, format="I" * (nodes_per_facet + 2)), dtype=int)
                # according to docs, we should ignore first two elements
                facet = facet[2:]
                # add facet to deque
                surf_facets.append(facet)

            try:
                surf_facets = np.array(surf_facets, dtype="int32")
            except Exception:
                surf_facets = np.array(surf_facets, dtype="object")

            new_surface = Elements(
                name=name,
                mat=None,
                part=surf_id,
                type=surf_type,
                connectivity=surf_facets,
                ids=np.arange(previous_el_id, previous_el_id + n_facets)
            )
            previous_el_id += n_facets

            all_surfaces.append(new_surface)

        console_log(f"->number of surfaces: {len(all_surfaces)}", 2, verbose)
        console_log(f"->extracted surfaces: {all_surfaces}", 3, verbose)

    return all_surfaces


def _read_nodeset_section(bf, verbose=0) -> List[NodeSet]:
    all_nodesets = list()
    # nodesets are optional.
    # So we need to check if there are nodesets in our file
    if search_block(bf, TAGS.NODESET_SECTION, verbose=verbose) > 0:

        while check_block(bf, TAGS.NODESET):
            # move pointer to next nodeset
            search_block(bf, TAGS.NODESET, verbose=verbose)
            # explore nodeset header
            search_block(bf, TAGS.NODESET_HEADER, verbose=verbose)
            # get nodeset id
            a = search_block(bf, TAGS.NODESET_ID, verbose=verbose)
            nodeset_id = int(read_bytes(bf, nb=a))
            # get number of nodes in nodeset
            search_block(bf, TAGS.NODESET_N_NODES, verbose=verbose)
            n_nodes = int(read_bytes(bf, nb=a))
            # nodeset name (if exists)
            a = search_block(bf, TAGS.NODESET_NAME, verbose=verbose)
            name = None
            if a > 0:
                name = bf.read(a)  # from docs, here should be CHAR64, but its DWORD from above
                try:
                    name = name.decode('ascii').split('\x00')[-1]
                except UnicodeDecodeError:
                    name = str(name).split('\\x00')[-1]
            # Log info if verbose is high enough
            if verbose >= 3:
                console_log("--nodeset_id: {}".format(nodeset_id), 3, verbose)
                console_log("--n_nodes: {}".format(n_nodes), 3, verbose)
                console_log("--name: {}".format(name), 3, verbose)
            # move pointer to nodelist
            a = search_block(bf, TAGS.NODESET_LIST, verbose=verbose)
            nodes_ids = np.array(read_bytes(bf, nb=a, format="I" * n_nodes), dtype="int")

            new_nodeset = NodeSet(
                name=name if name is not None else f"nodeset_{nodeset_id}",
                ids=nodes_ids,
            )

            all_nodesets.append(new_nodeset)
    return all_nodesets


def _read_parts_section(bf, verbose=0) -> List[XpltMeshPart]:
    all_parts = list()
    # Move pointer to PARTS section
    if search_block(bf, TAGS.PART_SECTION, verbose=verbose) > 0:
        console_log("---read_parts_section", 2, verbose)

        while check_block(bf, TAGS.PART):
            console_log("--reading part content", 3, verbose)

            # Move pointer to part content and header
            search_block(bf, TAGS.PART, verbose=verbose)

            # Read part ID
            search_block(bf, TAGS.PART_ID, verbose=verbose)
            part_id = int(read_bytes(bf))

            # Read part name
            a = search_block(bf, TAGS.PART_NAME, verbose=verbose)
            part_name = None
            if a > 0:
                part_name = bf.read(a).decode('ascii').strip('\x00')

            if verbose >= 3:
                console_log("--part_id: {}".format(part_id), 3, verbose)
                console_log("--part_name: {}".format(part_name), 3, verbose)

            new_part = XpltMeshPart(
                id=part_id,
                name=part_name,
            )
            all_parts.append(new_part)

        console_log(f"->number of parts: {len(all_parts)}", 2, verbose)
        console_log(f"->extracted parts: {all_parts}", 3, verbose)

    return all_parts


def read_mesh(bf, verbose=0) -> XpltMesh:
    # Move cursor to MESH section
    search_block(bf, TAGS.MESH, verbose=verbose)
    # Read NODES
    nodes = _read_nodes_section(bf, verbose)
    # Read DOMAINS (ELEMENTS)
    domains = _read_domain_section(bf, verbose)
    # Read SURFACES
    surfaces = _read_surface_section(bf, verbose)
    # Read NODESETS
    nodesets = _read_nodeset_section(bf, verbose)
    # Read PARTS
    parts = _read_parts_section(bf, verbose)
    # Return
    xplt_mesh = XpltMesh(
        nodes=nodes,
        elements=domains,
        surfaces=surfaces,
        nodesets=nodesets,
        parts=parts
    )
    return xplt_mesh


# =============================================================================
# READ STATES
# =============================================================================

def _decompress_state(bf, verbose=0):
    console_log("Decompressing states...", 1, verbose, header=True)
    alldata = bf.read(-1)
    bf.close()
    decompressed = decompress(alldata, verbose=verbose)
    bf = tempfile.TemporaryFile()
    bf.write(decompressed)
    bf.seek(0)
    filesize = len(decompressed)
    if filesize == 0:
        raise ValueError("Decompression failed. File is empty.")
    return bf


def _process_state_var(bf, filesize: int, states_dict: StatesDict, offset=0, verbose=0) -> List[StateVariable]:
    all_data = deque()

    while check_block(bf, TAGS.STATE_VARIABLE, filesize=filesize):
        # move pointer to state variable
        search_block(bf, TAGS.STATE_VARIABLE, verbose=verbose)
        # get index of state item
        a = search_block(bf, TAGS.STATE_VAR_ID, verbose=verbose)
        var_idx = int(read_bytes(bf, nb=a)) - 1 + offset
        # determine data dims based on item type
        var_name = states_dict.names[var_idx]
        var_type = DATA_TYPES(states_dict.types[var_idx]).name
        # get data dim, based on item type
        var_dim = DATA_DIMS[var_type].value
        if verbose >= 3:
            console_log("--var_type: {}".format(var_type), 3, verbose=verbose)
            console_log("--var_dim: {}".format(var_dim), 3, verbose=verbose)
        # move pointer to actual data
        a = search_block(bf, TAGS.STATE_VAR_DATA, verbose=verbose)
        # read domain id -> not used in this package (for now)
        domain_id = int(read_bytes(bf))
        # read data size
        data_size = read_bytes(bf)
        # read data
        n_data = data_size // var_dim // 4
        try:
            var_data = read_bytes(bf, nb=data_size, format='f' * (n_data * var_dim))
            var_data = np.array(var_data, dtype="float32")
            var_data = var_data.reshape((n_data, var_dim))

            new_variable = StateVariable(
                name=var_name,
                dim=var_dim,
                dom=domain_id,
                data=var_data
            )
            all_data.append(new_variable)
        except Exception as e:
            console_log(f"Failed to read state variable: {var_name} due to {e}. Skipping..."
                        "Posssible reasons: "
                        "1. Mismatch between data size and expected size based on data type."
                        "(Based on some tests, this happens when febio tries to write/append 'standard' variables "
                        "like 'stress', but not all simulations have stress data. "
                        "2. Data is not in the expected format."
                        "3. Data is corrupted."
                        "Please check the simulation 'output' section and try to re-run the simulation, "
                        "or check the data integrity.", level=1, verbose=verbose)

    offset += len(all_data)

    all_data = list(all_data)

    return all_data, offset


def _search_for_state_variables(bf, filesize: int, states_dict: StatesDict, verbose=0) -> Tuple[deque, deque, deque, deque]:
    all_node_variables = deque()
    all_elem_variables = deque()
    all_surf_variables = deque()
    timesteps = deque()
    while search_block(bf, TAGS.STATE, verbose=verbose) > 0:

        # So we need to check if there are statess in our file
        if check_block(bf, TAGS.STATE_HEADER):

            # move pointer to next STATE
            search_block(bf, TAGS.STATE_HEADER, verbose=verbose)

            # get state time
            a = search_block(bf, TAGS.STATE_HEADER_TIME, verbose=verbose)
            time = read_bytes(bf, nb=a, format="f")
            timesteps.append(time)
            console_log("state_time: {}".format(time), 3, verbose=verbose)

            # move pointer to next STATE_DATA
            search_block(bf, TAGS.STATE_DATA, verbose=verbose)

            # extract data
            offset = 0

            # Check if there are nodal data, if so, extract it
            if check_block(bf, TAGS.NODE_DATA, filesize=filesize):
                # move pointer to next nodal data
                search_block(bf, TAGS.NODE_DATA, verbose=verbose)
                nodes_variables, n_nodes_variables = _process_state_var(bf, filesize, states_dict, verbose=verbose)
                all_node_variables.extend(nodes_variables)
                offset += n_nodes_variables

            # Check if there are element data, if so, extract it
            if check_block(bf, TAGS.ELEMENT_DATA, filesize=filesize):
                # move pointer to next element data
                search_block(bf, TAGS.ELEMENT_DATA, verbose=verbose)
                elems_variables, n_elems_variables = _process_state_var(bf, filesize, states_dict, offset, verbose=verbose)
                all_elem_variables.extend(elems_variables)
                offset += n_elems_variables

            # Check if there are surface data, if so, extract it
            if check_block(bf, TAGS.SURFACE_DATA, filesize=filesize):
                # move pointer to next surface data
                search_block(bf, TAGS.SURFACE_DATA, verbose=verbose)
                surfs_variables, n_surf_data = _process_state_var(bf, filesize, states_dict, offset, verbose=verbose)
                all_surf_variables.extend(surfs_variables)
                offset += n_surf_data
        else:
            console_log("No state was found.", 1, verbose=verbose)

        if bf.tell() == filesize:  # reach end of states (EOF)
            break

    return (all_node_variables, all_elem_variables, all_surf_variables, timesteps)


def _process_state_variables(state_variables: Tuple[deque, deque, deque, deque], verbose=0) -> StateData:
    all_node_variables, all_elem_variables, all_surf_variables, timesteps = state_variables

    # convert time steps to numpy array
    timesteps = np.array(timesteps, dtype="float32")

    nodes_data = defaultdict(deque)
    for node_var in all_node_variables:
        key = (node_var.name, node_var.dom)
        nodes_data[key].append(node_var.data)

    elems_data = defaultdict(deque)
    for elem_var in all_elem_variables:
        key = (elem_var.name, elem_var.dom)
        elems_data[key].append(elem_var.data)

    surfs_data = defaultdict(deque)
    for surf_var in all_surf_variables:
        key = (surf_var.name, surf_var.dom)
        surfs_data[key].append(surf_var.data)

    # Now we need to convert to StateData objects
    nodes_state_data = list()
    for key, value in nodes_data.items():
        name, dom = key
        value = np.array(value, dtype="float32")
        nodes_state_data.append(StateData(name=name, dom=dom, data=value))

    elems_state_data = list()
    for key, value in elems_data.items():
        name, dom = key
        value = np.array(value, dtype="float32")
        elems_state_data.append(StateData(name=name, dom=dom, data=value))

    surfs_state_data = list()
    for key, value in surfs_data.items():
        name, dom = key
        value = np.array(value, dtype="float32")
        surfs_state_data.append(StateData(name=name, dom=dom, data=value))

    # Now we need to convert to States object
    states = States(
        nodes=nodes_state_data,
        elements=elems_state_data,
        surfaces=surfs_state_data,
        timesteps=timesteps
    )
    return states


def read_state(bf, states_dict, decompress=False, verbose=0) -> States:

    # 1: Check if file is compressed and decompress if needed
    # -------------------------------------------------------
    if decompress:
        bf = _decompress_state(bf, verbose=verbose)
    else:
        console_log("File is not compressed.", 3, verbose=verbose)
    # need to re-read filesize to account for decompression
    filesize = get_file_size(bf)

    # 2: Search for state variables
    # -----------------------------

    state_variables = _search_for_state_variables(bf, filesize, states_dict, verbose=verbose)

    # 3: Process state variables
    # --------------------------
    states = _process_state_variables(state_variables, verbose=verbose)

    return states
