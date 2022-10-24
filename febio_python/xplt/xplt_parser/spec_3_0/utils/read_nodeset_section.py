

def read_nodeset_section(bf, TAGS, verbose=0):
    from ...common.utils import search_block, check_block, read_bytes, num_el_nodes, console_log
    from numpy import zeros as npzeros
    from collections import deque
    from numpy import array as nparray

    nodeset_ids   = deque()
    nodeset_names = deque()
    nodeset_nodes = deque()
    
    nodeset_data = {
            "nodesets": nparray([]),
            "by_name": nparray([]),
            "by_id": nparray([]),
        }

    # nodesets are optional. 
    # So we need to check if there are nodesets in our file
    if search_block(bf, TAGS, 'NODESET_SECTION', verbose=verbose) > 0:
    
        while check_block(bf, TAGS, 'NODESET'):
            # move pointer to next nodeset
            search_block(bf, TAGS, 'NODESET', verbose=verbose)
            # explore nodeset header
            search_block(bf, TAGS, 'NODESET_HEADER', verbose=verbose)
            # get nodeset id
            a = search_block(bf, TAGS, 'NODESET_ID', verbose=verbose)
            nodeset_id = int(read_bytes(bf, nb=a))
            nodeset_ids.append(nodeset_id)
            console_log("--nodeset_id: {}".format(nodeset_id),3,verbose)
            # get number of nodes in nodeset
            search_block(bf, TAGS, 'NODESET_N_NODES', verbose=verbose)
            n_nodes = int(read_bytes(bf, nb=a))
            console_log("--n_nodes: {}".format(n_nodes),3,verbose)
            # nodeset name (if exists)
            a = search_block(bf, TAGS, 'NODESET_NAME', verbose=verbose)
            if a > 0:
                i_name = bf.read(a) # from docs, here should be CHAR64, but its DWORD from above
                try:
                    i_name = i_name.decode('ascii').split('\x00')[-1]
                except UnicodeDecodeError:
                    i_name = str(i_name).split('\\x00')[-1]
                nodeset_names.append(i_name)
            else:
                nodeset_names.append('')
                
            # move pointer to nodelist
            a = search_block(bf, TAGS, 'NODESET_LIST', verbose=verbose)
            nodes = nparray(read_bytes(bf, nb=a, format="I"*n_nodes), dtype="int")
            nodeset_nodes.append(nodes)
    
        # transform data to nparrays
        nodeset_ids = nparray(nodeset_ids, dtype="int32")
        nodeset_names = nparray(nodeset_names)
        nodeset_nodes = nparray(nodeset_nodes, dtype="object")
        
        nodesets_by_name = {}
        if len(nodeset_nodes) == len(nodeset_names):
            for nodeset, name in zip(nodeset_nodes,nodeset_names):
                nodesets_by_name[name] = nodeset
        
        nodesets_by_id = {}
        if len(nodeset_nodes) == len(nodeset_ids):
            for nodeset, key in zip(nodeset_nodes,nodeset_ids):
                nodesets_by_id[key] = nodeset
            
        nodeset_data = {
            "nodesets": nodeset_nodes,
            "by_name": nodesets_by_name,
            "by_id": nodesets_by_id,   
        }
    return nodeset_data


