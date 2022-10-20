from ...common.utils import search_block, check_block, read_bytes, num_el_nodes 
from numpy import zeros as npzeros
from collections import deque
from numpy import array as nparray

def read_nodeset_section(bf, TAGS, verbose=0):
    nodeset_ids   = deque()
    nodeset_names = deque()
    nodeset_nodes = deque()

    a0 = search_block(bf, TAGS, 'NODESET_SECTION', verbose=verbose)
    if a0 > 0:
        
        while check_block(bf, TAGS, 'NODESET'):
            search_block(bf, TAGS, 'NODESET', verbose=verbose)
            
            search_block(bf, TAGS, 'NODESET_HDR', verbose=verbose)
            a = search_block(bf, TAGS, 'NODESET_ID', verbose=verbose)
            nodeset_id = read_bytes(bf, nb=a)
            nodeset_ids.append(nodeset_id)
            # print("nodeset_id:", nodeset_id)
            
            search_block(bf, TAGS, 'NODESET_NN', verbose=verbose)
            NN = read_bytes(bf, nb=a)
            # print("NN:", NN)

            a = search_block(bf, TAGS, 'NODESET_NAME', verbose=verbose)
            i_name = bf.read(a) # from docs, here should be CHAR64, but its DWORD from above
            try:
                i_name = i_name.decode('ascii').split('\x00')[-1]
            except UnicodeDecodeError:
                i_name = str(i_name).split('\\x00')[-1]
            nodeset_names.append(i_name)
            # print("nodeset_name", i_name)
            
            a = search_block(bf, TAGS, 'NODESET_LIST', verbose=verbose)
            # nodes = nparray(read_bytes(bf, nb=a, format="I"*NN), dtype=int) + 1
            
            nodes = npzeros(NN, dtype=int)
            for i in range(NN):
                nodes[i] = read_bytes(bf) + 1
            nodeset_nodes.append(nodes)
    
    return (nodeset_ids, nodeset_names, nodeset_nodes)


