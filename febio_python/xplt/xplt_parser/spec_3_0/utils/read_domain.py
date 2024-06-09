from ...common.utils import search_block, check_block, read_bytes, num_el_nodes, console_log
from numpy import array
from enum import IntEnum
from collections import deque

def read_domain(bf, TAGS:IntEnum, ELEM_TYPES:IntEnum, NODES_PER_ELEM:IntEnum, verbose:int=0):
    console_log("---read_domain", 2, verbose)
    
    domains = deque()
    
    # move pointer to domain section
    search_block(bf, TAGS, 'DOMAIN_SECTION', verbose=verbose)
    
    while check_block(bf, TAGS, 'DOMAIN'):
        console_log("--reading domain content", 3, verbose)
        domain_info = {}
        
        # move pointer to domain content and header
        search_block(bf, TAGS, 'DOMAIN', verbose=verbose)
        search_block(bf, TAGS, 'DOMAIN_HEADER', verbose=verbose)
        
        # Read element type, part ID, and number of elements
        search_block(bf, TAGS, 'DOM_ELEM_TYPE', verbose=verbose)
        domain_info['elem_type'] = int(read_bytes(bf))
        search_block(bf, TAGS, 'DOM_PART_ID', verbose=verbose)
        domain_info['part_id'] = int(read_bytes(bf))
        search_block(bf, TAGS, 'DOM_N_ELEMS', verbose=verbose)
        domain_info['n_elems'] = int(read_bytes(bf))
        
        elem_type_name = ELEM_TYPES(domain_info['elem_type']).name
        n_nodes_per_element = NODES_PER_ELEM[elem_type_name].value
        
        # Prepare for reading elements list
        search_block(bf, TAGS, 'DOM_ELEM_LIST', verbose=verbose)
        elements = []
        for _ in range(domain_info['n_elems']):
            search_block(bf, TAGS, 'ELEMENT', verbose=verbose)
            element = array(read_bytes(bf, nb=n_nodes_per_element+1, format="I"), dtype=int)
            elements.append(element)
        
        domain_info['elements'] = array(elements, dtype=int)
        
        domains.append(domain_info)
        console_log("--elements shape: {}".format(domain_info['elements'].shape), 3, verbose)
    
    dom_data = {
        "n_doms": len(domains),
        "types": array([d['elem_type'] for d in domains], dtype=int),
        "ids": array([d['part_id'] for d in domains], dtype=int),
        "n_elems": array([d['n_elems'] for d in domains], dtype=int),
        "elems": array([d['elements'] for d in domains], dtype=object),
    }
    
    console_log("->domain items: [types, ids, n_elems, elems]", 2, verbose)
    console_log(dom_data, 3, verbose)
    
    return dom_data
