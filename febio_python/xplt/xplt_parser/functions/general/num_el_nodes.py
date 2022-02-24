# def num_el_nodes(dom_elem_type, ELEMS_TYPE, NODES_PER_ELEM):
#     for key, value in ELEMS_TYPE.items():
#         if dom_elem_type == value:
#             return NODES_PER_ELEM[key]
#     raise(ValueError("Could not find elem key: {}".format(dom_elem_type)))

def num_el_nodes(dom_elem_type, ELEMS_TYPE, NODES_PER_ELEM):
    # check if ELEMS_TYPE contains given elem number
    if str(dom_elem_type) not in ELEMS_TYPE:
        raise(ValueError("Could not find elem key: {}, in ELEMS_TYPE".format(dom_elem_type)))
    
    elem_type = ELEMS_TYPE[str(dom_elem_type)]
    # check if NODES_PER_ELEM contains given elem type
    if elem_type not in NODES_PER_ELEM:
        raise(ValueError("Could not find number of nodes of: {}, in NODES_PER_ELEM".format(dom_elem_type)))
    
    return NODES_PER_ELEM[elem_type]