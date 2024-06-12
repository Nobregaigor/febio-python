def read_domain(bf, TAGS, ELEM_TYPES, NODES_PER_ELEM, verbose=0):
  from ...common.utils import search_block, check_block, read_bytes, num_el_nodes, console_log
  from numpy import zeros as npzeros
  from numpy import array as nparray
  from collections import deque

  dom_elem_types = deque()
  dom_mat_ids = deque()
  dom_n_elems = deque()  # number of elements for each domain
  dom_elements = deque()  # elements for each domain

  search_block(bf, TAGS, 'DOMAIN_SECTION', verbose=verbose)

  while check_block(bf, TAGS, 'DOMAIN'):

    search_block(bf, TAGS, 'DOMAIN', verbose=verbose)
    search_block(bf, TAGS, 'DOMAIN_HEADER', verbose=verbose)
    search_block(bf, TAGS, 'DOM_ELEM_TYPE', verbose=verbose)
    elem_type = int(read_bytes(bf))
    dom_elem_types.append(elem_type)
    elem_type_name = ELEM_TYPES(elem_type).name

    search_block(bf, TAGS, 'DOM_MAT_ID', verbose=verbose)
    dom_mat_ids.append(int(read_bytes(bf)))

    search_block(bf, TAGS, 'DOM_N_ELEMS', verbose=verbose)
    n_elems = int(read_bytes(bf))
    dom_n_elems.append(n_elems) 

    # move pointer to elements list
    a = search_block(bf, TAGS, 'DOM_ELEM_LIST', verbose=verbose)   
    n_nodes_per_element = NODES_PER_ELEM[elem_type_name].value
    elements = deque()
    for _ in range(n_elems):
      # move pointer to next element
      a = search_block(bf, TAGS, 'ELEMENT', verbose=verbose)
      # real element nodes
      element = nparray(read_bytes(bf, nb=a, format="I"*(n_nodes_per_element+1)), dtype=int)
      elements.append(element)

    elements = nparray(elements, dtype=int)
    console_log("--elements shape: {}".format(elements.shape),3,verbose)
    
    dom_elements.append(elements)

  console_log("---read_domain",2,verbose)
  console_log("->domain items: [dom_elem_types, dom_mat_ids, dom_n_elems, dom_elements]", 2, verbose)
  console_log([dom_elem_types, dom_mat_ids, dom_n_elems, dom_elements], 3, verbose)
    
  dom_data = {
    "n_doms": len(dom_elem_types),
    "types": nparray(dom_elem_types, dtype="int32"),
    "ids": nparray(dom_mat_ids, dtype="int32"),
    "n_elems": nparray(dom_n_elems, dtype="int32"),
    "elems": nparray(dom_elements, dtype="object"), 
  }
  
  return dom_data