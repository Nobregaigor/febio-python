from ...common.utils import search_block, check_block, read_bytes, num_el_nodes, console_log
from numpy import zeros as npzeros
from numpy import array as nparray
from collections import deque

def read_domain(bf, TAGS, ELEMS_TYPES, NODES_PER_ELEM, verbose=0):

  dom_elem_types = deque()
  dom_mat_ids = deque()
  dom_n_elems = deque()  # number of elements for each domain
  dom_elements = deque()  # elements for each domain

  search_block(bf, TAGS, 'DOMAIN_SECTION', verbose=verbose)

  while check_block(bf, TAGS, 'DOMAIN'):

    search_block(bf, TAGS, 'DOMAIN', verbose=verbose)
    search_block(bf, TAGS, 'DOMAIN_HDR', verbose=verbose)
    search_block(bf, TAGS, 'DOM_ELEM_TYPE', verbose=verbose)
    dom_elem_type = int(read_bytes(bf))
    dom_elem_types.append(dom_elem_type)

    search_block(bf, TAGS, 'DOM_MAT_ID', verbose=verbose)
    dom_mat_ids.append(int(read_bytes(bf)))

    search_block(bf, TAGS, 'DOM_ELEMS', verbose=verbose)
    dom_n_elems.append(int(read_bytes(bf)))

    search_block(bf, TAGS, 'DOM_ELEM_LIST', verbose=verbose)

    ne = num_el_nodes(dom_elem_type, ELEMS_TYPES, NODES_PER_ELEM)

    elements = deque()
    while check_block(bf, TAGS, 'ELEMENT'):
      a = search_block(bf, TAGS, 'ELEMENT', print_tag=0, verbose=verbose)
      
      element = nparray(read_bytes(bf, nb=a, format="I"*(ne+1)), dtype=int)
      
      # element = npzeros(ne, dtype=int)
      # for j in range(ne + 1):
      #   if j == 0:
      #     read_bytes(bf)
      #   else:
      #     element[j -1] = read_bytes(bf)
      
      elements.append(element)

    dom_elements.append(nparray(elements, dtype=int))

  console_log("---read_domain",2,verbose)
  console_log("->domain items: [dom_elem_types, dom_mat_ids, dom_n_elems, dom_elements]", 2, verbose)
  console_log([dom_elem_types, dom_mat_ids, dom_n_elems, dom_elements], 3, verbose)
    
  return (dom_elem_types, dom_mat_ids, dom_n_elems, dom_elements)