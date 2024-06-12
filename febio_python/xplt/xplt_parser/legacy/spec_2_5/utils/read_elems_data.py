def read_elems_data(bf, TAGS, item_names, item_types, n_node_data, item_def_doms, filesize, item_data=[], verbose=0):
  from ...common.utils import search_block, check_block, read_bytes, num_el_nodes, console_log
  from numpy import zeros as npzeros
  from numpy import array as nparray
  from collections import deque
  import numpy as np
  
  console_log("-----reading elems------", 2, verbose)
  
  a = search_block(bf, TAGS, 'ELEMENT_DATA', verbose=verbose)
  while check_block(bf, TAGS, 'STATE_VARIABLE'):

    a = search_block(bf, TAGS, 'STATE_VARIABLE', verbose=verbose)
    a = search_block(bf, TAGS, 'STATE_VAR_ID', verbose=verbose)

    var_id = read_bytes(bf) + n_node_data
    console_log('variable_name: {}'.format(item_names[var_id - 1]), 2, verbose)

    a = search_block(bf, TAGS, 'STATE_VAR_DATA', verbose=verbose)
    a_end = bf.tell() + a
    if item_types[var_id - 1] == 0:  # FLOAT
      data_dim = 1
    elif item_types[var_id - 1] == 1:  # VEC3F
      data_dim = 3
    # MAT3FS (6 elements due to symmetry)
    elif item_types[var_id - 1] == 2:
      data_dim = 6
    else:
      print('unknwon data dimension!')
      return -1

    def_doms = deque()
    dom_data = deque()
    while(bf.tell() < a_end):
      dom_num = read_bytes(bf)
      data_size = read_bytes(bf)
      n_data = int(data_size / data_dim / 4.0)
      def_doms.append(dom_num)
      console_log('number of element data for domain %s = %d' % (dom_num, n_data), 2, verbose)

      if n_data > 0:
        elem_data = nparray(read_bytes(bf, nb=data_size, format="f"*n_data*data_dim), 
                        dtype=float).reshape((n_data, data_dim))
        
        dom_data.append(elem_data)

    item_def_doms.append(def_doms)
    item_data.append(np.concatenate(dom_data))

    if bf.tell() >= filesize:
      break

  return (item_def_doms, elem_data, item_data)