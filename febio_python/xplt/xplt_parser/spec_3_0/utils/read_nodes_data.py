from ...common.utils import search_block, check_block, read_bytes, num_el_nodes, console_log 
from numpy import zeros as npzeros
from numpy import array as nparray
from collections import deque
from enum import IntEnum

def read_nodes_data(bf, TAGS: IntEnum, item_names, item_types, verbose=0):
  
  console_log("-----reading nodes------", 2, verbose)

  a = search_block(bf, TAGS, 'NODE_DATA', verbose=verbose)
  n_node_data = 0
  item_def_doms = deque()
  item_data = deque()
  while check_block(bf, TAGS, 'STATE_VARIABLE'):
    n_node_data += 1
    
    a = search_block(bf, TAGS, 'STATE_VARIABLE', verbose=verbose)
    a = search_block(bf, TAGS, 'STATE_VAR_ID', verbose=verbose)
    var_id = read_bytes(bf)

    console_log('variable_name: {}'.format(item_names[var_id - 1]), 2, verbose)

    a = search_block(bf, TAGS, 'STATE_VAR_DATA', verbose=verbose)

    a_end = bf.tell() + a
    if item_types[var_id - 1] == 0:  # FLOAT
      data_dim = 1
    elif item_types[var_id - 1] == 1:  # VEC3F
      data_dim = 3
    elif item_types[var_id - 1] == 2: # MAT3FS (6 elements due to symmetry)
      data_dim = 6
    else:
      print('unknwon data dimension!')
      return -1

    # assumption: node data is defined for all the ndoes
    def_doms = deque()
    while(bf.tell() < a_end):
      dom_num = read_bytes(bf)
      data_size = read_bytes(bf)
      n_data = int(data_size / data_dim / 4.0)
      def_doms.append(dom_num)
      if verbose == 1:
        console_log('number of node data for domain %s = %d' % (dom_num, n_data), 2, verbose)
      if n_data > 0:
        node_data = nparray(read_bytes(bf, nb=data_size, format="f"*n_data*data_dim), 
                        dtype=float).reshape((n_data, data_dim))
        
        # node_data = npzeros([n_data, data_dim])
        # for i in range(0, n_data):
        #   for j in range(0, data_dim):
        #     node_data[i, j] = read_bytes(bf, format="f")
      else:
        node_data = deque()

    item_def_doms.append(def_doms)
    item_data.append(node_data)

  return (n_node_data, item_def_doms, item_data)