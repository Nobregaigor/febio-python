from .. import search_block, check_block, read_bytes, console_log
# from numpy import zeros as npzeros
from numpy import array as nparray
import struct

def read_nodes_coords(bf, TAGS, verbose=0):
  search_block(bf, TAGS, 'GEOMETRY', verbose=verbose)
  search_block(bf, TAGS, 'NODE_SECTION', verbose=verbose)
  a = search_block(bf, TAGS, 'NODE_COORDS', verbose=verbose)

  n_nodes = int(a / 3 / 4)
  node_coords = nparray(read_bytes(bf, nb=a, format="f"*n_nodes*3), 
                        dtype=float).reshape((n_nodes, 3))

  # node_coords = npzeros([n_nodes, 3])
  # for i in range(n_nodes):
  #   for j in range(0, 3):
  #     node_coords[i, j] = read_bytes(bf, format="f")
      
  console_log("---read_nodes_coords", 2, verbose)
  console_log("->number of node coords: {}".format(n_nodes), 2, verbose)
  console_log("->extracted coords shape: {}".format(node_coords.shape), 2, verbose)
  
  console_log(node_coords, 3, verbose)

  return node_coords