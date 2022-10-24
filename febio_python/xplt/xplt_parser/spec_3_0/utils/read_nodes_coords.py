def read_nodes_coords(bf, TAGS, verbose=0):
  from ...common.utils import search_block, check_block, read_bytes, console_log
  # from numpy import zeros as npzeros
  from numpy import array as nparray
  import struct

  search_block(bf, TAGS, 'NODE_SECTION', verbose=verbose)
  
  # move pointer to nodeset header
  search_block(bf, TAGS, 'NODE_HEADER', verbose=verbose)
  
  # --> explore nodes header to get shape of nodes section
  a = search_block(bf, TAGS, 'NODE_N_NODES', verbose=verbose)
  n_nodes = read_bytes(bf, a)
  a = search_block(bf, TAGS, 'NODE_DIM', verbose=verbose)
  n_dims = read_bytes(bf, a)
  nodes_shape = (n_nodes, n_dims)
  console_log("n_nodes: {} | n_dim: {} -> shape {}".format(n_nodes, n_dims, nodes_shape), 3, verbose=verbose)  
  
  # extract nodes 
  a = search_block(bf, TAGS, 'NODE_COORDS_3_0', verbose=verbose)
  # set format to extract data. 
  # In the 3_0 spec, index data is saved. leading to an "additional" dimension
  # [index(int), coord(float)*n_dim ....]
  f_format = "i" + "f"*n_dims
  node_data = nparray(read_bytes(bf, nb=a, format=f_format*n_nodes), 
                        dtype=float).reshape((n_nodes, n_dims+1))
  # now get all node ids and sort coordinates. 
  # (We do not want to return coords as 4 dims, each row will have the node id
  node_ids = node_data[:, 0].astype('int32') # get all node IDs
  node_coords = node_data[:, 1:][node_ids]
  
      
  console_log("---read_nodes_coords", 2, verbose)
  console_log("->number of node coords: {}".format(n_nodes), 2, verbose)
  console_log("->number of node dims: {}".format(n_dims), 2, verbose)
  console_log("->extracted coords shape: {}".format(node_coords.shape), 2, verbose)
  
  console_log(node_coords, 3, verbose)

  return node_coords