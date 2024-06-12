from ...common.utils import search_block, check_block, read_bytes, console_log
from numpy import array as nparray
import numpy as np
from enum import IntEnum
    
def read_nodes_coords(bf, TAGS:IntEnum, verbose=0):
  search_block(bf, TAGS, 'NODE_SECTION', verbose=verbose)
  search_block(bf, TAGS, 'NODE_HEADER', verbose=verbose)
  
  n_nodes = read_bytes(bf, search_block(bf, TAGS, 'NODE_N_NODES', verbose=verbose))
  n_dims = read_bytes(bf, search_block(bf, TAGS, 'NODE_DIM', verbose=verbose))
  # read next 
  
  name = search_block(bf, TAGS, 'NODE_NAME', verbose=verbose)
  print(f"NODES NAME: {name}")
  
  nodes_shape = (n_nodes, n_dims)
  console_log(f"n_nodes: {n_nodes} | n_dim: {n_dims} -> shape {nodes_shape}", 3, verbose=verbose)  

  # Prepare format string and read node data
  f_format = "i" + "f" * n_dims
  node_data = nparray(read_bytes(bf, search_block(bf, TAGS, 'NODE_COORDS_3_0', verbose=verbose), format=f_format * n_nodes), dtype=float).reshape((n_nodes, n_dims + 1))

  # Extract node IDs and coordinates
  node_ids = node_data[:, 0].astype(int)  # Assuming node IDs are the first column
  node_coords = node_data[:, 1:]  # Assuming subsequent columns are coordinates

  console_log("---read_nodes_coords", 2, verbose)
  console_log(f"->number of node coords: {n_nodes}", 2, verbose)
  console_log(f"->number of node dims: {n_dims}", 2, verbose)
  console_log(f"->extracted coords shape: {node_coords.shape}", 2, verbose)
  console_log(node_coords, 3, verbose)

  return node_coords
