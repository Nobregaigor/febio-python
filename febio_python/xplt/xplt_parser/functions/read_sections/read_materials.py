from .. import search_block, check_block, read_bytes, console_log
# from numpy import zeros as npzeros
from collections import deque

def read_materials(bf, TAGS, verbose=0):
  search_block(bf, TAGS, 'MATERIALS', verbose=verbose)
  mat_names = deque()
  mat_ids = deque()

  while check_block(bf, TAGS, 'MATERIAL', verbose=verbose):
    search_block(bf, TAGS, 'MATERIAL', verbose=verbose)

    search_block(bf, TAGS, 'MAT_ID')
    mat_ids.append(int(read_bytes(bf)))

    search_block(bf, TAGS, 'MAT_NAME', verbose=verbose)
    mat_names.append(bf.read(64).decode('ascii').split('\x00')[0])
  
  console_log("---read_materials", 2, verbose)
  console_log("->materials items: [mat_names, mat_ids]", 2, verbose)
  console_log([mat_names, mat_ids], 2, verbose )

  return (mat_names, mat_ids)