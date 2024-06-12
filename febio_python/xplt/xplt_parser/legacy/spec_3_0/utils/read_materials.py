from ...common.utils import search_block, check_block, read_bytes, console_log
from enum import IntEnum
from collections import deque

def read_materials(bf, TAGS: IntEnum, verbose:int=0):
    search_block(bf, TAGS, 'MATERIALS', verbose=verbose)
    
    mat_names = deque()
    mat_ids = deque()

    while check_block(bf, TAGS, 'MATERIAL', verbose=verbose):
        search_block(bf, TAGS, 'MATERIAL', verbose=verbose)

        search_block(bf, TAGS, 'MAT_ID', verbose=verbose)
        mat_ids.append(int(read_bytes(bf)))

        search_block(bf, TAGS, 'MAT_NAME', verbose=verbose)
        name_data = bf.read(64)
        try:
            mat_name = name_data.decode('ascii').split('\x00')[0]
        except UnicodeDecodeError:
            mat_name = name_data.decode('ascii', errors='replace').split('\x00')[0]
        mat_names.append(mat_name)

    console_log("---read_materials", 2, verbose)
    console_log("->materials items: [mat_names, mat_ids]", 2, verbose)
    console_log([mat_names, mat_ids], 2, verbose)

    return (mat_names, mat_ids)
