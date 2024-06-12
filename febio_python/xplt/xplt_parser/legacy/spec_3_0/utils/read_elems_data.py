from ...common.utils import search_block, check_block, read_bytes, console_log
from numpy import array as nparray, concatenate
from enum import IntEnum

def read_elems_data(bf, TAGS: IntEnum, item_names, item_types, n_node_data, item_def_doms, filesize, item_data=[], verbose=0):
    console_log("-----reading elems------", 2, verbose)
    search_block(bf, TAGS, 'ELEMENT_DATA', verbose=verbose)
  
    while check_block(bf, TAGS, 'STATE_VARIABLE'):
        search_block(bf, TAGS, 'STATE_VARIABLE', verbose=verbose)
        search_block(bf, TAGS, 'STATE_VAR_ID', verbose=verbose)
        var_id = read_bytes(bf) + n_node_data
        console_log(f'variable_name: {item_names[var_id - 1]}', 2, verbose)
        
        search_block(bf, TAGS, 'STATE_VAR_DATA', verbose=verbose)
        data_end = bf.tell() + read_bytes(bf, format="I")

        if item_types[var_id - 1] == 0:
            data_dim = 1
        elif item_types[var_id - 1] == 1:
            data_dim = 3
        elif item_types[var_id - 1] == 2:
            data_dim = 6
        else:
            console_log('Unknown data dimension!', 2, verbose)
            return -1

        def_doms = []
        dom_data = []

        while bf.tell() < data_end:
            dom_num = read_bytes(bf, format="I")
            data_size = read_bytes(bf, format="I")
            n_data = data_size // (data_dim * 4)
            console_log(f'number of element data for domain {dom_num} = {n_data}', 2, verbose)
            elem_data = nparray(read_bytes(bf, nb=data_size, format="f" * n_data * data_dim), dtype=float).reshape((n_data, data_dim))
            def_doms.append(dom_num)
            dom_data.append(elem_data)

        item_def_doms.append(def_doms)
        item_data.append(concatenate(dom_data) if dom_data else nparray([]))

        if bf.tell() >= filesize:
            break

    return (item_def_doms, item_data)
