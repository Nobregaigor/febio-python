def read_state(bf, TAGS, states_dict, decompress=False, verbose=0):
    from ...common.utils import search_block, check_block, read_bytes, num_el_nodes, console_log, decompress_state, get_file_size
    from numpy import array as nparray
    from numpy import vstack, hstack
      
    from collections import deque
  
    # decompress if needed.
    if decompress:
        bf = decompress_state(bf, verbose=verbose)
    else:
        console_log("File is not compressed.", 3, verbose=verbose)
    
    # need to re-read filesize to account for decompression
    filesize = get_file_size(bf) 
    
    
    def process_state_var(storage, offset=0):
        from ...common.database import DATA_TYPES, DATA_LENGTHS
        
        new_data = deque()
        while check_block(bf, TAGS, 'STATE_VARIABLE', filesize=filesize):
            # move pointer to state variable
            search_block(bf, TAGS, 'STATE_VARIABLE', verbose=verbose)
            # get index of state item
            a = search_block(bf, TAGS, 'STATE_VAR_ID', verbose=verbose)
            var_idx = int(read_bytes(bf, nb=a)) -1 + offset 
            
            # determine data dims based on item type
            var_type = DATA_TYPES(states_dict["types"][var_idx]).name
            console_log("--var_type: {}".format(var_type), 3, verbose=verbose)
            var_dim = DATA_LENGTHS[var_type].value
            console_log("--var_dim: {}".format(var_dim), 3, verbose=verbose)
            
            # move pointer to actual data
            a = search_block(bf, TAGS, 'STATE_VAR_DATA', verbose=verbose)
            
            # read domain id -> not used in this package (for now)
            domain_num = int(read_bytes(bf))
            
            # read data size 
            data_size = read_bytes(bf)
            
            # read data
            n_data = data_size // var_dim // 4
            var_data = read_bytes(bf, nb=data_size, format='f'*(n_data*var_dim))
            var_data = nparray(var_data, dtype="float32")
            var_data = var_data.reshape((n_data, var_dim))
            new_data.append(var_data)
            
        offset += len(new_data)
        storage.append(new_data)
        
        return offset

    def set_data_as_nparray_or_deque(data):
        if len(data) > 0 :
            if len(data[0]) == 1:
                data = nparray(data, dtype="float32")
                data = data.swapaxes(0, 1)
            else:
                n_subs = len(data[0])
                step_d = deque([deque() for _ in range(n_subs)])
                for values in data:
                    for i, v in enumerate(values):
                        step_d[i].append(v)
                for i, v in enumerate(step_d):
                    step_d[i] = nparray(v, dtype="float32")
                data = step_d
        return data
    
    state_time   = deque()
    state_node_data = deque()
    state_elem_data = deque()
    state_surf_data = deque()
    
    states_ref = deque()
    
    # set default value if error is encountered
    empty_state = {
            "n": 0,
            "time": [],
            "data": []
        }
    
    mark_refs = True
    while search_block(bf, TAGS, 'STATE', verbose=verbose) > 0:
 
        # So we need to check if there are statess in our file
        if check_block(bf, TAGS, 'STATE_HEADER'):
                    
            # while check_block(bf, TAGS, 'STATE_HEADER', filesize=filesize):
            # move pointer to next STATE
            # explore STATE header
            search_block(bf, TAGS, 'STATE_HEADER', verbose=verbose)
            # get state time
            a = search_block(bf, TAGS, 'STATE_HEADER_TIME', verbose=verbose)
            time = read_bytes(bf, nb=a, format="f")
            state_time.append(time)
            console_log("state_time: {}".format(time), 3, verbose=verbose)
            # print("-------------------------------------------> TIME:", time)
            
            # move pointer to next STATE_DATA
            search_block(bf, TAGS, 'STATE_DATA', verbose=verbose)
            # print("STATE DATA", a)
            
            # extract data
            offset = 0
            if check_block(bf, TAGS, 'NODE_DATA', filesize=filesize):
                # move pointer to next nodal data
                # print("found node data")
                search_block(bf, TAGS, 'NODE_DATA', verbose=verbose)
                n_nodes_data = process_state_var(state_node_data)
                if mark_refs:
                    states_ref.extend([0 for i in range(n_nodes_data)])
            if check_block(bf, TAGS, 'ELEMENT_DATA', filesize=filesize):
                # move pointer to next element data
                search_block(bf, TAGS, 'ELEMENT_DATA', verbose=verbose)
                # print("found element data")
                n_elems_data = process_state_var(state_elem_data, offset+n_nodes_data)
                if mark_refs:
                    states_ref.extend([1 for i in range(n_elems_data)])
            if check_block(bf, TAGS, 'SURFACE_DATA', filesize=filesize):
                # move pointer to next surface data
                # print("found surface data")
                search_block(bf, TAGS, 'SURFACE_DATA', verbose=verbose)
                n_surf_data = process_state_var(state_surf_data, offset+n_nodes_data+n_elems_data)
                if mark_refs:
                    states_ref.extend([2 for i in range(n_surf_data)])
            mark_refs = False
        else:
            console_log("No state was found.", 1, verbose=verbose)
            return empty_state
    
        if bf.tell() == filesize: # reach end of states (EOF)
            break
          
    
    # convert data to numpy arrays
    state_time = nparray(state_time, dtype="float32")    
    state_node_data = set_data_as_nparray_or_deque(state_node_data)
    state_elem_data = set_data_as_nparray_or_deque(state_elem_data)
    state_surf_data = set_data_as_nparray_or_deque(state_surf_data)

    
    # combine data names and values
    data_by_name = {}
    nn, ne, ns = 0, 0, 0
    for (r, key) in zip(states_ref, states_dict["names"]):
        print(key, r, nn, ne, ns)
        if r == 0:
            data_by_name[key] = state_node_data[nn]
            nn+=1
        elif r == 1:
            data_by_name[key] = state_elem_data[ne]
            ne+=1
        elif r == 2:
            data_by_name[key] = state_surf_data[ns]
            ns+=1
    
    state_data = {
            "n": len(state_time),
            "time": state_time,
            "data": data_by_name
        }
            
    return state_data