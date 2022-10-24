def read_state(bf, TAGS, states_dict, decompress=False, verbose=0):
    from ...common.utils import search_block, check_block, read_bytes, num_el_nodes, console_log, decompress_state, get_file_size
    from numpy import array as nparray       
    from collections import deque
  
    # decompress if needed.
    if decompress:
        bf = decompress_state(bf, verbose=verbose)
    else:
        console_log("File is not compressed.", 3, verbose=verbose)
    
    # need to re-read filesize to account for decompression
    filesize = get_file_size(bf) 
    
    def process_state_var(storage):
        from ...common.database import DATA_TYPES, DATA_LENGTHS
        
        while check_block(bf, TAGS, 'STATE_VARIABLE', filesize=filesize):
            # move pointer to state variable
            search_block(bf, TAGS, 'STATE_VARIABLE', verbose=verbose)
            # get index of state item
            a = search_block(bf, TAGS, 'STATE_VAR_ID', verbose=verbose)
            var_idx = int(read_bytes(bf, nb=a)) - 1
            
            # determine data dims based on item type
            var_type = DATA_TYPES(states_dict["types"][var_idx]).name
            console_log("--var_type: {}".format(var_type), 3, verbose=verbose)
            var_dim = DATA_LENGTHS[var_type].value
            console_log("--var_dim: {}".format(var_dim), 3, verbose=verbose)
            
            # move pointer to actual data
            a = search_block(bf, TAGS, 'STATE_VAR_DATA', verbose=verbose)
            n_data = int(a / 4.0)
            var_data = read_bytes(bf, nb=a, format='f'*n_data)
            var_data = nparray(var_data, dtype="float32")
            storage.append(var_data)
           
    
    state_time   = deque()
    state_node_data = deque()
    state_elem_data = deque()
    state_surf_data = deque()
    
    states_ref = deque()

    while search_block(bf, TAGS, 'STATE', verbose=verbose) > 0:
 
        # So we need to check if there are statess in our file
        if check_block(bf, TAGS, 'STATE_HEADER'):
        
            while check_block(bf, TAGS, 'STATE_HEADER', filesize=filesize):
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
                
                # extract data
                if check_block(bf, TAGS, 'NODE_DATA', filesize=filesize):
                    # move pointer to next nodal data
                    search_block(bf, TAGS, 'NODE_DATA', verbose=verbose)
                    process_state_var(state_node_data)
                    states_ref.append(0)
                if check_block(bf, TAGS, 'ELEMENT_DATA', filesize=filesize):
                    # move pointer to next element data
                    search_block(bf, TAGS, 'ELEMENT_DATA', verbose=verbose)
                    process_state_var(state_elem_data)
                    states_ref.append(1)
                if check_block(bf, TAGS, 'SURFACE_DATA', filesize=filesize):
                    # move pointer to next surface data
                    search_block(bf, TAGS, 'SURFACE_DATA', verbose=verbose)
                    process_state_var(state_surf_data)
                    states_ref.append(2)
                
        else:
            console_log("No state was found.", 1, verbose=verbose)
    
        if bf.tell() == filesize: # reach end of states (EOF)
            break
        
    # convert data to numpy arrays
    state_time = nparray(state_time, dtype="float32")
    state_node_data = nparray(state_node_data, dtype="float32")
    try:
        state_elem_data = nparray(state_elem_data, dtype="float32")
    except:
        state_elem_data = nparray(state_elem_data, dtype="object")
    state_surf_data = nparray(state_surf_data, dtype="float32")
    
    # combine data names and values
    data_by_name = {}
    for i, (r, key) in enumerate(zip(states_ref, states_dict["names"])):
        if r == 0:
            data_by_name[key] = state_node_data[i]
        elif r == 1:
            data_by_name[key] = state_elem_data[i]
        elif r == 2:
            data_by_name[key] = state_surf_data[i]
    
    state_data = {
            "n": len(state_time),
            "time": state_time,
            "data": data_by_name
        }
        
    return state_data