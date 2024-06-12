def read_surface_section(bf, TAGS, verbose=0, filesize=0):
  from ...common.utils import search_block, check_block, read_bytes, num_el_nodes, console_log
  from numpy import zeros as npzeros
  from numpy import array as nparray
  from collections import deque

  surface_ids = deque()          # id for each surface
  surface_facets_count = deque()  # number of facets per surface
  surface_names = deque()        # name for each surface
  face_ids = deque()             # facet id for each set of nodes within a surface
  surface_facets = deque()        # lists of each "facet node list" wihtin a surface
  surface_nodes = deque()        # sets of nodes within a surface
  faces = deque()
  
  surface_data = {
      "facets": nparray([]),
      "by_id": nparray([]),
      "by_name": nparray([]),    
    }
  
  if search_block(bf, TAGS, 'SURFACE_SECTION', verbose=verbose) > 0:
    while check_block(bf, TAGS, 'SURFACE'):
      search_block(bf, TAGS, 'SURFACE', verbose=verbose)
      search_block(bf, TAGS, 'SURFACE_HEADER', verbose=verbose)
      search_block(bf, TAGS, 'SURFACE_ID', verbose=verbose)

      surface_ids.append(read_bytes(bf))

      # number of facets for given surface
      search_block(bf, TAGS, 'SURFACE_N_FACETS', verbose=verbose)
      surface_facets_count.append(read_bytes(bf))

      a = search_block(bf, TAGS, 'SURFACE_NAME', verbose=verbose)
      i_name = bf.read(a) # from docs, here should be CHAR64, but its DWORD from above
      try:
        surface_names.append(i_name.decode('ascii').split('\x00')[-1])
      except UnicodeDecodeError:
        surface_names.append(str(i_name).split('\\x00')[-1])

      # Skip facet list if not existent
      if (check_block(bf, TAGS, 'FACET_LIST') == 0):
        continue
      else:
        search_block(bf, TAGS, 'FACET_LIST', verbose=verbose)

      i_surface_nodes = set()
      i_surface_facets = deque()
      
      while check_block(bf, TAGS, 'FACET'):
        a = search_block(bf, TAGS, 'FACET', verbose=verbose)
        cur_cur = bf.tell()
        face_ids.append(read_bytes(bf))
        nodes_per_facet = read_bytes(bf)
        
        face_nodes = npzeros(nodes_per_facet, dtype=int)
        for j in range(nodes_per_facet):
          j_node_id = read_bytes(bf) + 1
          face_nodes[j] = j_node_id
          i_surface_nodes.add(j_node_id)

        faces.append(face_nodes)
        i_surface_facets.append(face_nodes)             
        # skip junk
        bf.seek(cur_cur + a, 0)

      surface_facets.append(nparray(i_surface_facets, dtype="object"))
      surface_nodes.append(i_surface_nodes)
      # print("surface_nodes:", surface_nodes)

    console_log("---read_materials", 2, verbose)
    console_log("->surface items: [surface_ids, surface_facets, surface_names, surface_nodes, faces, face_ids]", 2, verbose)
    console_log([surface_ids, surface_facets, surface_names, surface_nodes, faces, face_ids], 3, verbose)

    # transform data to nparrays
    surface_ids = nparray(surface_ids, dtype="int32")
    surface_names = nparray(surface_names)
    surface_facets = nparray(surface_facets, dtype="object")
    
    facets_by_name = {}
    if len(surface_facets) == len(surface_names):
      for facet, name in zip(surface_facets,surface_names):
        facets_by_name[name] = facet
    
    facets_by_id = {}
    if len(surface_facets) == len(surface_ids):
      for facet, key in zip(surface_facets,surface_ids):
        facets_by_id[key] = facet
        
    surface_data = {
      "facets": surface_facets,
      "by_id": facets_by_id,
      "by_name": facets_by_name,    
    }
  
  return surface_data