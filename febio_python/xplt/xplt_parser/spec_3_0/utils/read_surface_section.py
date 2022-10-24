def read_surface_section(bf, TAGS, verbose=0, filesize=0):
  from ...common.utils import search_block, check_block, read_bytes, num_el_nodes, console_log
  from numpy import zeros as npzeros
  from numpy import array as nparray
  from collections import deque

  
  console_log("---read_surface_section", 2, verbose)
  
  surface_ids = deque()          # id for each surface
  surface_names = deque()        # name for each surface
  surface_facets = deque()        # lists of each "facet node list" wihtin a surface
  
  # return empty values if no surface is found (for consistency)
  surface_data = {
      "facets": nparray([]),
      "by_id": nparray([]),
      "by_name": nparray([]),    
    }
  
  # surfaces are optional. 
  # So we need to check if there are surfaces in our file
  if search_block(bf, TAGS, 'SURFACE_SECTION', verbose=verbose) > 0:
    
    while check_block(bf, TAGS, 'SURFACE'):
      # move poiter to next surface
      search_block(bf, TAGS, 'SURFACE', verbose=verbose)
      
      # explore surface header
      search_block(bf, TAGS, 'SURFACE_HEADER', verbose=verbose)
      # get surface id
      a = search_block(bf, TAGS, 'SURFACE_ID', verbose=verbose)
      surf_id = read_bytes(bf, nb=a)
      surface_ids.append(surf_id)
      console_log("--surf_id: {}".format(surf_id),3,verbose)
      # get number of facets in surface
      search_block(bf, TAGS, 'SURFACE_N_FACETS', verbose=verbose)
      a = n_facets = int(read_bytes(bf, nb=a))
      console_log("--n_facets: {}".format(n_facets),3,verbose)
      # get surface name (if it exists)
      a = search_block(bf, TAGS, 'SURFACE_NAME', verbose=verbose)
      if a > 0:
        i_name = bf.read(a) # from docs, here should be CHAR64, but its DWORD from above
        try:
          surface_names.append(i_name.decode('ascii').split('\x00')[-1])
        except UnicodeDecodeError:
          surface_names.append(str(i_name).split('\\x00')[-1])
      else:
        surface_names.append('')
      # get max facet nodes
      a = search_block(bf, TAGS, 'MAX_FACET_NODES', verbose=verbose)
      nodes_per_facet = int(read_bytes(bf))

      # move pointer to facet list
      search_block(bf, TAGS, 'FACET_LIST', verbose=verbose)
      
      surf_facets = deque()
      for _ in range(n_facets):
        # move pointer to next facet
        a = search_block(bf, TAGS, 'FACET', verbose=verbose)
        # read facet nodes
        facet = nparray(read_bytes(bf, nb=a, format="I"*(nodes_per_facet+2)), dtype=int)
        # according to docs, we should ignore first two elements
        facet = facet[2:]
        # add facet to deque
        surf_facets.append(facet)

      try:
        surf_facets = nparray(surf_facets, dtype="int32")
      except:
        surf_facets = nparray(surf_facets, dtype="object")
      
      surface_facets.append(surf_facets)

    console_log("->surface items: [surface_ids, surface_names, surface_facets]", 2, verbose)
    console_log([surface_ids, surface_names, surface_facets], 3, verbose)
    
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