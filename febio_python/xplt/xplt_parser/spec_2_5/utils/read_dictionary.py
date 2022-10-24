def read_dictionary(bf, TAGS, verbose=0):
  from ...common.utils import search_block, check_block, read_bytes, console_log
  from collections import deque

  # move cursor to "DICTIONARY"
  search_block(bf, TAGS, 'DICTIONARY', verbose=verbose)

  # move cursor to "DIC_NODAL"
  search_block(bf, TAGS, 'DIC_NODAL', verbose=verbose)

  item_types = deque()
  item_formats = deque()  # 0: nodeal values, 1: elemental values
  item_names = deque()

  while check_block(bf, TAGS, 'DIC_ITEM'):
    # get item type
    search_block(bf, TAGS, 'DIC_ITEM', verbose=verbose)
    search_block(bf, TAGS, 'DIC_ITEM_TYPE', verbose=verbose)
    item_types.append(int(read_bytes(bf)))
    
    # get item format
    search_block(bf, TAGS, 'DIC_ITEM_FMT', verbose=verbose)
    item_formats.append(int(read_bytes(bf)))

    # get item name
    search_block(bf, TAGS, 'DIC_ITEM_NAME', verbose=verbose)
    i_name = bf.read(64)
    try:
      item_names.append(i_name.decode('ascii').split('\x00')[0])
    except UnicodeDecodeError:
      item_names.append(i_name.split(b'\x00')[0].decode('ascii'))

  # Explore Dictionary domain
  # move cursor to "DIC_DOMAIN"
  search_block(bf, TAGS, 'DIC_DOMAIN', verbose=verbose)

  while check_block(bf, TAGS, 'DIC_ITEM'):
    search_block(bf, TAGS, 'DIC_ITEM', verbose=verbose)
    search_block(bf, TAGS, 'DIC_ITEM_TYPE', verbose=verbose)
    item_types.append(int(read_bytes(bf)))

    search_block(bf, TAGS, 'DIC_ITEM_FMT', verbose=verbose)
    item_formats.append(int(read_bytes(bf)))

    search_block(bf, TAGS, 'DIC_ITEM_NAME', verbose=verbose)
    i_name = bf.read(64)
    try:
      item_names.append(i_name.decode('ascii').split('\x00')[0])
    except UnicodeDecodeError:
      item_names.append(i_name.split(b'\x00')[0].decode('ascii'))
  
  console_log("---read_dictionary:", 2,verbose)
  console_log("->dict items: [item_types, item_formats, item_names]", 2, verbose)
  console_log([item_types, item_formats, item_names], 2, verbose)

  states_dict = {"types": item_types, "formats": item_formats, "names": item_names}
  
  return states_dict