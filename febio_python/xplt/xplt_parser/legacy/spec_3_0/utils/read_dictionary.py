from ...common.utils import search_block, check_block, read_bytes, console_log
from collections import deque
from enum import IntEnum

def read_items(bf, TAGS, verbose):
    item_types = deque()
    item_formats = deque()
    item_names = deque()

    while check_block(bf, TAGS, 'DIC_ITEM'):
        search_block(bf, TAGS, 'DIC_ITEM', verbose=verbose)

        # Read item type
        search_block(bf, TAGS, 'DIC_ITEM_TYPE', verbose=verbose)
        item_types.append(int(read_bytes(bf)))

        # Read item format
        search_block(bf, TAGS, 'DIC_ITEM_FMT', verbose=verbose)
        item_formats.append(int(read_bytes(bf)))

        # Read item name
        search_block(bf, TAGS, 'DIC_ITEM_NAME', verbose=verbose)
        i_name = bf.read(64)
        try:
            item_name = i_name.decode('ascii').split('\x00')[0]
        except UnicodeDecodeError:
            item_name = i_name.split(b'\x00')[0].decode('ascii')
        item_names.append(item_name)

    return item_types, item_formats, item_names

def read_dictionary(bf, TAGS: IntEnum, verbose:int = 0):
    # Navigate to "DICTIONARY" and "DIC_NODAL"
    search_block(bf, TAGS, 'DICTIONARY', verbose=verbose)
    search_block(bf, TAGS, 'DIC_NODAL', verbose=verbose)

    # Read nodal items
    item_types, item_formats, item_names = read_items(bf, TAGS, verbose)

    # Navigate to "DIC_DOMAIN" and read domain items
    search_block(bf, TAGS, 'DIC_DOMAIN', verbose=verbose)
    domain_types, domain_formats, domain_names = read_items(bf, TAGS, verbose)

    # Append domain items to nodal items
    item_types.extend(domain_types)
    item_formats.extend(domain_formats)
    item_names.extend(domain_names)

    console_log("---read_dictionary:", 2, verbose)
    console_log("->dict items: [item_types, item_formats, item_names]", 2, verbose)
    console_log([item_types, item_formats, item_names], 2, verbose)

    states_dict = {"types": item_types, "formats": item_formats, "names": item_names}

    return states_dict
