from .read_bytes import read_bytes

def seek_block(f, TAGS, BLOCK_TAG):
    """
    Search for a block in a binary file by comparing a predefined tag with a read value.
    
    Args:
    f: Binary file object.
    TAGS: IntEnum of tags with hexadecimal string values.
    BLOCK_TAG: Key for the tag to compare against.
    
    Returns:
    The result of the second read operation, regardless of comparison outcome.
    """
    target_id = int(TAGS[BLOCK_TAG], base=16)
    current_id = read_bytes(f)

    if target_id == current_id:
        print(f'Found block: {BLOCK_TAG}')
    
    return read_bytes(f)