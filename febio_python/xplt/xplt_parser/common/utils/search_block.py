import struct
from .console_log import console_log
from enum import IntEnum


def search_block(f, TAGS: IntEnum, BLOCK_TAG: str, max_depth: int = 5, cur_depth: int = 0, verbose: int = 0):
    # Log the start of a new search at the first depth level
    if cur_depth == 0:
        ini_pos = f.tell()
        console_log(f'Searching for BLOCK_TAG: {BLOCK_TAG}', 2, verbose)

    # Stop recursion if maximum depth exceeded
    if cur_depth > max_depth:
        console_log(f'Max iteration depth reached: Cannot find {BLOCK_TAG}.', 2, verbose)
        return -1

    # Read and unpack the block identifier
    buf = f.read(4)
    if buf == b'':  # Check for end of file
        console_log(f'EOF: Cannot find {BLOCK_TAG}', 2, verbose)
        return -1
    cur_id = struct.unpack('I', buf)[0]

    # Read and unpack the size of the block
    block_size = struct.unpack('I', f.read(4))[0]

    # Debugging information
    if verbose >= 3:
        cur_id_str = f'0x{cur_id:08x}'
        id_name = TAGS(cur_id).name if cur_id in TAGS._value2member_map_ else "NOT IN TAGS"
        console_log(f'cur_ID: {cur_id_str} -> {id_name} | searching for: {BLOCK_TAG}', 4, verbose)
        console_log(f'-cur_depth: {cur_depth}, max_depth: {max_depth}', 4, verbose)
        console_log(f'-block size: {block_size}', 4, verbose)

    # Check if current block matches the search tag
    if TAGS[BLOCK_TAG].value == cur_id:
        console_log(f'Found BLOCK_TAG: {BLOCK_TAG}', 3, verbose)
        return block_size
    else:
        # Recursively search within the block
        f.seek(block_size, 1)
        result = search_block(f, TAGS, BLOCK_TAG, max_depth, cur_depth + 1, verbose)
        if result == -1 and cur_depth == 0:
            f.seek(ini_pos)  # Restore file position only at the topmost level
        return result