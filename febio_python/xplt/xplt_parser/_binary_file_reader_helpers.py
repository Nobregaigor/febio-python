import struct
from enum import IntEnum
from typing import Union
from febio_python.core.enums import XPLT_TAGS as TAGS
from febio_python.utils.log import console_log


def get_file_size(bf):
    curr_pos = bf.tell()
    
    bf.seek(0, 2)
    filesize = bf.tell()
    bf.seek(curr_pos)

    if filesize == 0:
        raise(ValueError("File size is zero. Please, check file."))
    return filesize

def read_bytes(bf, nb=4, format="I"):
    """
    Read a specific number of bytes from a binary file and unpack them into a tuple or a single value.
    
    Args:
    bf: Binary file object.
    nb: Number of bytes to read (default is 4).
    format: Format string for unpacking (default is "I" for an unsigned integer).
    
    Returns:
    A single unpacked value if only one value is unpacked, otherwise a tuple of unpacked values.
    """
    data_bytes = bf.read(nb)
    unpacked_data = struct.unpack(format, data_bytes)
    return unpacked_data[0] if len(unpacked_data) == 1 else unpacked_data

def search_block(f, BLOCK_TAG: IntEnum, max_depth: int = 5, cur_depth: int = 0, verbose: int = 0):
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
    if BLOCK_TAG.value == cur_id:
        console_log(f'Found BLOCK_TAG: {BLOCK_TAG}', 3, verbose)
        return block_size
    else:
        # Recursively search within the block
        f.seek(block_size, 1)
        result = search_block(f, BLOCK_TAG, max_depth, cur_depth + 1, verbose)
        if result == -1 and cur_depth == 0:
            f.seek(ini_pos)  # Restore file position only at the topmost level
        return result

def check_block(bf, BLOCK_TAG, filesize=-1, verbose=0):
    """
    Check if the current position in the binary file matches a specific block tag.
    
    Args:
        bf: Binary file object.
        BLOCK_TAG: The specific block tag to check.
        filesize: Optional total size of the file for EOF checking.
        verbose: Verbosity level for logging.
    
    Returns:
        1 if the block matches, 0 otherwise.
    """
    # Check for EOF before attempting to read
    if filesize > 0 and bf.tell() + 4 > filesize:
        console_log("EOF reached before checking block", 1, verbose)
        return 0

    # Read the next identifier from the file
    block_id = read_bytes(bf)

    # Reset the file pointer to the original position before the read
    bf.seek(-4, 1)

    # Compare the read block identifier against the expected block tag value
    expected_id = BLOCK_TAG.value
    if block_id == expected_id:
        return 1

    return 0
