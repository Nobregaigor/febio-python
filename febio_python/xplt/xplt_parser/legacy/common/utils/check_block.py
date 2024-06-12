from .read_bytes import read_bytes
from .console_log import console_log

def check_block(bf, TAGS, BLOCK_TAG, filesize=-1, verbose=0):
    """
    Check if the current position in the binary file matches a specific block tag.
    
    Args:
        bf: Binary file object.
        TAGS: Enum or dictionary holding block tags and their corresponding values.
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
    expected_id = TAGS[BLOCK_TAG].value
    if block_id == expected_id:
        return 1

    return 0
