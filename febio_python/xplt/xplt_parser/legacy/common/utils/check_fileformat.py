from .read_bytes import read_bytes
from .console_log import console_log
from enum import IntEnum

def check_fileformat(bf, TAGS:IntEnum, verbose=0):
    """
    Check if the binary file starts with a specific tag indicating the correct file format.
    
    Args:
        bf: Binary file object.
        TAGS: IntEnum with their corresponding values.
        verbose: Verbosity level for logging.
    
    Throws:
        ValueError: If the file does not start with the expected format tag.
    """
    # Attempt to read the format tag from the file
    try:
        file_format_tag = read_bytes(bf)
    except Exception as e:
        console_log(f"Failed to read file format: {e}", 2, verbose)
        raise ValueError("Failed to read from the file, possibly due to an IO error.")

    # Check if the read tag matches the expected 'FEBIO' tag
    if TAGS['FEBIO'] == file_format_tag:
        console_log('Correct FEBio format', 2, verbose)
    else:
        raise ValueError("Input XPLIT file does not have the correct format.")
