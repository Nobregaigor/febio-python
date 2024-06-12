import struct

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
