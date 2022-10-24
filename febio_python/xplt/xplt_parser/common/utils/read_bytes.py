
def read_bytes(bf, nb=4, format="I"):
    import struct
    
    a = bf.read(nb)
    data = struct.unpack(format, a)
    if len(data) == 1:
        return data[0]
    else:
        return data
