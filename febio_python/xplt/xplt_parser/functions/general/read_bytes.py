import struct

def read_bytes(bf, nb=4, format="I", p=False):
    a = bf.read(nb)
    if p:
        # print(len(a))
        print("a", a)
    data = struct.unpack(format, a)
    if len(data) == 1:
        return data[0]
    else:
        return data
    # return struct.unpack(format, a)[0]

