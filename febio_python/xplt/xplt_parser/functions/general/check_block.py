from .read_bytes import read_bytes
from .console_log import console_log

def check_block(bf, TAGS, BLOCK_TAG, filesize=-1, verbose=0):
    if filesize > 0:
        if bf.tell() + 4 > filesize:
            console_log("EOF reached", 1, verbose)
            return 0

    buf = read_bytes(bf)
    bf.seek(-4, 1)

    if(int(TAGS[BLOCK_TAG], base=16) == buf):
        return 1

    return 0