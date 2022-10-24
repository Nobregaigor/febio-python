def seek_block(f, TAGS, BLOCK_TAG):
    from .read_bytes import read_bytes
    
    if(int(TAGS[BLOCK_TAG], base=16) == read_bytes(bf)):
        print('%s' % BLOCK_TAG)

    return read_bytes(bf)