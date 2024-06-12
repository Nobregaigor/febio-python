
def decompress_state(bf, verbose=0):
    from .console_log import console_log
    from .decompress import decompress
    import tempfile
    
    console_log("\n_____Begin decompressing states_____", 1, verbose)
    alldata = bf.read(-1)
    bf.close()
    
    decompressed = decompress(alldata, verbose=verbose)
    bf = tempfile.TemporaryFile()
    bf.write(decompressed)
    bf.seek(0)
    filesize = len(decompressed)
    
    return bf