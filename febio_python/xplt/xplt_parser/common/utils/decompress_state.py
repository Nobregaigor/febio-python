
def decompress_state(bf, verbose=0):
    from .decompress import decompress
    
    console_log("\n_____Begin decompressing states_____", 1, verbose)
    alldata = bf.read(-1)
    bf.close()
    
    decompressed = decompress(alldata, verbose=verbose)
    bf = tempfile.TemporaryFile()
    bf.write(decompressed)
    bf.seek(0)
    filesize = len(decompressed)
    
    return bf