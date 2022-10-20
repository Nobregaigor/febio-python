def decompress(data, min_bytes=0, decompressed=b"", i=0, verbose=0):
    import zlib
    from .console_log import console_log

    console_log("decompressing ...", 1, verbose)

    total_data_len = len(data)
    DecoObj = zlib.decompressobj()
    decompressed += DecoObj.decompress(data)
    unused_data = DecoObj.unused_data

    if min_bytes == 0:
        min_bytes = len(decompressed) / 2

    max_count = 10000 # prevent infinity loop
    count = 0
    while len(unused_data) >= min_bytes and count < max_count:
        console_log("--> decompressing: {:.3f} %".format(100 * (1 - len(unused_data)/total_data_len)), 2, verbose)
        DecoObj = zlib.decompressobj()
        decompressed += DecoObj.decompress(unused_data)
        unused_data = DecoObj.unused_data
        
        count += 1
    
    console_log("--> decompressing: {:.3f}".format(100), 2, verbose)
    
    return decompressed