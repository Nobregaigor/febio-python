def get_file_size(bf):
    curr_pos = bf.tell()
    
    bf.seek(0, 2)
    filesize = bf.tell()
    bf.seek(curr_pos)

    if filesize == 0:
        raise(ValueError("File size is zero. Please, check file."))
    return filesize