def check_fileformat(bf, TAGS, verbose=0):
    from .read_bytes import read_bytes
    from .console_log import console_log
    
    if(TAGS['FEBIO'] == read_bytes(bf)):
        console_log('Correct FEBio format', 2, verbose)
    else:
        raise(ValueError("Input XPLIT file does not have the correct format."))