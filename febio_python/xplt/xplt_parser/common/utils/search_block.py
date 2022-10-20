import struct
from .console_log import console_log

def search_block(f, TAGS, BLOCK_TAG, max_depth=5, cur_depth=0,
                 verbose=0, inv_TAGS=0):

    console_log('Search for BLOCK_TAG: {}'.format(BLOCK_TAG), 2, verbose)
    # record the initial cursor position
    if cur_depth == 0:
        ini_pos = f.tell()
    # check if we passed the target
    if cur_depth > max_depth:
        console_log('Max iteration depth reached: Cannot find %s.' % BLOCK_TAG, 2, verbose)
        return -1
    # read block
    buf = f.read(4)
    
    # check for end of file
    if buf == b'':
        console_log('EOF: Cannot find %s' % BLOCK_TAG, 2, verbose)
        return -1
    else:
        cur_id = struct.unpack('I', buf)[0]

    # read size of block (content after header)
    a = struct.unpack('I', f.read(4))[0]  # size of the block

    # for debugging
    if verbose == 3:
        cur_id_str = '0x' + '{0:08x}'.format(cur_id)
        if TAGS.in_values(cur_id):
            console_log('cur_ID: {} -> {} | searching for: {}'.format(cur_id_str, TAGS(cur_id).name, BLOCK_TAG), 4, verbose)
        else:
            console_log('cur_ID: {} -> NOT IN TAGS | searching for: {}'.format(cur_id_str, BLOCK_TAG), 4, verbose)
        console_log('-cur_depth: {}, max_depth: {}'.format(cur_depth, max_depth), 4, verbose)
        console_log('-block size: {}'.format(a), 4, verbose)
        

    if (TAGS[BLOCK_TAG].value == cur_id):
        console_log('Found BLOCK_TAG: {}'.format(BLOCK_TAG), 3, verbose)
        return a


    else:
        f.seek(a, 1)
        d = search_block(f, TAGS, BLOCK_TAG, 
                         cur_depth=cur_depth + 1,
                         verbose=verbose,
                         inv_TAGS=inv_TAGS)
        if d == -1:
            # put the cursor position back
            if cur_depth == 0:
                f.seek(ini_pos, 0)
            return -1
        else:
            return d

# def search_block(f, TAGS, BLOCK_TAG, max_depth=5, cur_depth=0,
#                  verbose=0, inv_TAGS=0, print_tag=0):

#     # record the initial cursor position
#     if cur_depth == 0:
#         ini_pos = f.tell()

#     if cur_depth > max_depth:
#         console_log('Max iteration reached: Cannot find %s' % BLOCK_TAG, 2, verbose)
#         return -1

#     buf = f.read(4)
    
#     if buf == b'':
#         console_log('EOF: Cannot find %s' % BLOCK_TAG, 2, verbose)
#         return -1
#     else:
#         cur_id = struct.unpack('I', buf)[0]

#     a = struct.unpack('I', f.read(4))[0]  # size of the block

#     cur_id_str = '0x' + '{0:08x}'.format(cur_id)
#     console_log('cur_ID: {}'.format(cur_id_str), 3, verbose)
#     if isinstance(inv_TAGS, dict):
#         console_log('cur_tag: {}'.format(inv_TAGS[cur_id_str]) , 3, verbose)
#     console_log('size: {}'.format(a), 3, verbose)

#     if (int(TAGS[BLOCK_TAG], base=16) == cur_id):
#         if print_tag == 1:
#             print('%s' % BLOCK_TAG)
#         return a

#     else:
#         f.seek(a, 1)
#         d = search_block(f, TAGS, BLOCK_TAG, cur_depth=cur_depth + 1,
#                          verbose=verbose,
#                          inv_TAGS=inv_TAGS,
#                          print_tag=print_tag)
#         if d == -1:
#             # put the cursor position back
#             if cur_depth == 0:
#                 f.seek(ini_pos, 0)
#             return -1
#         else:
#             return d