import struct
from .console_log import console_log

def search_block(f, TAGS, BLOCK_TAG, max_depth=5, cur_depth=0,
                 verbose=0, inv_TAGS=0, print_tag=0):

    # record the initial cursor position
    if cur_depth == 0:
        ini_pos = f.tell()

    if cur_depth > max_depth:
        console_log('Max iteration reached: Cannot find %s' % BLOCK_TAG, 2, verbose)
        return -1

    buf = f.read(4)
    
    if buf == b'':
        console_log('EOF: Cannot find %s' % BLOCK_TAG, 2, verbose)
        return -1
    else:
        cur_id = struct.unpack('I', buf)[0]

    a = struct.unpack('I', f.read(4))[0]  # size of the block

    cur_id_str = '0x' + '{0:08x}'.format(cur_id)
    console_log('cur_ID: {}'.format(cur_id_str), 3, verbose)
    if isinstance(inv_TAGS, dict):
        console_log('cur_tag: {}'.format(inv_TAGS[cur_id_str]) , 3, verbose)
    console_log('size: {}'.format(a), 3, verbose)

    if (int(TAGS[BLOCK_TAG], base=16) == cur_id):
        if print_tag == 1:
            print('%s' % BLOCK_TAG)
        return a

    else:
        f.seek(a, 1)
        d = search_block(f, TAGS, BLOCK_TAG, cur_depth=cur_depth + 1,
                         verbose=verbose,
                         inv_TAGS=inv_TAGS,
                         print_tag=print_tag)
        if d == -1:
            # put the cursor position back
            if cur_depth == 0:
                f.seek(ini_pos, 0)
            return -1
        else:
            return d