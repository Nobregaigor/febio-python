from .. import search_block, check_block, read_bytes, num_el_nodes, console_log 
from numpy import zeros as npzeros

def read_state(bf, TAGS, exit_at_data=True, exit_at_next_state=False, verbose=0):
  a_state = search_block(bf, TAGS, 'STATE', verbose=verbose)
  cur_cur = bf.tell()

  search_block(bf, TAGS, 'STATE_HEADER', verbose=verbose)
  search_block(bf, TAGS, 'STATE_HDR_TIME', verbose=verbose)
  time = read_bytes(bf, format="f")

  if exit_at_next_state:
    console_log('\n-> Skip state at %f time' % (time), 2, verbose)
  else:
    console_log('\n-> Read state at %f time' % (time), 1, verbose)

  if exit_at_next_state:
    bf.seek(cur_cur + a_state, 0)
    exit_at_data = False

  if exit_at_data:
    search_block(bf, TAGS, 'STATE_DATA', verbose=verbose)
    return time
  
  return time