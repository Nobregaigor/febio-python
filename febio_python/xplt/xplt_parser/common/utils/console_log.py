def print_list(a, dots=False, limit=5):
  if isinstance(a,list) and len(a) > limit:
    return print_list(a[:limit])

  if isinstance(a[0], list):
    return print_list(a[0][:limit], dots=True)

  if dots:
    print("[ ...")
    for i in a:
      print(i)
    print(" ... ]")
  else:
    print(a)


def console_log(to_print, verbose_mode, verbose):
  if verbose >= verbose_mode:
    if isinstance(to_print, str):
      print(to_print)

    elif isinstance(to_print, list):
      for a in to_print:
        if isinstance(a, list):
          print_list(a)
        else:
          print(a)
          
    else:
      try:
        print(to_print)
      except:
        pass