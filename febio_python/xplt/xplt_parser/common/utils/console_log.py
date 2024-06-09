# def print_list(a, dots=False, limit=5):
#   if isinstance(a,list) and len(a) > limit:
#     return print_list(a[:limit])

#   if isinstance(a[0], list):
#     return print_list(a[0][:limit], dots=True)

#   if dots:
#     print("[ ...")
#     for i in a:
#       print(i)
#     print(" ... ]")
#   else:
#     print(a)


def console_log(to_print, verbose_mode, verbose):
    # Check if the verbosity level allows for logging
    if verbose >= verbose_mode:
        # Handle string messages directly
        if isinstance(to_print, str):
            print(to_print)
        # Handle lists by iterating through each element
        elif isinstance(to_print, list):
            for a in to_print:
                print(a if not isinstance(a, list) else ' '.join(map(str, a)))
        # Attempt to print all other data types
        else:
            try:
                print(to_print)
            except Exception as e:
                print(f"Error printing: {e}")