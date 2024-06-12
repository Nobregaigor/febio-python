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

def console_log(to_print, level, verbose, header=False, indent=0):
    # Check if the verbosity level allows for logging
    if verbose >= level:
        if header:
            print(f"\n{'-'*20} {to_print} {'-'*20}")
        if isinstance(to_print, list):
            for a in to_print:
                value = a if not isinstance(a, list) else ' '.join(map(str, a))
                print(f"{'-'*indent}{value}")
        elif isinstance(to_print, dict):
            for key, value in to_print.items():
                print(f"{'-'*indent}{key}: {value}")
        elif isinstance(to_print, tuple):
            for a in to_print:
                print(f"{'-'*indent}{a}")   
        # Attempt to print all other data types
        else:
            try:
                print(f"{'-'*indent}{to_print}")
            except Exception as e:
                print(f"Error printing: {e}")
