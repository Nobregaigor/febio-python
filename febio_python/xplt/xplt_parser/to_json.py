

def to_json(xplt_filepath, output_filepath, verbose=0):
    from .read_xplt import read_xplt
    from .common.utils import NumpyEncoder
    import json
    
    data = read_xplt(xplt_filepath, -1, verbose=verbose)
    
    data_strkeys = dict()
    for key in data:
        if isinstance(data[key], dict):
            data_strkeys[key] = dict()
            for subkey in data[key]:
                data_strkeys[key][str(subkey)] = data[key][subkey]
        else:
            data_strkeys[key] = data[key]
    
    with open(output_filepath, "w") as outfile:
        # json.dump(serialized_d, outfile, indent=indent, sort_keys=sort_keys, **kwargs)
        json.dump(data_strkeys, outfile, 
                  indent=4,
                  sort_keys=False, cls=NumpyEncoder)