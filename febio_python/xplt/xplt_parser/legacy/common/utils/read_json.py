def read_json(filepath, directory=None):
    import json
    from os import path

    if directory:
        filepath = path.join(directory, filepath)
    
    with open(filepath, "r") as db_file:
        data = json.load(db_file)

    return data