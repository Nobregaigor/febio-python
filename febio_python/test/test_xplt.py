import unittest
from febio_python import xplt


# class test_xplt(unittest.TestCase):
#     def test_loadxplt(self):
#         print("Loading xplt")
#         xplt.read_xplt("./sample_v3.7.xplt")


if __name__ == '__main__':
    # unittest.main()
    
    xplt.read_xplt("./sample_v3.7.xplt", -1, 3)
    xplt.to_json("./sample_v3.7.xplt", "./sample_v3.7.json")
    
    # xplt.read_xplt("./testing.xplt", -1, 3)
    
    
    # import required modules
    import json
    import numpy as np
    
    # open and read json file
    with open("./sample_v3.7.json", "r") as jfile:
        data = json.load(jfile)
    
    # extract required data to variables
    nodes = data["NODES"]
    elems = data["ELEMENTS"][0]
    
    # convert data to numpy arrays
    nodes = np.asarray(nodes, dtype=np.float32)
    elems = np.asarray(elems, dtype=np.int32)
    
    # display info
    print("Number of nodes: {}".format(nodes.shape[0]))
    print("Number of dims: {} -> [X,Y,Z]".format(nodes.shape[1]))
    print("Nodes:")
    print(nodes)
    print("Number of elements: {}".format(elems.shape[0]))
    print("Number of edges per element: {}".format(elems.shape[1]))
    print("Elements:")
    print(elems)
    
    
    
    