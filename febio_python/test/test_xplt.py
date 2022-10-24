import unittest
from febio_python import xplt


# class test_xplt(unittest.TestCase):
#     def test_loadxplt(self):
#         print("Loading xplt")
#         xplt.read_xplt("./sample_v3.7.xplt")


if __name__ == '__main__':
    # unittest.main()
    
    data = xplt.read_xplt("./sample_v3.7.xplt", -1, 1)
    # data = xplt.read_xplt("./sample_v2.5.xplt", -1, 1)
    # data = xplt.read_xplt("C:/Users/igorp/Downloads/65.55_65.55_1_65.55_65.55_02.08.2022.1_LVRNN_TYPEA_V4.xplt", -1, 1)
    
    # print("N nodes: ", data["N_NODES"])
    # print("N elements: ", data["N_ELEMENTS"])
    
    
    # print(data["N_STATES"])
    
    # print("displacement shape:", data["STATES"]["displacement"].shape)
    # # print("--> after reshape:", data["STATES"]["displacement"].reshape((-1, 3)))
    # print("stress shape:", data["STATES"]["stress"].shape)
    
    # print("--> after reshape:", data["STATES"]["stress"].reshape((-1, 6)))