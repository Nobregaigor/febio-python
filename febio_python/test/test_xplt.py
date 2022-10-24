import unittest
from febio_python import xplt


# class test_xplt(unittest.TestCase):
#     def test_loadxplt(self):
#         print("Loading xplt")
#         xplt.read_xplt("./sample_v3.7.xplt")


if __name__ == '__main__':
    # unittest.main()
    
    xplt.read_xplt("./sample_v3.7.xplt", -1, 3)

    
    