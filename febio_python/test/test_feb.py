import unittest
from febio_python import feb


class test_feb(unittest.TestCase):
    def test_load_feb(self):
        feb.FEBio_feb("./sample.feb")
