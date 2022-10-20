# import struct
# import json
from os import path, curdir
import numpy as np
# import sys
import tempfile
import pathlib
from collections import deque



# from .functions import *  # pylint: disable=unused-wildcard-import
from .common.utils import * # pylint: disable=unused-wildcard

from .common.database import TAGS


# ==  MACROS == #
PATH_DATABASE = pathlib.Path(path.dirname(__file__), path.join("common", "database"))

# FILEPATH_TAGSDATA = PATH_DATABASE/"tags.json"
FILEPATH_ELEMTYPES = PATH_DATABASE/"element_types.json"
FILEPATH_NDSPERELEM = PATH_DATABASE/"nodes_per_elem.json"

# read database
# TAGS = read_json(FILEPATH_TAGSDATA)

# INV_TAGS = dict(zip(TAGS.values(), TAGS.keys()))


def get_filesize(bf):
    bf.seek(0, 2)
    filesize = bf.tell()
    bf.seek(0, 0)

    if filesize == 0:
        raise(ValueError("Input xplit file size is zero. Check file."))
    return filesize


def check_fileformat(bf, verbose):
    fformat = read_bytes(bf)
    if(TAGS['FEBIO'].value == fformat):
        console_log('Correct FEBio format.', 2, verbose)
    else:
        raise(ValueError("Input XPLIT file does not have the correct format. Expected: {}, received: {}".format(TAGS['FEBIO'].value, fformat)))


def check_fileversion(bf, verbose):
    version = read_bytes(bf)
    # return version
    if (version == TAGS['VERSION_2_5'].value):
        console_log('Current spec version is: 2.5 -> %d' % version, 2, verbose)
        return 2.5
    elif (version == TAGS['VERSION_3_0'].value):
        console_log('Current spec version is: 3.0 -> %d' % version, 2, verbose)
        return 3.0
    elif (version == 49):
        console_log('Current spec version is: 3.0 -> %d | WARNING: Docs say version should be 8, but it is 49.' % version, 2, verbose)
        return 3.0
    else:
        raise(ValueError(
            "Incorrect XPLIT file version: {}, expected version: {} or {}"
            .format(version, int(TAGS['VERSION_2_5'], base=16), int(TAGS['VERSION_3_0'], base=16))))


def read_xplt(xplit_filepath, nstate=-1, verbose=0):
    
    # open binary file
    bf = open(xplit_filepath, "rb")

    # get file size and check if its not empty
    filesize = get_filesize(bf)

    # check if file format meets requirement
    check_fileformat(bf, verbose)

    console_log("\n___Begin read header___\n", 1, verbose)

    # move cursor to "ROOT"
    search_block(bf, TAGS, 'ROOT', verbose=verbose)
    # move cursor to "HEADER"
    search_block(bf, TAGS, 'HEADER', verbose=verbose)
    # move cursor to "HDR_VERSION" -> feb file version
    search_block(bf, TAGS, 'HDR_VERSION', verbose=verbose)
    version = check_fileversion(bf, verbose)
    
    # close binary file
    bf.close()
    
    if version == 2.5:
        from .spec_2_5 import read_xplt_2_5
        return read_xplt_2_5(xplit_filepath, nstate=nstate, verbose=verbose)
    elif version == 3.0:
        from .spec_3_0 import read_xplt_3_0
        return read_xplt_3_0(xplit_filepath, nstate=nstate, verbose=verbose)