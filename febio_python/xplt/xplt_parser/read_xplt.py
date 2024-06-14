from pathlib import Path
from typing import Tuple, Union
from febio_python.utils.log import console_log
from febio_python.core.enums import XPLT_TAGS as TAGS
from febio_python.core import (
    XpltMesh,
    States,
)
from ._binary_file_reader_helpers import search_block, read_bytes, get_file_size


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
        raise RuntimeError("XPLT version 2.5 is no longer supported.")
    elif (version == TAGS['VERSION_3_0'].value):
        console_log('Current spec version is: 3.0 -> %d' % version, 2, verbose)
        return 3.0
    elif (version == 49):
        console_log('Current spec version is: 3.0 -> %d | WARNING: Docs say version should be 8, but it is 49.' % version, 2, verbose)
        return 3.0
    else:
        raise(ValueError(
            "Incorrect XPLIT file version: {}, expected version: {} or [{} or 49]"
            .format(version, int(TAGS['VERSION_2_5'], base=16), int(TAGS['VERSION_3_0'], base=16))))

def read_xplt(xplit_filepath: Union[Path, str], verbose: int=0) -> Tuple[XpltMesh, States]:
    """Reads a XPLT file and returns a XpltMesh and States object.

    Args:
        xplit_filepath (Union[Path, str]): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        Tuple[XpltMesh, States]: _description_
    """
    
    xplit_filepath = Path(xplit_filepath)
    if not xplit_filepath.exists():
        raise(FileNotFoundError(f"File not found: {xplit_filepath}"))
    
    # open binary file
    with open(xplit_filepath, "rb") as bf:

        # get file size and check if its not empty
        filesize = get_file_size(bf)
        if filesize == 0:
            raise(ValueError("Input xplit file size is zero. Please, check file."))

        # check if file format meets requirement
        check_fileformat(bf, verbose)

        console_log("Checking filer version", 1, verbose, header=True)

        # move cursor to "ROOT"
        search_block(bf, TAGS.ROOT, verbose=verbose)
        # move cursor to "HEADER"
        search_block(bf, TAGS.HEADER, verbose=verbose)
        # move cursor to "HDR_VERSION" -> feb file version
        search_block(bf, TAGS.HDR_VERSION, verbose=verbose)
        version = check_fileversion(bf, verbose)
        console_log(f"File version: {version}", 2, verbose)
        
    if version == 3.0:
        from ._spec_30 import read_spec30
        return read_spec30(xplit_filepath, verbose=verbose)
    else:
        raise(ValueError(f"XPLT file version not supported: {version}"
                        "We currently only support version 3.0."))
