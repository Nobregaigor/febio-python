from pathlib import Path
from typing import Tuple

from febio_python.utils.log import console_log
from febio_python.core.enums import XPLT_TAGS as TAGS
from febio_python.core import (
    XpltMesh,
    States,
    StatesDict
)

from .. import _binary_file_reader_helpers as bf_helpers
from .._binary_file_reader_helpers import search_block, read_bytes

from ._xplt_sections import (
    check_fileformat,
    read_dictionary,
    read_mesh,
    read_state,
)


def check_fileversion(bf, verbose):
    version = bf_helpers.read_bytes(bf)
    if (version == TAGS.VERSION_3_0.value):
        console_log(f'Current version is: {version}', 2, verbose)
    elif (version == 49):
        console_log(f'Current spec version is: 3.0 -> {version} | WARNING: Docs say version should be 8, but it is 49.', 2, verbose)
    elif (version == 52):
        console_log(f'Current spec version is: 3.0 -> {version} | WARNING: Possible new spec_version (4.0), but missing Xplt reader for 4.0', 2, verbose)
        return 3.0
    else:
        raise ValueError(
            f"Incorrect XPLIT file version: {version}, expected version: {TAGS.VERSION_2_5} or [{TAGS.VERSION_3_0} or 49]"
            # .format(version, int(TAGS.VERSION_2_5, base=16), int(TAGS.VERSION_3_0, base=16)))
        )


def read_spec30(filepath: Path, verbose=0) -> Tuple[XpltMesh, States]:

    with open(filepath, "rb") as bf:

        # Part 1: check file size
        # ------------------------------------
        # - get file size and check if its not empty
        filesize = bf_helpers.get_file_size(bf)
        if filesize == 0:
            raise ValueError("Input xplit file size is zero. Check file.")

        # Part 2: check file format
        # ------------------------------------
        # - check if file format meets requirement
        check_fileformat(bf, TAGS.FEBIO, verbose=verbose)

        # Part 3: Read header
        # ------------------------------------
        console_log("Reading header...", 1, verbose, header=True)

        # move cursor to "ROOT"
        search_block(bf, TAGS.ROOT, verbose=verbose)
        # move cursor to "HEADER"
        search_block(bf, TAGS.HEADER, verbose=verbose)
        # move cursor to "HDR_VERSION" -> feb file version
        search_block(bf, TAGS.HDR_VERSION, verbose=verbose)
        # check file version
        check_fileversion(bf, verbose)
        # move cursor to "HDR_COMPRESSION" and check wheter states are compressed or not
        a = search_block(bf, TAGS.HDR_COMPRESSION, verbose=verbose)
        file_is_compressed = read_bytes(bf, nb=a)
        console_log(f"Compression: {file_is_compressed}", 2, verbose)

        # Part 4: Read dictionary
        # ------------------------------------
        console_log("Reading dictionary...", 1, verbose, header=True)
        states_dict: StatesDict = read_dictionary(bf, verbose=verbose)

        # Part 5: Read mesh
        # ------------------------------------
        console_log("Reading mesh...", 1, verbose, header=True)
        mesh: XpltMesh = read_mesh(bf, verbose=verbose)

        # Part 6: Read Parts (skip for now)
        # ------------------------------------

        # Part 7: Read States
        # ------------------------------------
        states: States = read_state(bf, states_dict, verbose=verbose)

    return mesh, states
