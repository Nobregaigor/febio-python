from pathlib import Path
from typing import Tuple

from febio_python.core import XpltMesh, States
from .._spec_30.reader import read_spec30


def read_spec40(filepath: Path, verbose=0) -> Tuple[XpltMesh, States]:
    try:
        return read_spec30(filepath, verbose)
    except Exception as e:
        raise RuntimeError(
            "Failed to read XPLT file with spec version >= 4.0. "
            "This is likely due to some changes in the binary file format, "
            "FEBio has not provided documentation for this version yet. "
            "In most scenarions, the reader for spec version 3.0 works, "
            "However, some data might be missing or incorrect. "
            "Please contact the developers for support.") from e
