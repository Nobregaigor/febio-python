from pathlib import Path
from febio_python import FEBioContainer
from febio_python.utils.vtk_utils import febio_to_vtk

this_dir = Path(__file__).parent
samples_dir = this_dir.parent / 'samples'

feb_filepath_v30 = samples_dir / "sample2d_v3.feb"
xplt_filepath_v30 = samples_dir / "sample2d_v3.xplt"
febio_container_v30 = FEBioContainer(feb=feb_filepath_v30, xplt=xplt_filepath_v30)

febio_to_vtk(febio_container_v30, output_directory=samples_dir)