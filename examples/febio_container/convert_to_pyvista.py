from pathlib import Path
from febio_python import FEBioContainer
from febio_python.utils.pyvista_utils import febio_to_pyvista

this_dir = Path(__file__).parent
samples_dir = this_dir.parent / 'samples'

feb_filepath_v30 = samples_dir / "plate_3D_sample.feb"
xplt_filepath_v30 = samples_dir / "plate_3D_sample.xplt"
febio_container_v30 = FEBioContainer(feb=feb_filepath_v30, xplt=xplt_filepath_v30)

from pprint import pprint

pprint(febio_container_v30.materials)

pyvista_data = febio_to_pyvista(febio_container_v30)
last_timestep = pyvista_data[-1]


print(last_timestep.field_data["mat_parameters:1"])
print(last_timestep.cell_data["mat_parameters:1"])
