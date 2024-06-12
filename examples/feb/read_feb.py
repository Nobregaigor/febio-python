from pathlib import Path
from febio_python import Feb
from febio_python.feb._feb_25 import Feb25
from febio_python.feb._feb_30 import Feb30
from febio_python.utils.pyvista_utils import febio_to_pyvista

this_dir = Path(__file__).parent
samples_dir = this_dir.parent / 'samples'

feb_filepath_v25 = samples_dir / "sample2d.feb"
feb_filepath_v30 = samples_dir / "sample2d_v3.feb"

feb_v25: Feb25 = Feb(filepath=feb_filepath_v25)
feb_v30: Feb30 = Feb(filepath=feb_filepath_v30)


print("FEB object for sample2d.feb")
print(feb_v25)
print("FEB object for sample2d_v3.feb")
print(feb_v30)

# print(feb_v30.get_elements())

# febio_to_pyvista(feb_v25).plot(show_edges=True)
# febio_to_pyvista(feb_v30).plot(show_edges=True)

print(feb_v30.get_mesh_domains())