import pyvista as pv
from pathlib import Path
from febio_python import FEBioContainer
from febio_python.utils.pyvista_utils import febio_to_pyvista


this_dir = Path(__file__).parent
samples_dir = this_dir.parent / 'samples'

feb_filepath = samples_dir / "beam_3D_non_uniform_pressure.feb"
febio_container = FEBioContainer(feb=feb_filepath, auto_find=True)

grids = febio_to_pyvista(febio_container)

sample = grids[-1]
print([pv.CellType(k) for k in sample.cells_dict.keys()])

surf = sample.extract_cells_by_type([pv.CellType.TRIANGLE])
# surf.plot(scalars="pressure_load")
plotter = pv.Plotter()
plotter.add_mesh(surf, scalars="pressure_load_magnitude")
plotter.add_arrows(surf.cell_centers().points, surf["pressure_load"], mag=0.1)
plotter.show_grid()
plotter.show()