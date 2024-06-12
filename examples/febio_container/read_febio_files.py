from febio_python import FEBioContainer
from time import time
from febio_python.core import Nodes
from febio_python.utils.pyvista_utils import febio_to_pyvista

from pathlib import Path

from febio_python.utils.pyvista_utils import febio_to_pyvista

this_dir = Path(__file__).parent
samples_dir = this_dir.parent / 'samples'

feb_filepath_v25 = samples_dir / "sample2d.feb"
feb_filepath_v30 = samples_dir / "sample2d_v3.feb"

xplt_filepath_v25 = samples_dir / "sample2d.xplt"
xplt_filepath_v30 = samples_dir / "sample2d_v3.xplt"

febio_container_v25 = FEBioContainer(feb=feb_filepath_v25, xplt=xplt_filepath_v25)
febio_container_v30 = FEBioContainer(feb=feb_filepath_v30, xplt=xplt_filepath_v30)

# grid = febio_to_pyvista(febio_container_v25)
grid = febio_to_pyvista(febio_container_v30)[-1]

print(f"point data: {grid.point_data.keys()}")
print(f"cell data: {grid.cell_data.keys()}")
print(f"field data: {grid.field_data.keys()}")

fixed = grid["fix"].sum(1)
strain = grid["Lagrange strain"][:, 0]

import pyvista as pv
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars=strain, cmap="coolwarm", show_edges=True, scalar_bar_args={"title": "Strain - XX"})
plotter.add_mesh(grid.points, scalars=fixed, cmap="viridis", style="points", point_size=10, 
                 render_points_as_spheres=True, show_scalar_bar=False)
plotter.add_arrows(grid.points, grid["nodal_load"], mag=10000.0, show_scalar_bar=False, color="orange")
plotter.show(cpos="xy")
