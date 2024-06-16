import pathlib
import pyvista as pv    # For visualization
from febio_python.container import FEBioContainer
from febio_python.utils.pyvista_utils import febio_to_pyvista


samples_dir = pathlib.Path(__file__).parent.parent / "samples"
filepath = samples_dir / "sample_cfd.xplt"

if __name__ == "__main__":
    # read xplt file and load it into a container
    container = FEBioContainer(xplt=filepath)
    # print the xplt object
    print(container.xplt)
    # print(container.xplt)
    last_grid = febio_to_pyvista(container)[-1]
    # extract volumes (hexahedrons in this case):
    last_grid = last_grid.extract_cells_by_type(pv.CellType.HEXAHEDRON)
    # convert to nodal data (for visualization)
    last_grid = last_grid.cell_data_to_point_data()
    # get sample data:
    fluid_stress = last_grid["fluid stress"]
    # plot
    last_grid.plot(scalars=fluid_stress[:, 0], cmap="coolwarm",
                   scalar_bar_args={"title": "Fluid Stress - XX"}, show_edges=True)
