from febio_python import FEBioContainer
from time import time
from febio_python.core import Nodes
from febio_python.utils.pyvista_utils import febio_to_pyvista

# Load the FEBio file
# filepath = "./sample_heterogeneous_beam2d.feb"
filepath = "./sample_beam2d.feb"
febio_container = FEBioContainer(feb=filepath)
print(febio_container.load_curves)
# febio_container.element_data
# print(febio_container.feb.get_nodesets())

# feb = febio_container.feb
# nodesets: dict = feb.get_tag_data(feb.LEAD_TAGS.GEOMETRY, feb.MAJOR_TAGS.NODESET, content_type="id")
# print(nodesets)

mb = febio_to_pyvista(febio_container)
# mb.
# for name, grid in mb.items():
#     print(name)
#     print(grid)
#     grid.plot(show_edges=True, show_axes=True, cpos="xy")
    
grid = mb[0]
# print(grid.name)

print(f"point data: {grid.point_data.keys()}")
print(f"cell data: {grid.cell_data.keys()}")
print(f"field data: {grid.field_data.keys()}")
print(f"nodal load: {grid['nodal_load']}")

import pyvista as pv
# pv.set_plot_theme("document")
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.add_arrows(grid.points, grid["nodal_load"], mag=1000.0)
plotter.show(cpos="xy")

# grid.plot(scalars="neo-Hookean:SMGNN_BENCHMARK_MATERIAL", show_edges=True)


# print(grid[0].points)
# print(grid.get_block_name(0))
# grid["SMGNN_BENCHMARK"].plot(show_edges=True)
# grid.plot(show_edges=True)

# start_time = time()
# print(febio_container.feb)
# end_time = time()
# print(f"Time to get print: {end_time - start_time}")

# start_time = time()
# nodes = febio_container.nodes
# end_time = time()
# print(f"Time to get nodes: {end_time - start_time}")
# start_time = time()
# nodes = febio_container.nodes
# end_time = time()
# print(f"Time to get nodes: {end_time - start_time}")
# start_time = time()
# nodes = febio_container.nodes
# end_time = time()
# print(f"Time to get nodes: {end_time - start_time}")

# # nodes[0].coordinates += 1
# coords = nodes[0].coordinates
# coords += 1

# new_nodes = Nodes(name="new_nodes", coordinates=coords)

# febio_container.feb.add_nodes([new_nodes])

# start_time = time()
# nodes = febio_container.nodes
# end_time = time()
# print(f"Time to get nodes: {end_time - start_time}")
# print(nodes)

# # start_time = time()
# # nodes = febio_container.nodes
# # end_time = time()
# # print(f"Time to get nodes: {end_time - start_time}")
# # start_time = time()
# # nodes = febio_container.nodes
# # end_time = time()
# # print(f"Time to get nodes: {end_time - start_time}")