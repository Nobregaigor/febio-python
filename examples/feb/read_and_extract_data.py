from febio_python.feb.feb_object import Feb
from febio_python.utils.interops import feb_to_pyvista


filepath = "./sample_beam_2d.feb"

print("Reading an existing FEB file")
feb = Feb(filepath=filepath)

grid = feb_to_pyvista(feb)
grid.plot(show_edges=True, show_axes=True, cpos="xy")

# print("Successfully read the FEB file:")
# print(feb)

# print("Extracting data from the FEB file")
# print(f"Nodes: {feb.get_nodes()[0].coordinates.shape[0]}")
# print(f"Elements: {feb.get_elements()[0].connectivity.shape[0]}")


# print(feb.get_materials())

# from febio_python.feb.core.meta_data import Material
# new_material = Material(id=1, type="Mooney-Rivlin", parameters={'c1': 10, 'c2': 20, 'k': 10.0}, name="new_material", attributes=None)
# feb.add_materials([new_material])
# print(feb.get_materials())
# 

# # print(feb.get_materials())
# # 

# from febio_python.feb.core import Nodes
# new_nodes = Nodes(name="SMGNN_BENCHMARK", coordinates=feb.get_nodes()[0].coordinates[:2] + 1)
# print(new_nodes)

# # feb.add_nodes([new_nodes])

# # print(feb.get_nodes())