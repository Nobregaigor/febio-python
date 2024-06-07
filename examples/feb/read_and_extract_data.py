from febio_python.feb.feb_object import Feb


filepath = "./sample_beam_2d.feb"

print("Reading an existing FEB file")
feb = Feb(filepath=filepath)
print("Successfully read the FEB file:")
print(feb)

print("Extracting data from the FEB file")
print(f"Nodes: {feb.get_nodes()}")
print(f"Elements: {feb.get_elements()}")

print("Node data:")
print(feb.get_nodal_data())

