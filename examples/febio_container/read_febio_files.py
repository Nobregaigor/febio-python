from febio_python import FEBioContainer
from time import time
from febio_python.core import Nodes

# Load the FEBio file
filepath = "./sample_beam_2d.feb"
febio_container = FEBioContainer(feb=filepath)

start_time = time()
print(febio_container.feb)
end_time = time()
print(f"Time to get print: {end_time - start_time}")

start_time = time()
nodes = febio_container.nodes
end_time = time()
print(f"Time to get nodes: {end_time - start_time}")
start_time = time()
nodes = febio_container.nodes
end_time = time()
print(f"Time to get nodes: {end_time - start_time}")
start_time = time()
nodes = febio_container.nodes
end_time = time()
print(f"Time to get nodes: {end_time - start_time}")

# nodes[0].coordinates += 1
coords = nodes[0].coordinates
coords += 1

new_nodes = Nodes(name="new_nodes", coordinates=coords)

febio_container.feb.add_nodes([new_nodes])

start_time = time()
nodes = febio_container.nodes
end_time = time()
print(f"Time to get nodes: {end_time - start_time}")
print(nodes)

# start_time = time()
# nodes = febio_container.nodes
# end_time = time()
# print(f"Time to get nodes: {end_time - start_time}")
# start_time = time()
# nodes = febio_container.nodes
# end_time = time()
# print(f"Time to get nodes: {end_time - start_time}")