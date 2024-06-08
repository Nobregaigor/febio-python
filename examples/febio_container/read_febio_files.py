from febio_python import FEBioContainer
from time import time

# Load the FEBio file
filepath = "./sample_beam_2d.feb"
feb = FEBioContainer(feb=filepath)

start_time = time()
nodes = feb.nodes
end_time = time()
print(f"Time to get nodes: {end_time - start_time}")
start_time = time()
nodes = feb.nodes
end_time = time()
print(f"Time to get nodes: {end_time - start_time}")
start_time = time()
nodes = feb.nodes
end_time = time()
print(f"Time to get nodes: {end_time - start_time}")