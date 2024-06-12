# from febio_python.xplt import read_xplt
from febio_python.xplt import read_xplt
from time import time
import pathlib

samples_dir = pathlib.Path(__file__).parent.parent / "samples"
filepath = samples_dir / "sample2d_v3.xplt"


def measure_performance(file_path, runs=50):
    times = []
    
    for _ in range(runs):
        start = time()
        # Read xplt file
        xplt_data = read_xplt(file_path, verbose=0)
        elapsed_time = time() - start
        times.append(elapsed_time)
        
    avg_time = sum(times) / len(times)
    print(f"Average elapsed time over {runs} runs: {avg_time:.4f} seconds")
    
    return xplt_data

if __name__ == "__main__":
    data = measure_performance(filepath)
    mesh, states = read_xplt(filepath, verbose=0)
    print(states.nodes[0].data.shape)
    print(states.timesteps.shape)
    # print(mesh.elements)
    # print(data.keys())
    # print(data["N_DOMAINS"])
    # print(data["NODES"])
    # print(data["ELEMENTS"])
    # print("STATES:")
    # print(data["STATES"])