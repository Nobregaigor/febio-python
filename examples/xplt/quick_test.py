# from febio_python.xplt import read_xplt
from febio_python.xplt import read_xplt
from febio_python.xplt.xplt_object import Xplt
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
    
    xplt_obj = Xplt(filepath)
