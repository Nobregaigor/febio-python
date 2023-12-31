from typing import Union, List
from pathlib import Path
import subprocess
import shutil
from tqdm import tqdm


def run(files: Union[str, List[Union[str, Path]]],
        num_processes: int = 1,
        executable: Union[str, Path] = "febio3.exe"):

    files_to_run = []

    # Ensure that files is a list of strings or pathlib.Path objects
    if isinstance(files, (str, Path)):
        files_to_run.append(files)
    else:
        assert isinstance(files, list), (
            "files must be a string or a list of strings or pathlib.Path objects. "
            f"Got {type(files)} instead."
        )
        for i, item in enumerate(files):
            assert isinstance(item, (str, Path)), (
                "files must be a string or a list of strings or pathlib.Path objects. "
                f"Got {type(item)} at index {i} instead."
            )
        files_to_run = files
        
    # Ensure that files is a list of Paths and they do exist
    for i, item in enumerate(files_to_run):
        files_to_run[i] = Path(item)
        assert files_to_run[i].exists(
        ), f"File[{i}] {files_to_run[i]} does not exist."

    if isinstance(executable, Path):
        assert executable.exists(), f"Executable {executable} does not exist."
        executable = str(executable)

    # Find the executable
    febio_path = shutil.which(executable)
    
    if febio_path:
        
        if len(files_to_run) == 1:
            filename = files_to_run[0]
            print(f"Starting process for: {filename}")
            process = subprocess.Popen([febio_path, str(filename)])
            print("Waiting for process to complete...")
            process.wait()
            print(f"Completed process for: {filename}")
        else:

            processes = []  # List to store the processes

            # Start processes up to the defined number by the user
            for i, filename in enumerate(files_to_run):
                if i < num_processes:  # Only start up to the defined number of processes
                    print(f"Starting process for: {filename}")
                    processes.append(subprocess.Popen(
                        [executable, str(filename)]))

            # Using tqdm for progress feedback
            for i, process in enumerate(tqdm(processes, 
                                             desc="Waiting for initial processes", 
                                             unit="file")):
                process.wait()
                print(f"\nCompleted process for: {files_to_run[i]}")

            # If there are more files to process, repeat the above process for the remaining files
            for i in tqdm(range(num_processes, len(files_to_run)), 
                          desc="Processing remaining files", unit="file"):
                # Wait for one of the previous processes to finish
                processes[i % num_processes].wait()
                print(
                    f"\nCompleted process for: {files_to_run[i - num_processes]}")
                print(f"Starting process for: {files_to_run[i]}")
                processes[i % num_processes] = subprocess.Popen(
                    [executable, str(files_to_run[i])])

            # Ensure all remaining processes complete
            if len(files_to_run) > num_processes:  # Add this condition
                for i, process in enumerate(tqdm(processes, 
                                                 desc="Waiting for final processes", 
                                                 unit="file")):
                    process.wait()
                    print(
                        f"\nCompleted process for: {files_to_run[num_processes + i]}")
    else:
        raise FileNotFoundError(
            f"Could not find febio executable at {executable}."
        )

    return None
