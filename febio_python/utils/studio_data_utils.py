from pathlib import Path
from typing import Union, Tuple
import numpy as np


def parse_vector_data(file_path: Union[str, Path]) -> Tuple[np.ndarray[int], np.ndarray[float]]:
    """
    Parses vector data from a data file exported from FEBio Studio.

    Args:
        file_path (Union[str, Path]): Path to the file containing the vector data. Should be a CSV file.

    Returns:
        Tuple[np.ndarray[int], np.ndarray[float]]: A tuple containing:
            - ids (np.ndarray[int]): The parsed IDs.
            - reshaped_vectors (np.ndarray[float]): The reshaped vectors in the format (num_timesteps, num_vectors, 3).

    Raises:
        ValueError: If the number of columns in the data file is not divisible by 3.
    """
    data = np.loadtxt(file_path, delimiter=",")
    ids = data[:, 0].astype(int)  # first column contains IDs
    vectors = data[:, 1:].astype(float)  # remaining columns contain vector data
    if not vectors.shape[1] % 3 == 0:
        raise ValueError("Number of columns in the data file is not divisible by 3.")
    num_timesteps = vectors.shape[1] // 3
    num_vectors = vectors.shape[0]
    reshaped_vectors = vectors.reshape(num_timesteps, num_vectors, 3).transpose(1, 0, 2)
    return ids, reshaped_vectors


def parse_tensor_data(file_path: Union[str, Path]) -> Tuple[np.ndarray[int], np.ndarray[float]]:
    """
    Parses tensor data from a data file exported from FEBio Studio.

    Args:
        file_path (Union[str, Path]): Path to the file containing the tensor data. Should be a CSV file.

    Returns:
        Tuple[np.ndarray[int], np.ndarray[float]]: A tuple containing:
            - ids (np.ndarray[int]): The parsed IDs.
            - reshaped_tensor_data (np.ndarray[float]): The reshaped tensor data in the format 
              (num_timesteps, num_elements, 6 (symmetric) or 9).

    Raises:
        ValueError: If the number of columns in the data file is not divisible by 6 or 9.
    """
    data = np.loadtxt(file_path, delimiter=",")
    ids = data[:, 0].astype(int)  # first column contains element IDs
    tensor_data = data[:, 1:].astype(float)  # remaining columns contain tensor data
    
    num_elements = tensor_data.shape[0]
    print(f"Found total of {num_elements} elements.")
    print(f"Total tensor data shape: {tensor_data.shape}")
    is_divisible_by_6 = tensor_data.shape[1] % 6 == 0
    if_divisible_by_9 = tensor_data.shape[1] % 9 == 0
    if not is_divisible_by_6 and not if_divisible_by_9:
        raise ValueError("Number of columns in the data file is not divisible by 6 or 9.")
    if is_divisible_by_6:
        num_timesteps = tensor_data.shape[1] // 6
        last_col = 6
    else:
        num_timesteps = tensor_data.shape[1] // 9
        last_col = 9
    
    # Reshape to (num_timesteps, num_elements, last_col) and transpose for correct format
    reshaped_tensor_data = tensor_data.reshape(num_elements, num_timesteps, last_col).transpose(1, 0, 2)
    
    return ids, reshaped_tensor_data


def parse_data(file_path: Union[str, Path]) -> Tuple[np.ndarray[int], np.ndarray[float]]:
    """Parses data from a file exported from FEBio Studio. The function will automatically determine
    whether the data is vector or tensor data based on the number of columns in the file.
    
    Args:
        file_path (Union[str, Path]): Path to the file containing the tensor data. Should be a CSV file.

    Returns:
        Tuple[np.ndarray[int], np.ndarray[float]]: A tuple containing:
            - ids (np.ndarray[int]): The parsed IDs.
            - reshaped_data (np.ndarray[float]): The reshaped data in the format 
              (num_timesteps, num_elements, data_size).
    """
    try:
        return parse_tensor_data(file_path)
    except ValueError:
        return parse_vector_data(file_path)
