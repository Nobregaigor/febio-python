from pathlib import Path
from tqdm import tqdm
from febio_python import FEBioContainer
from .pyvista_utils import febio_to_pyvista

def febio_to_vtk(data:FEBioContainer, output_directory=None, stem=None, **febio_to_pyvista_kwargs):
    """Convert FEBioContainer to VTK format. This function will convert the FEBioContainer to PyVista 
    and save the resulting grids to VTK files. For more information on the conversion process, see
    `febio_to_pyvista` function.

    Args:
        data (FEBioContainer): FEBioContainer object
        output_directory (str, optional): Output directory. Defaults to None.
    """
    # Determine output directory and stem
    datapath = None
    data_dir = Path.cwd()
    data_stem = stem or "vtk_output"
    if data.feb is not None and data.feb.path_to_file is not None:
        datapath = Path(data.feb.path_to_file)
    elif data.xplt is not None and data.xplt.filepath is not None:
        datapath = Path(data.xplt.filepath)
    if datapath:
        data_dir = datapath.parent
        data_stem = datapath.stem

    # Set default output directory
    default_dir = data_dir / data_stem
    
    # Set output directory
    if output_directory is None:
        output_directory = default_dir
    
    # Create output directory if it does not exist
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    
    try:
        all_grids = febio_to_pyvista(data, **febio_to_pyvista_kwargs)
    except Exception as e:
        raise RuntimeError(f"Error converting FEBioContainer to PyVista: {e}")

    if len(all_grids) == 1:
        filename = f"{data_stem}.vtk"
        try:
            all_grids[0].save(output_directory / filename)
        except Exception as e:
            print(f"Error saving {filename}: {e}")
    else:
        pbar = tqdm(all_grids, desc="Saving VTK files")
        for grid in pbar:
            timestep = grid.field_data["timestep"][0]
            filename = f"{data_stem}_timestep_{timestep:.4f}.vtk"
            try:
                grid.save(output_directory / filename)
            except Exception as e:
                print(f"Error saving {filename}: {e}")
