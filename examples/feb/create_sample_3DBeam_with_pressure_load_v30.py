from pathlib import Path
import pyvista as pv
import numpy as np
from febio_python import FEBioContainer, Feb, run
from febio_python.utils.pyvista_utils import febio_to_pyvista
from febio_python.core import (
    Nodes,
    Elements,
    Surfaces,
    Material,
    NodeSet,
    FixCondition,
    SurfaceLoad,
    LoadCurve,
    SolidDomain
)

# ==============================================================================
# Setup Paths

# Get the current working directory
CURR_DIR = Path.cwd()
SAMPLES_DIR = CURR_DIR.parent / "samples"
HEXBEAM_VTK_FILEPATH = SAMPLES_DIR / "hexbeam.vtk"

# ==============================================================================

if __name__ == "__main__":
    
    # ==============================================================================
    # Create (load) mesh
    
    # --------------------------------------------------------------------
    # NOTE:
    
    # In this examples, we are simply interested in manipulating the FEB file
    # and not the mesh itself, therefore we will be used pre-defined mesh.
    # If you wish to create the mesh using python, we recommend using the pyvista
    # or the gmsh and pygmsh libraries. 
    
    # --------------------------------------------------------------------
    
    # Load the mesh
    grid = pv.read(HEXBEAM_VTK_FILEPATH)
    print(f"Number of nodes: {grid.n_points}")
    print(f"Number of cells: {grid.n_cells}")
    
    # ==============================================================================
    
    # Create a FEB object
    feb = Feb(version=3.0)
    
    # --------------------------------------------------------------------
    # Setup basic configurations:
    
    feb.setup_module(module_type="solid") # default values
    feb.setup_globals(T=0, R=0, Fc=0) # default values
    feb.setup_controls(analysis="static") # here, you can change basic settings. See docs for more info.
    feb.setup_output(variables=["displacement", "Lagrange strain", "stress"]) # default values
    
    # --------------------------------------------------------------------
    # Add materials
    
    # Define the material properties
    material = Material(
        id=1,
        type="isotropic elastic",
        name="Material 1",
        parameters=dict(
            E=1e3,
            v=0.3,
            density=1,
        )
    )
    # Add the material to the FEB object
    feb.add_materials([material])
    
    # --------------------------------------------------------------------
    # Add mesh
    
    # Create nodes and elements objects
    nodes = Nodes(name="Part1-Nodes", coordinates=grid.points)
    elements = Elements(name="Part1", 
                        type="HEXAHEDRON", 
                        mat=1,
                        connectivity=grid.cells_dict[pv.CellType.HEXAHEDRON])
    # Add the nodes and elements to the FEB object
    feb.add_nodes([nodes])
    feb.add_elements([elements])
    
    # --------------------------------------------------------------------
    # Add surface (will be used to apply pressure load)
    
    # first, we need to create a surface mesh.
    surf_grid = grid.extract_surface()  # will extract the surface as a new grid
    surf_grid = surf_grid.cast_to_unstructured_grid()  # convert the grid to unstructured grid
    surf_cells: np.ndarray = surf_grid.cells_dict[pv.CellType.QUAD]  # get the surface cells
    original_mesh_node_indices = surf_grid["vtkOriginalPointIds"]  # get the original node indices
    # map indices from original mesh to the new mesh
    surf_cells_mapped = np.zeros_like(surf_cells)
    for i, cell in enumerate(surf_cells):
        surf_cells_mapped[i] = original_mesh_node_indices[cell]
    # Select the surface to the right of the beam (negative x direction)
    surf_centers = surf_grid.cell_centers().points
    max_x = np.max(surf_centers[:, 0])
    surf_cells_mapped = surf_cells_mapped[abs(surf_centers[:, 0] - max_x) < 1e-6]    
    # Create the surface object
    surfaces = Surfaces(
        name="SurfaceLoad_X",
        type="QUAD",
        connectivity=surf_cells_mapped)
    # Add the surface to the FEB object
    feb.add_surface_elements([surfaces])
    
    # --------------------------------------------------------------------
    # Add Domain
    
    domain = SolidDomain(
        name="Part1",  # must match the element set name
        mat="Material 1",  # must match the material name
    )
    feb.add_mesh_domains([domain])

    # --------------------------------------------------------------------
    # Define the boundary conditions
    # We will fix the "left" side of the beam, e.g. negative x direction
    # Get the nodes on the left side of the beam
    min_x = np.min(grid.points[:, 0])
    left_nodes = np.where(abs(grid.points[:, 0] - min_x) < 1e-6)[0]
    # Create the node set
    node_set = NodeSet(name="FixedNodes", ids=left_nodes)
    # Add the node set to the FEB object
    feb.add_node_sets([node_set])
    
    # create the fixed boundary condition
    fixed = FixCondition(
        dof="x,y,z",
        node_set="FixedNodes",
        name="Fixed")
    # Add the fixed boundary condition to the FEB object
    feb.add_boundary_conditions([fixed])
    
    
    # --------------------------------------------------------------------
    # Define the pressure load
    
    load = SurfaceLoad(
        surface="SurfaceLoad_X",  # this is the surface name (must be in the FEB object)
        load_curve=1,  # this is the load curve ID
        name="SurfaceLoad",  # optional name
        scale=-500, # scale factor
        linear=True,  # linear pressure load
        symmetric_stiffness=True)  # symmetric stiffness matrix
    
    # Add the pressure load to the FEB object
    feb.add_surface_loads([load])
    
    # --------------------------------------------------------------------
    # Define the load curve
    
    lc = LoadCurve(
        id=1,
        interpolate_type="linear",
        data=np.array([[0, 0], [1, 1]]))
    # Add the load curve to the FEB object
    feb.add_load_curves([lc])
    
    # --------------------------------------------------------------------
    # Write the FEB file
    print(feb)

    output_feb_file = SAMPLES_DIR / "hexbeam_with_pressure_load_v30.feb"
    feb.write(output_feb_file)
    
    # ==============================================================================
    # Visualize the Feb
    
    container = FEBioContainer(feb=feb, auto_find=False)
    feb_grid = febio_to_pyvista(container)[0]
    feb_grid.plot(show_edges=True)
    
    # ==============================================================================
    
    # Run the FEBio analysis
    run(output_feb_file)
    
    # # ==============================================================================
    
    # Load the results
    results_file = output_feb_file.with_suffix(".xplt")
    
    # Create a FEBioContainer object
    container = FEBioContainer(feb=feb, xplt=results_file, auto_find=False)
    
    # print(len(container.elements))
    print(container.xplt)
        
    # Convert the FEBioContainer to PyVista
    all_grids = febio_to_pyvista(container)
    
    # plot last time step
    grid = all_grids[-1]
    grid.plot(scalars="stress", show_edges=True, show_axes=True, cpos="xy")
