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
    DiscreteSet,
    DiscreteMaterial,
    RigidBodyCondition
)

# ==============================================================================
# Setup Paths

# Get the current working directory
CURR_DIR = Path.cwd()
SAMPLES_DIR = CURR_DIR.parent / "samples"
HEXELLIPSOID_VTK_FILEPATH = SAMPLES_DIR / "hex_half_ellipsoid.vtk"

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
    grid = pv.read(HEXELLIPSOID_VTK_FILEPATH)
    print(f"Number of nodes: {grid.n_points}")
    print(f"Number of cells: {grid.n_cells}")
    
    # Create a FEB object
    feb = Feb(version=2.5)
    
    # --------------------------------------------------------------------
    # Setup basic configurations:
    
    feb.setup_module(module_type="solid") # default values
    feb.setup_globals(T=0, R=0, Fc=0) # default values
    feb.setup_controls(analysis_type="static") # here, you can change basic settings. See docs for more info.
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
    elements = Elements(name="Part1-Elements", 
                        type="HEXAHEDRON", 
                        mat=1,
                        connectivity=grid.cells_dict[pv.CellType.HEXAHEDRON])
    # Add the nodes and elements to the FEB object
    feb.add_nodes([nodes])
    feb.add_elements([elements])
    
    # --------------------------------------------------------------------
    # Add surface (will be used to apply pressure load)
    
    # first, we need to create a surface mesh.
    surf_grid: pv.PolyData = grid.extract_surface()  # will extract the surface as a new grid
    # extract normals (will be used later)
    surf_normals = surf_grid.compute_normals(cell_normals=True, point_normals=False)
    surf_grid = surf_grid.cast_to_unstructured_grid()  # convert the grid to unstructured grid
    surf_cells: np.ndarray = surf_grid.cells_dict[pv.CellType.QUAD]  # get the surface cells
    original_mesh_node_indices = surf_grid["vtkOriginalPointIds"]  # get the original node indices
    # map indices from original mesh to the new mesh
    surf_cells_mapped = np.zeros_like(surf_cells)
    for i, cell in enumerate(surf_cells):
        surf_cells_mapped[i] = original_mesh_node_indices[cell]
    # Select the surfaces that have normals pointing to the centroid of the ellipsoid
    # (i.e. inner surfaces)
    surf_centers = surf_grid.cell_centers().points
    surf_normals = surf_normals["Normals"]
    # get the centroid of the ellipsoid
    centroid = np.mean(grid.points, axis=0)
    # Compute vectors from surface centers to the centroid
    vecs = surf_centers - centroid
    # Compute the dot product between the vectors and the normals
    dots = np.sum(vecs * surf_normals, axis=1)
    # Compute the angle between the vectors and the normals
    angles = np.arccos(dots / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(surf_normals, axis=1)))
    # Select the surfaces that have an angle more than 90 degrees
    mask = angles > np.pi / 2
    # create a visualization of the selected surfaces
    vis_selected_surfaces = np.zeros(surf_centers.shape[0], dtype=int)
    vis_selected_surfaces[mask] = 1
    surf_grid["SelectedSurfaces"] = vis_selected_surfaces
    surf_grid.plot(scalars="SelectedSurfaces", show_edges=True)    
    # apply mask
    surf_cells_mapped = surf_cells_mapped[mask]
    # Create the surface object
    surfaces = Surfaces(
        name="SurfaceLoadSurface",
        type="QUAD",
        connectivity=surf_cells_mapped)
    # Add the surface to the FEB object
    feb.add_surfaces([surfaces])
    
    # --------------------------------------------------------------------
    # Define the pressure load
    
    load = SurfaceLoad(
        surface="SurfaceLoadSurface",  # this is the surface name (must be in the FEB object)
        load_curve=1,  # this is the load curve ID
        name="SurfaceLoad",  # optional name
        scale=100, # scale factor
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
    # Define the boundary conditions
    # We will fix the "top" nodes (z = max_z)
    max_z = np.max(grid.points[:, 2])
    left_nodes = np.where(abs(grid.points[:, 2] - max_z) < 1e-6)[0]
    # Create the node set
    node_set = NodeSet(name="FixedNodes", ids=left_nodes)
    # Add the node set to the FEB object
    feb.add_node_sets([node_set])
    
    # create the fixed boundary condition
    fixed = FixCondition(
        dof="z",  # fix only the z direction
        node_set="FixedNodes",
        name="Fixed")
    # Add the fixed boundary condition to the FEB object
    feb.add_boundary_conditions([fixed])
    
    # --------------------------------------------------------------------
    # Create "plane" to apply discrete rigid body condition
    
    # Extract the top surface of the ellipsoid
    top_nodes = np.where(abs(surf_grid.points[:, 2] - max_z) < 1e-6)[0]
    top_grid = surf_grid.extract_points(top_nodes, adjacent_cells=False)
    # Move grid a little bit upwards
    top_grid.points[:, 2] += 0.05
    # Visuzalize the top grid and the ellipsoid
    plotter = pv.Plotter()
    plotter.add_mesh(top_grid, color="yellow", show_edges=True)
    plotter.add_mesh(grid, color="beige", show_edges=True)
    plotter.show()
    
    # now, let's compute the discrete set.
    # This set will connect the top surface of the ellipsoid to a plane
    # that is located above the ellipsoid.
    
    # select the top nodes of the ellipsoid
    top_nodes_indices = np.where(abs(grid.points[:, 2] - max_z) < 1e-6)[0]
    # for each node, compute the respective shortest node (euclidean distance) 
    # on the top grid. We can use the cdist function from scipy.spatial.distance
    from scipy.spatial.distance import cdist
    # compute the distance matrix
    dist_matrix = cdist(grid.points[top_nodes_indices], top_grid.points)
    # get the indices of the minimum distances for each node
    min_indices = np.argmin(dist_matrix, axis=1)
    # create the discrete set, which is a numpy array of shape (n, 2)
    discrete_set = np.column_stack((top_nodes_indices, min_indices))
    
    # Visualize the discrete set
    plotter = pv.Plotter()
    plotter.add_mesh(grid, color="beige", show_edges=True)
    plotter.add_mesh(top_grid, color="yellow", show_edges=True)
    # plotter.add_lines(grid.points[discrete_set], color="red")
    src_points = grid.points[discrete_set[:, 0]]
    dst_points = top_grid.points[discrete_set[:, 1]]
    lines = np.stack((src_points, dst_points), axis=1).reshape(-1, 3)
    plotter.add_lines(lines, color="red")
    plotter.show()
    
    # Create the discrete set object
    discrete_set_obj = DiscreteSet(
        name="DiscreteSet",
        src=discrete_set[:, 0],
        dst=discrete_set[:, 1] + grid.n_points,  # add the number of nodes in the top grid
        dmat=1)
    
    # Create the discrete material object
    discrete_material = DiscreteMaterial(
        id=1,
        type="linear spring",
        name="DiscreteMaterial",
        parameters=dict(
            E=1e6
        )
    )
    
    # We also need to make sure we have the nodes and elements of the top grid
    # in the FEB object. So, let's add them.
    top_nodes = Nodes(name="TopNodes", coordinates=top_grid.points)
    top_elements = Elements(name="TopElements", 
                            type="QUAD", 
                            mat=2,  # will be a rigid body material
                            connectivity=top_grid.cells_dict[pv.CellType.QUAD] + grid.n_points)
    
    # Add the nodes, elements, discrete set, and discrete material to the FEB object
    feb.add_nodes([top_nodes])
    feb.add_elements([top_elements])
    feb.add_discrete_sets([discrete_set_obj])
    feb.add_discrete_materials([discrete_material])
    
    # --------------------------------------------------------------------
    # Create the rigid body condition
    rigid_body = RigidBodyCondition(
        dof="x,y,z,Rx,Ry,Rz",
        name="FixedPlane",
        material=2)  # rigid body material ID
    feb.add_boundary_conditions([rigid_body])
    
    # --------------------------------------------------------------------
    # Create the rigid body material
    rigid_material = Material(
        id=2,
        type="rigid body",
        name="RigidBodyMaterial",
        parameters=dict(
            density=1,
            center_of_mass=", ".join(map(str, centroid)),
        )
    )
    feb.add_materials([rigid_material])
    
    # --------------------------------------------------------------------
    # Add a virtual node at the centroid of the ellipsoid
    vir_node = Nodes(
        name="VirtualNode",
        coordinates=np.array([0,0,0]))
    # Add a discrete set between the top nodes and the virtual node
    src_ids = top_nodes_indices
    dst_ids = np.full((top_nodes_indices.shape[0],), grid.n_points + top_grid.n_points)
    vir_discrete_set = DiscreteSet(
        name="VirtualDiscreteSet",
        src=src_ids,  # virtual node
        dst=dst_ids,  # centroid of the ellipsoid
        dmat=1)
    # Add a nodeset for the virtual node
    vir_Nodeset = NodeSet(name="VirtualNode", ids=[grid.n_points + top_grid.n_points])
    # add fixed condition to the virtual node (fix all dofs)
    vir_fixed = FixCondition(
        dof="x,y,z,Rx,Ry,Rz",
        node_set="VirtualNode",
        name="VirtualNodeFixed")
    # Add the virtual node, discrete set, and node set to the FEB object
    feb.add_nodes([vir_node])
    feb.add_discrete_sets([vir_discrete_set])
    feb.add_node_sets([vir_Nodeset])
    feb.add_boundary_conditions([vir_fixed])
    
    # --------------------------------------------------------------------

    # Write the FEB file
    output_file = SAMPLES_DIR / "hex_half_ellipsoid_with_discrete_set.feb"
    
    feb.write(output_file)
    
    
    
    