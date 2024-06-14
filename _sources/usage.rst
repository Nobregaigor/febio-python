Usage
=====

Then, import the package in your Python code:

.. code-block:: python

   import febio

Reading a FEBio input file (`.feb`)
------------------------------------

To read a FEBio input file, use the `febio.Feb` function. 
This function returns a `Feb` object that can be used to access the contents of the FEBio input file.
It automatically parses the input file and stores the contents in a structured format using xml.etree.ElementTree.
Additionally, it automatically identifies the FEBio version used in the input file.

.. code-block:: python

   from febio_python import Feb
   feb = Feb('path/to/file.feb')


The `Feb` object has a number of attributes and methods that can be used to manipulate the contents of the FEBio input file.
For example, to access the root element of the input file, use the `root` attribute:

.. code-block:: python

   root = feb.root

The `root` attribute is an `xml.etree.ElementTree.Element` object that can be used to access the contents of the input file.
Here you can direcly modify any part of the tree using the ElementTree API.
The `Feb` object also has a number of convenience attributes that can be used to access specific parts of the input file,
these are attributes are the lead tags for the Feb file. For example, to access the `Material` section of the input file, use the `materials` attribute:

.. code-block:: python

   materials = feb.materials

The `materials` attribute is a list of `xml.etree.ElementTree.Element` objects that represent the `Material` section of the input file.
The `Feb` object also has a number of convenience methods that can be used to manipulate the contents of the input file.
For example we can add Nodes to the input file using the `add_node` method:

.. code-block:: python

   from febio_python.core import Nodes

   new_nodes = Nodes(name="NewNodes", coordinates=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
   feb.add_node([new_nodes])

The `add_node` method takes a list of `Nodes` objects and adds them to the input file. For full documentation on the `Feb` object, see the API documentation.

Writing a FEBio input file (`.feb`)
------------------------------------

To write a FEBio input file, use the `write` method of the `Feb` object. It takes a single argument, the path to the output file:

.. code-block:: python

   feb.write('path/to/output.feb')

The `write` method writes the contents of the `Feb` object to the specified output file in FEBio format. 

Running a FEBio simulation
---------------------------

To run a FEBio simulation, use the `run` function. You can run a simulation or multiple simulations depending on the input argument.
If the input is a filepath, it will run the simulation using the specified input file. If the input is a directory, it will run all the simulations in the directory.

.. code-block:: python

   from febio_python import run

   run('path/to/file.feb')

Reading the results of a FEBio simulation (`.xplt`)
---------------------------------------------------

To read the results of a FEBio simulation, use the `febio.Xplt` function.
This function returns an `Xplt` object that can be used to access the contents of the FEBio results file.

.. code-block:: python

   from febio_python import Xplt
   xplt = Xplt('path/to/file.xplt')

The `Xplt` object has a number of attributes that can be used to access the contents of the FEBio results file.
You can retrieve the nodes, elements and surfaces from the simulation. For example, to access the nodes of the simulation, use the `nodes` attribute:

.. code-block:: python

   nodes = xplt.nodes

Or you can access the states (results) of the simulation. Each state is either stored as node, element or surface data. 
For example, to access the node states of the simulation, use the `states` attribute:

.. code-block:: python

   states = xplt.states
   node_states = states.node

This will return a list of `StateData` for the nodes. Each `StateData` object has a `name` and `data` attribute that contains the information of the state.
For full documentation on the `Xplt` object, see the API documentation.


The FEBio Container
-------------------

For convenience, we have a `Container` class that can be used to store both `Feb` and `Xplt` objects. Allowing to access data from both the input 
and output files in a single object. For instance, you may want to access the input conditions (boundary and loads) while accessing the results of the simulation.

.. code-block:: python

   from febio_python import FEBioContainer

   container = FEBioContainer(feb='path/to/file.feb', xplt='path/to/file.xplt') # or feb=feb, xplt=xplt if you already have the objects.


Converting to `pyvista`
-----------------------

We have some utility functions to convert the Febio-Python objects to `pyvista` objects. This is useful to visualize the results of the simulation
or to provide a more interactive way to access the data, such as manipulating the mesh of the input file or plotting the results.

.. code-block:: python

   from febio_python.utils.pyvista_utils import febio_to_pyvista

   listed_grid = febio_to_pyvista(container)

This will return a `pyvista.UnstructuredGrid` object for each state in the simulation. For consistency, it will always return a list of `pyvista.UnstructuredGrid` objects.
Thus, if only one state is present, it will return a list with a single `pyvista.UnstructuredGrid` object. 
Additionally, if only the input (Feb) is provide, it will also return a list with a single `pyvista.UnstructuredGrid` object. 
To plot the mesh you can simply use:

.. code-block:: python

   listed_grid[0].plot()

For full documentation on the `FEBioContainer` object, see the API documentation.


Converting to `vtk`
-------------------

For convenience, we have an utility function to convert the Febio-Python objects to `vtk` objects. This is useful to visualize the results of the simulation
and post-process the data using the `vtk` software. This function will use `febio_to_pyvista` and save each `pyvista.UnstructuredGrid` object to a `.vtk` file.

.. code-block:: python

   from febio_python.utils.vtk_utils import febio_to_vtk

   febio_to_vtk(container)

This will save the `.vtk` files in the same directory as the input file. 
For full documentation on the `febio_to_vtk` function, see the API documentation.
