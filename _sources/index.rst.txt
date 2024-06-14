.. Febio-Python documentation master file, created by
   sphinx-quickstart on Fri Jun 14 08:57:58 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FEBio-Python's documentation!
========================================

.. note::

   This project is under active development.


Introduction
------------

Welcome to **FEBio-Python**, an open-source toolkit designed to bridge the gap between the FEBio software suite and Python. 
This project extends the functionality of FEBio by allowing users to interact with FEBio input and output files through Python scripts. 
With FEBio-Python, users can create and modify mesh structures, assign loads and boundary conditions, run batch simulations, 
and much more, all within an intuitive Python environment. 

FEBio-Python facilitates the rapid configuration and execution of simulations, making it an essential tool for performing extensive 
studies involving complex biomechanical models. With features that support the customization of simulation parameters and the 
extraction of valuable data, this tool helps bridge the gap between complex FEA operations and practical, actionable insights.

FEBio-Python simplifies the manipulation of `.feb` files, offering capabilities to parse these files into structured Python objects, 
manipulate the XML tree directly, and write back modifications with ease. 
Moreover, the toolkit facilitates the reading of FEBio simulation results in `.xplt` format, enabling the extraction of simulation 
states and the conversion of data for visualization and further analysis.

Whether you are an academic researcher, a professional in biomechanics, or just someone interested in FEA, 
FEBio-Python provides the tools to enhance your workflow and achieve more precise results.

Key Features of FEBio-Python
----------------------------

- **Comprehensive API for `.feb` File Manipulation:** Easily read, modify, and write `.feb` files to customize simulations as per specific requirements.
- **Advanced Results Processing and Analysis:** Import FEBio simulation results from `.xplt` files directly into Python, facilitating comprehensive post-processing.
   Users can export data to `pandas` for structured data analysis, apply `scipy` for scientific computing, and employ other Python libraries to perform detailed
   evaluations and optimizations of simulation outcomes.
- **Simulation Management:** Run simulations directly from scripts, managing single or multiple simulation runs efficiently.
- **Results Analysis and Visualization:** Leverage Python's extensive ecosystem to analyze and visualize results; includes integration with `pyvista` for advanced 3D visualization and `vtk` for exporting data.
- **Extensible and Customizable:** Perfect for developing custom workflows in biomechanics research, including parameter studies and optimization loops.

About FEBio
-----------

[FEBio](https://febio.org/) (Finite Elements for Biomechanics) is a specialized software for nonlinear finite element analysis in biomechanics and biophysics. It is tailored to manage the unique properties of biological materials and interactions. Learn more about FEBio by visiting their [official website](https://febio.org/) or exploring their [GitHub repository](https://github.com/febiosoftware/FEBio.git).

Getting Started
===============

To get started with FEBio-Python, please visit our [documentation](https://nobregaigor.github.io/febio-python/index.html) for detailed instructions:
- [Installation Guide](https://nobregaigor.github.io/febio-python/installation.html)
- [Usage Instructions](https://nobregaigor.github.io/febio-python/usage.html)

We have some basic examples to help you get started:
- [Project Examples](https://github.com/Nobregaigor/febio-python/tree/main/examples)

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: API:

   api/index

Contributing
============

We welcome contributions to Febio-Python. To contribute, follow these steps:
- Fork the repository on GitHub.
- Create a new branch from the `main` branch.
- Make your changes.
- Add tests for your changes.
- Run the tests using `pytest`.
- Commit your changes.
- Push your changes to your fork.
- Create a pull request on GitHub.

We will review your changes and merge them into the repository.

We are an open-source project and welcome contributions from the community. 
Right now, we are focused on specific areas of FEBio, and there may be other areas that 
are missing or need improvement. If you have any ideas or suggestions, please let us know.
Feel free to make improvements or add new features to the project. All contributions are welcome!

If you have any questions or need help, please feel free to reach out to us.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
