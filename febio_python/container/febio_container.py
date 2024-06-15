from pathlib import Path
from typing import Union, List, Optional

from febio_python.core import (
    Nodes,
    Elements,
    NodeSet,
    SurfaceSet,
    ElementSet,
    Material,
    NodalLoad,
    PressureLoad,
    LoadCurve,
    BoundaryCondition,
    FixCondition,
    RigidBodyCondition,
    NodalData,
    SurfaceData,
    ElementData,
    XpltMeshPart,
    GenericDomain
)

from febio_python.feb import Feb25, Feb30, Feb40, Feb
from febio_python.xplt import Xplt


class FEBioContainer():
    """Container class for FEBio files (FEB and XPLT files).
    """
    def __init__(self, feb: Union[Feb30, Feb25, str, Path] = None, xplt: Union[Xplt, str, Path] = None, auto_find: bool = True) -> None:
        self.feb: Optional[Union[Feb30, Feb25]] = self._load_feb(feb) if feb else None
        self.xplt: Optional[Xplt] = self._load_xplt(xplt) if xplt else None

        if auto_find:
            self._auto_find_files(feb, xplt)

        # Make sure that we have the correct input
        if self.feb is None and self.xplt is None:
            raise ValueError("No FEB or XPLT file is provided")

        if self.feb is not None and not isinstance(self.feb, (Feb30, Feb25, Feb40)):
            raise ValueError("FEB is not valid. Check input file or input parameters.")
        if self.xplt is not None and not isinstance(self.xplt, Xplt):
            raise ValueError("XPLT is not valid. Check input file or input parameters.")

        # Now, if both files are provided, we must ensure that they are compatible.
        # We will compare the number of nodes and elements in the FEB and XPLT files.
        if self.feb is not None and self.xplt is not None:
            feb_node_count = sum([node.ids.size for node in self.feb.get_nodes()])
            feb_elem_count = sum([elem.ids.size for elem in self.feb.get_elements()])
            xplt_node_count = sum([node.ids.size for node in self.xplt.nodes])
            xplt_elem_count = sum([elem.ids.size for elem in self.xplt.elements])

            if feb_node_count != xplt_node_count:
                raise ValueError("Number of nodes in FEB and XPLT files do not match."
                                 "Please, make sure that the FEB and XPLT files are compatible.")
            if feb_elem_count != xplt_elem_count:
                raise ValueError("Number of elements in FEB and XPLT files do not match."
                                 "Please, make sure that the FEB and XPLT files are compatible.")

    def _load_feb(self, feb: Union[Feb30, Feb25, str, Path]) -> Optional[Union[Feb30, Feb25]]:
        if isinstance(feb, (str, Path)):
            feb_path = Path(feb)
            return Feb(filepath=feb_path)
        return feb

    def _load_xplt(self, xplt: Union[Xplt, str, Path]) -> Optional[Xplt]:
        if isinstance(xplt, (str, Path)):
            return Xplt(filepath=Path(xplt))
        return xplt

    def _auto_find_files(self, feb: Union[Feb30, Feb25, str, Path], xplt: Union[Xplt, str, Path]):
        if not self.xplt and feb and isinstance(feb, (str, Path)):
            xplt_path = Path(feb).with_suffix('.xplt')
            if xplt_path.is_file():
                self.xplt = Xplt(filepath=xplt_path)

        if not self.feb and xplt and isinstance(xplt, (str, Path)):
            feb_path = Path(xplt).with_suffix('.feb')
            if feb_path.is_file():
                self.feb = self._load_feb(feb_path)

    # ========================================================================
    # Properties
    # ========================================================================

    # Main geometry (mesh) properties
    # -------------------------------

    @property
    def nodes(self) -> List[Nodes]:
        if self.feb is not None:
            return self.feb.get_nodes()
        elif self.xplt is not None:
            return self.xplt.nodes
        else:
            raise ValueError("No FEB or XPLT file is provided")

    @property
    def elements(self) -> List[Elements]:
        if self.feb is not None:
            return self.feb.get_elements()
        elif self.xplt is not None:
            return self.xplt.elements
        else:
            raise ValueError("No FEB or XPLT file is provided")

    @property
    def surfaces(self) -> List[Elements]:
        if self.feb is not None:
            return self.feb.get_surface_elements()
        elif self.xplt is not None:
            return self.xplt.surfaces
        else:
            raise ValueError("No FEB or XPLT file is provided")

    @property
    def volumes(self) -> List[Elements]:
        if self.feb is not None:
            return self.feb.get_volume_elements()
        elif self.xplt is not None:
            return self.xplt.volumes
        else:
            raise ValueError("No FEB or XPLT file is provided")

    @property
    def mesh_domains(self) -> List[Union[GenericDomain, XpltMeshPart]]:
        if self.feb is not None:
            return self.feb.get_mesh_domains()
        elif self.xplt is not None:
            return self.xplt.parts
        else:
            raise ValueError("No FEB or XPLT file is provided")

    # Other geometry (mesh) properties
    # --------------------------------

    @property
    def nodesets(self) -> List[NodeSet]:
        if self.feb is not None:
            return self.feb.get_nodesets()
        elif self.xplt is not None:
            return self.xplt.nodesets
        else:
            raise ValueError("No FEB or XPLT file is provided")

    @property
    def surfacesets(self) -> List[SurfaceSet]:
        if self.feb is not None:
            return self.feb.get_surfacesets()
        elif self.xplt is not None:
            raise RuntimeError("XPLT file does not save surface sets. Please provide a FEB file.")
        else:
            raise ValueError("No FEB or XPLT file is provided")

    @property
    def elementsets(self) -> List[ElementSet]:
        if self.feb is not None:
            return self.feb.get_elementsets()
        elif self.xplt is not None:
            raise RuntimeError("XPLT file does not save element sets. Please provide a FEB file.")
        else:
            raise ValueError("No FEB or XPLT file is provided")

    # Material properties
    # -------------------

    @property
    def materials(self) -> List[Material]:
        if self.feb is not None:
            return self.feb.get_materials()
        else:
            raise RuntimeError(
                "Trying to access material data without a FEB file. "
                "Currently only FEB files save material data."
                "To access material data, provide a FEB file.")

    # Loads
    # -------------------

    @property
    def nodal_loads(self) -> List[NodalLoad]:
        if self.feb is not None:
            return self.feb.get_nodal_loads()
        else:
            raise RuntimeError(
                "Trying to access nodal load data without a FEB file. "
                "Currently only FEB files save nodal load data."
                "To access nodal load data, provide a FEB file.")

    @property
    def pressure_loads(self) -> List[PressureLoad]:
        if self.feb is not None:
            return self.feb.get_pressure_loads()
        else:
            raise RuntimeError(
                "Trying to access pressure load data without a FEB file. "
                "Currently only FEB files save pressure load data."
                "To access pressure load data, provide a FEB file.")

    @property
    def load_curves(self) -> List[LoadCurve]:
        if self.feb is not None:
            return self.feb.get_loadcurves()
        else:
            raise RuntimeError(
                "Trying to access load curve data without a FEB file. "
                "Currently only FEB files save load curve data."
                "To access load curve data, provide a FEB file.")

    # Boundary conditions
    # -------------------

    @property
    def boundary_conditions(self) -> List[Union[BoundaryCondition, FixCondition, RigidBodyCondition]]:
        if self.feb is not None:
            return self.feb.get_boundary_conditions()
        else:
            raise RuntimeError(
                "Trying to access boundary condition data without a FEB file. "
                "Currently only FEB files save boundary condition data."
                "To access boundary condition data, provide a FEB file.")

    # Mesh data
    # -------------------

    @property
    def nodal_data(self) -> List[NodalData]:
        if self.feb is not None:
            return self.feb.get_nodal_data()
        elif self.xplt is not None:
            raise RuntimeError("XPLT file does not save nodal data. Please provide a FEB file."
                               "If you are looking for nodal state data, please, use the 'states' property.")
        else:
            raise ValueError("No FEB or XPLT file is provided")

    @property
    def surface_data(self) -> List[SurfaceData]:
        if self.feb is not None:
            return self.feb.get_surface_data()
        elif self.xplt is not None:
            raise RuntimeError("XPLT file does not save surface data. Please provide a FEB file."
                               "If you are looking for surface state data, please, use the 'states' property.")
        else:
            raise ValueError("No FEB or XPLT file is provided")

    @property
    def element_data(self) -> List[ElementData]:
        if self.feb is not None:
            return self.feb.get_element_data()
        elif self.xplt is not None:
            raise RuntimeError("XPLT file does not save element data. Please provide a FEB file."
                               "If you are looking for element state data, please, use the 'states' property.")
        else:
            raise ValueError("No FEB or XPLT file is provided")

    # States (results)
    # -------------------

    @property
    def states(self) -> None:
        if self.xplt is not None:
            return self.xplt.states
        else:
            raise RuntimeError(
                "Trying to access state data without a XPLT file. "
                "Currently XPLT files save state data."
                "To access state data, provide a XPLT file.")

    @property
    def node_states(self) -> List[Nodes]:
        if self.xplt is not None:
            return self.xplt.node_states
        else:
            raise RuntimeError(
                "Trying to access node state data without a XPLT file. "
                "Currently XPLT files save state data."
                "To access state data, provide a XPLT file.")

    @property
    def element_states(self) -> List[Elements]:
        if self.xplt is not None:
            return self.xplt.element_states
        else:
            raise RuntimeError(
                "Trying to access element state data without a XPLT file. "
                "Currently XPLT files save state data."
                "To access state data, provide a XPLT file.")

    @property
    def surface_states(self) -> List[Elements]:
        if self.xplt is not None:
            return self.xplt.surface_states
        else:
            raise RuntimeError(
                "Trying to access surface state data without a XPLT file. "
                "Currently XPLT files save state data."
                "To access state data, provide a XPLT file.")
