from pathlib import Path
from typing import Union, List
from febio_python.feb import Feb
from febio_python.core import (
    SURFACE_EL_TYPE,
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
    # FEBioElementType,
)


class FEBioContainer():
    def __init__(self, feb: Union[Feb, str, Path]=None, xplt: Union[str, Path]=None) -> None:
        
        self.feb: None | Feb = feb
        if isinstance(feb, str) or isinstance(feb, Path):
            self.feb: Feb = Feb(filepath=feb)            
        
        self.xplt: None | str = xplt
        if isinstance(xplt, str) or isinstance(xplt, Path):
            self.xplt: str = xplt
        
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
            raise NotImplementedError("XPLT file is not yet supported")
        else:
            raise ValueError("No FEB or XPLT file is provided")
    
    @property
    def elements(self) -> List[Elements]:
        if self.feb is not None:
            return self.feb.get_elements()
        elif self.xplt is not None:
            raise NotImplementedError("XPLT file is not yet supported")
        else:
            raise ValueError("No FEB or XPLT file is provided")
    
    @property
    def surfaces(self) -> List[Elements]:
        if self.feb is not None:
            return self.feb.get_surfaces()
        elif self.xplt is not None:
            raise NotImplementedError("XPLT file is not yet supported")
        else:
            raise ValueError("No FEB or XPLT file is provided")
    
    @property
    def volumes(self) -> List[Elements]:
        if self.feb is not None:
            return self.feb.get_volumes()
        elif self.xplt is not None:
            raise NotImplementedError("XPLT file is not yet supported")
        else:
            raise ValueError("No FEB or XPLT file is provided")
    
    # Other geometry (mesh) properties
    # --------------------------------
    
    @property
    def nodesets(self) -> List[NodeSet]:
        if self.feb is not None:
            return self.feb.get_nodesets()
        elif self.xplt is not None:
            raise NotImplementedError("XPLT file is not yet supported")
        else:
            raise ValueError("No FEB or XPLT file is provided")
    
    @property
    def surfacesets(self) -> List[SurfaceSet]:
        if self.feb is not None:
            return self.feb.get_surfacesets()
        elif self.xplt is not None:
            raise NotImplementedError("XPLT file is not yet supported")
        else:
            raise ValueError("No FEB or XPLT file is provided")
    
    @property
    def elements(self) -> List[ElementSet]:
        if self.feb is not None:
            return self.feb.get_elements()
        elif self.xplt is not None:
            raise NotImplementedError("XPLT file is not yet supported")
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
                "To access material data, provide a FEB file."
                )
    
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
                "To access nodal load data, provide a FEB file."
                )
    
    @property
    def pressure_loads(self) -> List[PressureLoad]:
        if self.feb is not None:
            return self.feb.get_pressure_loads()
        else:
            raise RuntimeError(
                "Trying to access pressure load data without a FEB file. "
                "Currently only FEB files save pressure load data."
                "To access pressure load data, provide a FEB file."
                )
    
    @property
    def load_curves(self) -> List[LoadCurve]:
        if self.feb is not None:
            return self.feb.get_load_curves()
        else:
            raise RuntimeError(
                "Trying to access load curve data without a FEB file. "
                "Currently only FEB files save load curve data."
                "To access load curve data, provide a FEB file."
                )
    
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
                "To access boundary condition data, provide a FEB file."
                )
    
    # Mesh data
    # -------------------
    
    @property
    def nodal_data(self) -> List[NodalData]:
        if self.feb is not None:
            return self.feb.get_nodal_data()
        elif self.xplt is not None:
            raise NotImplementedError("XPLT file is not yet supported")
        else:
            raise ValueError("No FEB or XPLT file is provided")
    
    @property
    def surface_data(self) -> List[SurfaceData]:
        if self.feb is not None:
            return self.feb.get_surface_data()
        elif self.xplt is not None:
            raise NotImplementedError("XPLT file is not yet supported")
        else:
            raise ValueError("No FEB or XPLT file is provided")
    
    @property
    def element_data(self) -> List[ElementData]:
        if self.feb is not None:
            return self.feb.get_element_data()
        elif self.xplt is not None:
            raise NotImplementedError("XPLT file is not yet supported")
        else:
            raise ValueError("No FEB or XPLT file is provided")
    
    # States (results)
    # -------------------
    
    @property
    def states(self) -> None:
        if self.xplt is not None:
            raise NotImplementedError("XPLT file is not yet supported")
        else:
            raise RuntimeError(
                "Trying to access state data without a XPLT file. "
                "Currently XPLT files save state data."
                "To access state data, provide a XPLT file."
                )
