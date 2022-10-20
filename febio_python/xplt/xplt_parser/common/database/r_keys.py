from enum import Enum

class R_KEYS(Enum):
    SPEC_VERSION = "SPEC_VERSION"
    
    N_NODES = "N_NODES"     # -> number of nodes
    
    N_DOMS = "N_DOMS"       # -> number of element domains
    N_ELEMS = "N_ELEMS"     # -> number of elements
    
    NODES = "NODES"         # nodes (coords)    
    ELEMENTS = "ELEMENTS"   # elements
    
    FACETS = "FACETS"       # facets 
    FACETS_BY_NAME = "FACETS_BY_NAME" #
    FACETS_BY_ID = "FACETS_BY_ID" #
    
    NODESETS = "NODESETS"       # NODESETs 
    NODESETS_BY_NAME = "NODESETS_BY_NAME" #
    NODESETS_BY_ID = "NODESETS_BY_ID" #
    
    N_STATES = "N_STATES"   # number of states 
    STATES = "STATES"       # states data
    
    TIMESTEPS = "TIMESTEPS" # timesteps
    
    
    
    