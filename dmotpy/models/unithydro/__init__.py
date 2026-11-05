from .uh_identity_0 import DplIdentity0
from .uh_half_1 import DplHalf1
from .uh_full_2 import DplFull2
from .uh_tri_3 import DplTri3
from .uh_tri_4 import DplTri4
from .uh_exp_5 import DplExp5
from .uh_gamma_6 import DplGamma6
from .uh_uniform_7 import DplUniform7
from .uh_delay_8 import DplDelay8

# Map semantic names and numeric identifiers to unit-hydrograph classes.
UH_MAP = {
    # Semantic names.
    "identity0": DplIdentity0,
    "half1": DplHalf1,
    "full2": DplFull2,
    "tri3": DplTri3,
    "tri4": DplTri4,
    "exp5": DplExp5,
    "gamma6": DplGamma6,
    "uniform7": DplUniform7,
    "delay8": DplDelay8,
    
    # Numeric identifiers are accepted as strings for convenience.
    "0": DplIdentity0,
    "1": DplHalf1,
    "2": DplFull2,
    "3": DplTri3,
    "4": DplTri4,
    "5": DplExp5,
    "6": DplGamma6,
    "7": DplUniform7,
    "8": DplDelay8
}

# Public exports for wildcard imports.
__all__ = [
    "DplIdentity0",
    "DplHalf1",
    "DplFull2",
    "DplTri3",
    "DplTri4",
    "DplExp5",
    "DplGamma6",
    "DplUniform7",
    "DplDelay8",
    "UH_MAP"
]