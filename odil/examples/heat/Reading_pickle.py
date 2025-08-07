import numpy as np
import odil
from odil import Field
from odil.core import checkpoint_load

# rebuild the domain you *think* it came from
domain = odil.Domain(
    cshape=(64,64,64),               # you think it’s 64³
    dimnames=("t","x","y"),
    lower=(0.0,0.0,0.0),
    upper=(1.0,1.0,1.0),
    multigrid=False,
    dtype=np.float64
)
state = odil.State(fields={"u": Field()})
checkpoint_load(domain, state, "out_heat2D_direct/ref_solution.pickle")
u = np.array(domain.field(state, "u"))
print("loaded u shape:", u.shape)
