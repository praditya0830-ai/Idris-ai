"""All ACE solvers. Importing this package registers every solver in ace_core.registry.

To add a solver: create a module here that calls ace_core.register(SolverSpec(...)),
then import it below. See SOLVER_FRAMEWORK.md.
"""
from . import machine_elements  # noqa: F401
from . import structural        # noqa: F401
from . import structural3d      # noqa: F401
from . import analysis_housing  # noqa: F401
from . import gear_iso          # noqa: F401
from . import bolted          # noqa: F401
from . import power_loss      # noqa: F401
from . import nafems          # noqa: F401
from . import bearing_life    # noqa: F401
from . import mesh_convergence  # noqa: F401
from . import gear_pair_iso   # noqa: F401
from . import part_fea        # noqa: F401
