"""All ACE solvers. Importing this package registers every solver in ace_core.registry.

To add a solver: create a module here that calls ace_core.register(SolverSpec(...)),
then import it below. See SOLVER_FRAMEWORK.md.
"""
from . import machine_elements  # noqa: F401
from . import structural        # noqa: F401
from . import structural3d      # noqa: F401
