# Licenses of what this worker runs

IDRIS does not include CalculiX or gmsh source code. The container installs both as
separate programs and calls them as subprocesses (`gmsh` through its Python
bindings, `ccx` as a command). They keep their own licenses, and the GPL of those
programs applies to those programs, not to IDRIS code that runs them.

| Program | Version reported by `/capabilities` | License | How it is used |
|---|---|---|---|
| CalculiX `ccx` | `ccx_version` | GPL-2.0-or-later (CalculiX is distributed under the GPL; `ccx` also links ARPACK and SPOOLES) | Solver executable: reads the `.inp` we write, writes `.frd`/`.dat` we parse. Started with `subprocess`, one temporary folder per request. |
| gmsh | `gmsh_version` | GPL-2.0-or-later (with an option for a commercial license from the authors) | Mesher: STEP import, surface picking, tetrahedral mesh, written out as a mesh for ccx. Called through the Python bindings in the same container. |
| numpy | (installed by pip) | BSD-3-Clause | Only in the worker's own post-processing (field normalisation, convergence ratios). |
| Python 3 | (image) | PSF-2.0 | Runtime. |
| Debian slim base image | (image) | Debian Free Software Guidelines | Base filesystem. |

Rules that keep it that way:

1. No CalculiX or gmsh source file is copied into this repository, and no part of
   either is linked into a Python extension of ours.
2. The worker's own code only writes input decks, runs the executables, and reads
   their text output.
3. The image is built from `apt` packages and a PyPI wheel; nothing is vendored.
4. Anyone redistributing an image built from this Dockerfile must also make the
   corresponding source of the GPL programs available — the `apt` packages do
   that, and this note records where they came from.
