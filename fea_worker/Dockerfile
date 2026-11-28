# IDRIS FEA worker — CalculiX + gmsh behind one small HTTP service.
#
# CalculiX and gmsh are GPL programs, executed here as separate processes
# (subprocess), never imported or copied into IDRIS code. See LICENSES.md.
#
# Cloud Run sets PORT and bills image pull + cold start, so the image stays small:
# no compilers, no build tools.
FROM debian:bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8080

# calculix-ccx: the solver. gmsh: the mesher (CLI plus its Python bindings).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
        calculix-ccx \
        gmsh \
    && rm -rf /var/lib/apt/lists/*

# Python bindings for gmsh: the wheel bundles libgmsh, so the STEP import and the
# surface picking do not depend on where the apt package puts its modules.
RUN pip3 install --no-cache-dir --break-system-packages "gmsh>=4.12,<5" "numpy>=1.26,<3"

WORKDIR /app
COPY fea_worker ./fea_worker

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s \
    CMD python3 -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health').read()"

CMD ["python3", "-m", "fea_worker.app"]
