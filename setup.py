"""
Build the pyglu Python extension.

Requires:
  - NVIDIA CUDA Toolkit (nvcc).  Set CUDA_HOME if not at /usr/local/cuda.
  - GCC / G++ (for NICSLU and GLU CPU code).
  - pybind11 (pip install pybind11).

Install (editable, recommended during development):
    pip install -e . --no-build-isolation

Install (regular):
    pip install . --no-build-isolation
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import pybind11

# ── Directory layout ────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.resolve()
SRC     = ROOT / "src"
NICSLU  = SRC / "nicslu"
PREPROC = SRC / "preprocess"
INCLUDE = ROOT / "include"

# ── CUDA home ───────────────────────────────────────────────────────────────
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
NVCC      = os.path.join(CUDA_HOME, "bin", "nvcc")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _run(*args, **kwargs):
    """Run a subprocess command and raise on failure."""
    subprocess.check_call(list(args), **kwargs)


# ── Custom build_ext ─────────────────────────────────────────────────────────
class GLUBuildExt(build_ext):
    """
    Custom builder that:
      1. Builds NICSLU static libraries (nicslu.a, nicslu_util.a) via make.
      2. Compiles the GLU CPU/GPU object files with gcc/g++/nvcc.
      3. Links everything via nvcc so device code is handled correctly.
    """

    def build_extensions(self):
        # Step 1 — NICSLU static libraries
        _run("make", "lib", "util", cwd=str(NICSLU))

        # Step 2 — Compile GLU CPU/GPU translation units
        build_tmp = Path(self.build_temp)
        build_tmp.mkdir(parents=True, exist_ok=True)

        inc_flags = [
            f"-I{INCLUDE}",
            f"-I{NICSLU / 'include'}",
            f"-I{NICSLU / 'util'}",
            f"-I{PREPROC}",
        ]

        # preprocess.c
        preproc_o = build_tmp / "preprocess.o"
        _run("gcc", "-O2", "-Wall", "-msse2",
             *inc_flags, "-DNO_ATOMIC", "-DSSE2",
             "-c", str(PREPROC / "preprocess.c"), "-o", str(preproc_o))

        # preprocess_arrays.c
        preproc_arr_o = build_tmp / "preprocess_arrays.o"
        _run("gcc", "-O2", "-Wall", "-msse2",
             *inc_flags, "-DNO_ATOMIC", "-DSSE2",
             "-c", str(PREPROC / "preprocess_arrays.c"),
             "-o", str(preproc_arr_o))

        # symbolic.cc
        symbolic_o = build_tmp / "symbolic.o"
        _run("g++", "-O3", "-m64", "-std=c++11", "-Wall",
             *inc_flags,
             "-c", str(SRC / "symbolic.cc"), "-o", str(symbolic_o))

        # Timer.cpp
        timer_o = build_tmp / "Timer.o"
        _run("g++", "-O3", "-m64", "-std=c++11", "-Wall",
             *inc_flags,
             "-c", str(SRC / "Timer.cpp"), "-o", str(timer_o))

        # numeric.cu  (compiled by nvcc)
        numeric_o = build_tmp / "numeric.o"
        _run(NVCC, "-O3", "-std=c++11",
             f"-I{INCLUDE}",
             f"-I{NICSLU / 'include'}",
             f"-I{NICSLU / 'util'}",
             f"-I{PREPROC}",
             "-Xcompiler", "-fPIC",
             "-c", str(SRC / "numeric.cu"), "-o", str(numeric_o))

        # Attach pre-built objects to the extension so they get linked in
        extra_objs = [
            str(preproc_o),
            str(preproc_arr_o),
            str(symbolic_o),
            str(timer_o),
            str(numeric_o),
            str(NICSLU / "lib" / "nicslu.a"),
            str(NICSLU / "util" / "nicslu_util.a"),
        ]
        for ext in self.extensions:
            ext.extra_objects = extra_objs

        # Step 3 — Use nvcc as the linker to handle CUDA device code
        # Redirect the compiler's link command to nvcc.
        self.compiler.set_executable(
            "linker_exe",
            [NVCC, "--compiler-options", "-fPIC"])

        super().build_extensions()


# ── Extension definition ─────────────────────────────────────────────────────
glu_ext = Extension(
    name="pyglu._pyglu",
    sources=[str(SRC / "glu_binding.cpp")],
    include_dirs=[
        str(INCLUDE),
        str(NICSLU / "include"),
        str(NICSLU / "util"),
        str(PREPROC),
        pybind11.get_include(),
        os.path.join(CUDA_HOME, "include"),
    ],
    library_dirs=[
        os.path.join(CUDA_HOME, "lib64"),
    ],
    libraries=["cudart", "m", "rt", "pthread"],
    extra_compile_args=["-O3", "-std=c++11", "-Wall", "-fPIC"],
    language="c++",
)

setup(
    name="pyglu",
    version="0.1.0",
    packages=["pyglu"],
    ext_modules=[glu_ext],
    cmdclass={"build_ext": GLUBuildExt},
    python_requires=">=3.9",
    install_requires=["numpy>=1.22"],
    extras_require={"scipy": ["scipy>=1.9"]},
)
