from __future__ import annotations

import sys
from pathlib import Path

from setuptools import Extension, setup


def build_extensions():
    try:
        import numpy as np
        from Cython.Build import cythonize
    except Exception:
        return []

    compile_args = ["/O2"] if sys.platform.startswith("win") else ["-O3"]
    extensions = [
        Extension(
            "signalprocessor._core_cy",
            [str(Path("src") / "signalprocessor" / "_core_cy.pyx")],
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
        )
    ]
    return cythonize(extensions, compiler_directives={"language_level": "3"})


setup(ext_modules=build_extensions())
