"""GH #587: bngsim ships a PEP 561 ``py.typed`` marker.

Without it mypy, pyright and IDEs treat the package as untyped however well it
is annotated. scikit-build-core's ``wheel.packages`` copies the whole package
directory, so the marker only has to exist there, next to ``_bngsim_core.pyi``.
"""

from importlib.resources import files


def test_package_ships_a_py_typed_marker():
    assert files("bngsim").joinpath("py.typed").is_file()
