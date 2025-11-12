import pathlib

from setuptools import setup


CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the mo-gymnasium version."""
    path = CWD / "meta_traffic" / "__init__.py"
    content = path.read_text()
    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")



setup(name="metatraffic", version=get_version())


