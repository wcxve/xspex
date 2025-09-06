import importlib.metadata as metadata

try:
    __version__ = metadata.version('xspex')
except metadata.PackageNotFoundError:
    __version__ = 'dev'

__all__ = ['__version__']
