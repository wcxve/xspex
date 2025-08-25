from importlib.metadata import PackageNotFoundError, version as _version

try:
    _dist_name = __package__.split('.', 1)[0] if __package__ else 'xspex'
    __version__: str = _version(_dist_name)
except PackageNotFoundError:
    __version__: str = 'dev'

__all__ = ['__version__']
