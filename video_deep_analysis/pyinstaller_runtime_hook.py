"""Allow libraries that inspect optional source comments to run when frozen."""
import inspect

_original_getsource = inspect.getsource


def _frozen_safe_getsource(object):
    try:
        return _original_getsource(object)
    except OSError:
        return ""


inspect.getsource = _frozen_safe_getsource
