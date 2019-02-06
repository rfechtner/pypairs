import sys
import warnings

from pypairs import settings
from pypairs import log
from pypairs import pairs
from pypairs import utils
from pypairs import datasets

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

if sys.version_info < (3, 0):
    warnings.warn('PyPairs only runs reliably with Python 3, preferrably >=3.6.')
