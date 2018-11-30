"""Settings

Mostly copied and adapted from https://github.com/theislab/scanpy/
"""
import os

verbosity = 1
"""Set global verbosity level.
Level 0: only show 'error' messages.
Level 1: also show 'warning' messages.
Level 2: also show 'info' messages.
Level 3: also show 'hint' messages.
Level 4: also show very detailed progress.
Level 5: also show even more detailed progress.
etc.
"""

writedir = './write/'
"""Directory where the function scanpy.write writes to by default.
"""

cachedir = './cache/'
"""Default cache directory.
"""

figdir = './figures/'
"""Directory where plots are saved.
"""

max_memory = 15
"""Maximal memory usage in Gigabyte.
Is currently not well respected....
"""

n_jobs = os.cpu_count()
"""Default number of jobs/ CPUs to use for parallel computing.
"""

logfile = ''
"""Name of logfile. By default is set to '' and writes to standard output."""

# ------------------------------------------------------------------------------
# Private global variables & functions
# ------------------------------------------------------------------------------

def _set_start_time():
    from time import time
    return time()

_start = _set_start_time()
"""Time when the settings module is first imported."""

_previous_time = _start
"""Variable for timing program parts."""

_previous_memory_usage = -1
"""Stores the previous memory usage."""


def _is_run_from_ipython():
    """Determines whether run from Ipython.
    Only affects progress bars.
    """
    try:
        __IPYTHON__
        return True
    except NameError:
        return False