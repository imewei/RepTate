"""
Define the C-variables and functions from the C-files that are needed in Python
"""
from ctypes import c_double
import sys
import os

from RepTate.core.ctypes_loader import load_ctypes_library
dir_path = os.path.dirname(
    os.path.realpath(__file__)
)  # get the directory path of current file
if sys.maxsize > 2 ** 32:
    # 64-bit system
    lib_path = os.path.join(dir_path, "kww_lib_%s.so" % (sys.platform))
else:
    # 32-bit system
    lib_path = os.path.join(dir_path, "kww_lib_%s_i686.so" % (sys.platform))
kww_lib = load_ctypes_library(lib_path, "KWW library")

kwwc = kww_lib.kwwc
kwwc.argtypes = [c_double, c_double]
kwwc.restype = c_double
kwws = kww_lib.kwws
kwws.argtypes = [c_double, c_double]
kwws.restype = c_double
