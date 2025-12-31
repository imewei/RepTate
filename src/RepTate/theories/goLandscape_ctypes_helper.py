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
    lib_path = os.path.join(dir_path, "landscape_%s.so" % (sys.platform))
else:
    # 32-bit system
    lib_path = os.path.join(dir_path, "landscape_%s_i686.so" % (sys.platform))

landscape_function_lib = load_ctypes_library(lib_path, "landscape library")

python_c_landscape = landscape_function_lib.landscape
python_c_landscape.restype = c_double

def GO_Landscape(NT, epsilon, mu):
    """Compute the quiescent free energy landscape for polymer chain configurations.

    Calculates the dimensionless free energy of a polymer chain in a quiescent
    (unstressed) state using the Graham-Olmsted landscape theory. This represents
    the potential energy landscape that governs chain conformations in the tube.

    Args:
        NT (float): Number of tube segments (chain length in tube diameters)
        epsilon (float): Confinement parameter representing tube constraint strength
        mu (float): Chemical potential or dimensionless chain tension parameter

    Returns:
        float: Dimensionless free energy of the chain configuration
    """
    c_doub_NT = (c_double)(NT)
    c_doub_mu = (c_double)(mu)
    c_doub_epsilon = (c_double)(epsilon)
    return python_c_landscape(c_doub_NT, c_doub_mu, c_doub_epsilon)
