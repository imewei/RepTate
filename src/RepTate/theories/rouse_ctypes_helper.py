"""
Define the C-variables and functions from the C-files that are needed in Python
"""
import numpy as np
from ctypes import c_double, c_int
import sys
import os

from RepTate.core.ctypes_loader import load_ctypes_library
dir_path = os.path.dirname(
    os.path.realpath(__file__)
)  # get the directory path of current file
if sys.maxsize > 2 ** 32:
    # 64-bit system
    lib_path = os.path.join(dir_path, "rouse_lib_%s.so" % (sys.platform))
else:
    # 32-bit system
    lib_path = os.path.join(dir_path, "rouse_lib_%s_i686.so" % (sys.platform))
rouse_lib = load_ctypes_library(lib_path, "Rouse library")

continuous_rouse_freq_interp = rouse_lib.continuous_rouse_freq_interp
continuous_rouse_freq_interp.restype = None

continuous_rouse_time_interp = rouse_lib.continuous_rouse_time_interp
continuous_rouse_time_interp.restype = None


def approx_rouse_frequency(params):
    """Calculate Rouse model rheological moduli in frequency domain with continuous N.

    Computes storage (G') and loss (G'') moduli for an unentangled polymer melt
    using the continuous Rouse model with interpolation for non-integer chain
    lengths. The Rouse model treats polymers as Gaussian chains with bead-spring
    dynamics and no entanglement constraints.

    Args:
        params: Tuple containing (G0, tau0, N, w) where:
            - G0 (float): Plateau modulus or characteristic modulus (Pa)
            - tau0 (float): Segmental relaxation time (s)
            - N (float): Number of Rouse segments (can be non-integer)
            - w (np.ndarray): Array of angular frequencies (rad/s)

    Returns:
        tuple: (gp, gpp) where:
            - gp (np.ndarray): Storage modulus G'(w) values (Pa)
            - gpp (np.ndarray): Loss modulus G''(w) values (Pa)
    """
    G0, tau0, N, w = params
    n = len(w)

    w_arr = (c_double * n)()
    gp_arr = (c_double * n)()
    gpp_arr = (c_double * n)()
    w_arr[:] = w[:]
    gp_arr[:] = np.zeros(n)[:]
    gpp_arr[:] = np.zeros(n)[:]

    continuous_rouse_freq_interp(
        c_int(n), c_double(G0), c_double(tau0), c_double(N), w_arr, gp_arr, gpp_arr
    )

    # convert ctypes array to numpy
    return (np.asarray(gp_arr[:]), np.asarray(gpp_arr[:]))


def approx_rouse_time(params):
    """Calculate Rouse model relaxation modulus in time domain with continuous N.

    Computes the stress relaxation modulus G(t) for an unentangled polymer melt
    using the continuous Rouse model with interpolation for non-integer chain
    lengths. Describes how polymer stress relaxes through bead-spring chain dynamics.

    Args:
        params: Tuple containing (G0, tau0, N, t) where:
            - G0 (float): Plateau modulus or characteristic modulus (Pa)
            - tau0 (float): Segmental relaxation time (s)
            - N (float): Number of Rouse segments (can be non-integer)
            - t (np.ndarray): Array of time values (s)

    Returns:
        np.ndarray: Relaxation modulus G(t) values (Pa)
    """
    G0, tau0, N, t = params
    n = len(t)

    t_arr = (c_double * n)()
    gt_arr = (c_double * n)()
    t_arr[:] = t[:]
    gt_arr[:] = np.zeros(n)[:]

    continuous_rouse_time_interp(
        c_int(n), c_double(G0), c_double(tau0), c_double(N), t_arr, gt_arr
    )

    # convert ctypes array to numpy
    return np.asarray(gt_arr[:])
