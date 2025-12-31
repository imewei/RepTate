"""
Define the C-variables and functions from the C-files that are needed in Python
"""
import numpy as np
from ctypes import c_double, c_int, c_bool
import sys
import os

from RepTate.core.ctypes_loader import load_ctypes_library
dir_path = os.path.dirname(
    os.path.realpath(__file__)
)  # get the directory path of current file
if sys.maxsize > 2 ** 32:
    # 64-bit system
    lib_path = os.path.join(dir_path, "dtd_lib_%s.so" % (sys.platform))
else:
    # 32-bit system
    lib_path = os.path.join(dir_path, "dtd_lib_%s_i686.so" % (sys.platform))
dtd_lib = load_ctypes_library(lib_path, "DTD library")

dynamic_tube_dilution_freq = dtd_lib.dynamic_tube_dilution_freq
dynamic_tube_dilution_freq.restype = c_bool

dynamic_tube_dilution_time = dtd_lib.dynamic_tube_dilution_time
dynamic_tube_dilution_time.restype = c_bool


def calculate_dtd_freq(params, EPS):
    """Calculate dynamic tube dilution rheological moduli in frequency domain.

    Computes the storage (G') and loss (G'') moduli using the dynamic tube dilution
    (DTD) model, which accounts for constraint release and tube dilation effects in
    entangled polymer systems under oscillatory shear.

    Args:
        params: Tuple containing (G0, a, tau_e, z, w) where:
            - G0 (float): Plateau modulus (Pa)
            - a (float): Dilution exponent parameter (dimensionless)
            - tau_e (float): Entanglement relaxation time (s)
            - z (float): Number of entanglements per chain (dimensionless)
            - w (np.ndarray): Array of angular frequencies (rad/s)
        EPS (float): Numerical tolerance for integration convergence

    Returns:
        tuple: (gp, gpp, success) where:
            - gp (np.ndarray): Storage modulus G'(w) values (Pa)
            - gpp (np.ndarray): Loss modulus G''(w) values (Pa)
            - success (bool): True if calculation converged within tolerance
    """
    G0, a, tau_e, z, w = params
    n = len(w)

    w_arr = (c_double * n)()
    gp_arr = (c_double * n)()
    gpp_arr = (c_double * n)()
    w_arr[:] = w[:]
    gp_arr[:] = np.zeros(n)[:]
    gpp_arr[:] = np.zeros(n)[:]

    success = dynamic_tube_dilution_freq(
        c_double(G0),
        c_double(a),
        c_double(tau_e),
        c_double(z),
        c_int(n),
        w_arr,
        gp_arr,
        gpp_arr,
        c_double(EPS),
    )

    # convert ctypes array to numpy
    return (np.asarray(gp_arr[:]), np.asarray(gpp_arr[:]), success)


def calculate_dtd_time(params, EPS):
    """Calculate dynamic tube dilution relaxation modulus in time domain.

    Computes the stress relaxation modulus G(t) using the dynamic tube dilution
    (DTD) model, which describes how entangled polymer melts relax stress over
    time through constraint release and tube dilation mechanisms.

    Args:
        params: Tuple containing (G0, a, tau_e, z, t) where:
            - G0 (float): Plateau modulus (Pa)
            - a (float): Dilution exponent parameter (dimensionless)
            - tau_e (float): Entanglement relaxation time (s)
            - z (float): Number of entanglements per chain (dimensionless)
            - t (np.ndarray): Array of time values (s)
        EPS (float): Numerical tolerance for integration convergence

    Returns:
        tuple: (gt, success) where:
            - gt (np.ndarray): Relaxation modulus G(t) values (Pa)
            - success (bool): True if calculation converged within tolerance
    """
    G0, a, tau_e, z, t = params
    n = len(t)

    t_arr = (c_double * n)()
    gt_arr = (c_double * n)()
    t_arr[:] = t[:]
    gt_arr[:] = np.zeros(n)[:]

    success = dynamic_tube_dilution_time(
        c_double(G0),
        c_double(a),
        c_double(tau_e),
        c_double(z),
        c_int(n),
        t_arr,
        gt_arr,
        c_double(EPS),
    )

    # convert ctypes array to numpy
    return (np.asarray(gt_arr[:]), success)
