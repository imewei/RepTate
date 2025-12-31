"""
Define the C-variables and functions from the C-files that are needed in Python
"""
import numpy as np
from ctypes import c_double
import sys
import os

from RepTate.core.ctypes_loader import load_ctypes_library
dir_path = os.path.dirname(
    os.path.realpath(__file__)
)  # get the directory path of current file
if sys.maxsize > 2 ** 32:
    # 64-bit system
    lib_path = os.path.join(dir_path, "rp_blend_lib_%s.so" % (sys.platform))
else:
    # 32-bit system
    lib_path = os.path.join(dir_path, "rp_blend_lib_%s_i686.so" % (sys.platform))
rp_blend_lib = load_ctypes_library(lib_path, "RP blend library")

derivs_rp_blend_shear = rp_blend_lib.derivs_rp_blend_shear
derivs_rp_blend_shear.restype = None

derivs_rp_blend_uext = rp_blend_lib.derivs_rp_blend_uext
derivs_rp_blend_uext.restype = None


def compute_derivs_shear(sigma, p, t, with_fene):
    """Compute time derivatives of stress tensor components for shear flow.

    Calculates the rate of change of the conformation tensor (stress) components
    for a polymer blend under steady shear flow using the Rolie-Poly model. The
    model combines convective constraint release and chain stretch dynamics.

    Args:
        sigma (np.ndarray): Flattened stress tensor components (3*n*n elements)
                            representing upper triangular components of n modes
        p (tuple): Parameter tuple (n, lmax, phi, taud, taus, beta, delta, gamma_dot, _):
            - n (int): Number of relaxation modes
            - lmax (float): Maximum chain stretch ratio
            - phi (np.ndarray): Volume fractions of each mode
            - taud (np.ndarray): Reptation/orientation times for each mode (s)
            - taus (np.ndarray): Stretch relaxation times for each mode (s)
            - beta (float): CCR parameter (convective constraint release strength)
            - delta (float): Chain stretch parameter
            - gamma_dot (float): Shear rate (1/s)
        t (float): Current time value (s)
        with_fene (float): Flag (0 or 1) to enable finite extensibility effects

    Returns:
        list: Time derivatives d(sigma)/dt as flattened array (3*n*n elements)
    """
    c = 3
    n, lmax, phi, taud, taus, beta, delta, gamma_dot, _ = p
    # void derivs_rp_blend_shear(double *deriv, double *sigma, double *phi, double *taus, double *taud, double *p, double t)
    # n = p[0];
    # lmax = p[1];
    # beta = p[2];
    # delta = p[3];
    # gamma_dot = p[4];
    # with_fene = p[5];

    p_arr = (c_double * 6)()
    p_arr[:] = [n, lmax, beta, delta, gamma_dot, with_fene]
    deriv_arr = (c_double * (c * n * n))(*np.zeros(c * n * n))
    sigma_arr = (c_double * (c * n * n))(*sigma[:])
    phi_arr = (c_double * n)(*phi[:])
    taud_arr = (c_double * n)(*np.array(taud) / 2.0)  # hard coded factor 2 in C routine
    taus_arr = (c_double * n)(*taus[:])

    derivs_rp_blend_shear(
        deriv_arr, sigma_arr, phi_arr, taus_arr, taud_arr, p_arr, c_double(t)
    )

    # return results as numpy array
    return deriv_arr[:]


def compute_derivs_uext(sigma, p, t, with_fene):
    """Compute time derivatives of stress tensor components for uniaxial extension.

    Calculates the rate of change of the conformation tensor (stress) components
    for a polymer blend under uniaxial extensional flow using the Rolie-Poly model.
    Uniaxial extension has fewer independent stress components (2 vs 3 for shear).

    Args:
        sigma (np.ndarray): Flattened stress tensor components (2*n*n elements)
                            for axial and radial directions of n modes
        p (tuple): Parameter tuple (n, lmax, phi, taud, taus, beta, delta, gamma_dot, _):
            - n (int): Number of relaxation modes
            - lmax (float): Maximum chain stretch ratio
            - phi (np.ndarray): Volume fractions of each mode
            - taud (np.ndarray): Reptation/orientation times for each mode (s)
            - taus (np.ndarray): Stretch relaxation times for each mode (s)
            - beta (float): CCR parameter (convective constraint release strength)
            - delta (float): Chain stretch parameter
            - gamma_dot (float): Extension rate (1/s)
        t (float): Current time value (s)
        with_fene (float): Flag (0 or 1) to enable finite extensibility effects

    Returns:
        list: Time derivatives d(sigma)/dt as flattened array (2*n*n elements)
    """
    c = 2
    n, lmax, phi, taud, taus, beta, delta, gamma_dot, _ = p
    # void derivs_rp_blend_shear(double *deriv, double *sigma, double *phi, double *taus, double *taud, double *p, double t)
    # n = p[0];
    # lmax = p[1];
    # beta = p[2];
    # delta = p[3];
    # gamma_dot = p[4];
    # with_fene = p[5];

    p_arr = (c_double * 6)()
    p_arr[:] = [n, lmax, beta, delta, gamma_dot, with_fene]

    deriv_arr = (c_double * (c * n * n))(*np.zeros(c * n * n))
    sigma_arr = (c_double * (c * n * n))(*sigma[:])
    phi_arr = (c_double * n)(*phi[:])
    taud_arr = (c_double * n)(*np.array(taud) / 2.0)  # hard coded factor 2 in C routine
    taus_arr = (c_double * n)(*taus[:])

    derivs_rp_blend_uext(
        deriv_arr, sigma_arr, phi_arr, taus_arr, taud_arr, p_arr, c_double(t)
    )

    # return results as numpy array
    return deriv_arr[:]
