# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:15:57 2018

@author: phydjr
"""

from __future__ import annotations

import jax.numpy as jnp

from RepTate.core.fitting import nlsq_optimize as optimize


def _minimize_scalar(fun, bounds, args=()):
    lower, upper = bounds
    x0 = jnp.array([(lower + upper) / 2.0])
    result = optimize.minimize(lambda x: fun(x[0], *args), x0, bounds=bounds)
    return result


def wfun(phi, Df, P, B, NS):
    """Calculate weight distribution of strand components in smooth polydisperse model.

    Computes the weight fractions w_i of different strand length components in the
    polydisperse smooth strand system. The weights depend on volume fractions phi,
    differential free energies Df, alignment parameter P, normalization B, and
    the bending stiffness kappa. The exponential factor accounts for energetic
    and entropic contributions from strand deformation.

    Args:
        phi: Volume fractions of strand length components.
        Df: Differential free energies of components.
        P: Alignment parameter for strand orientation distribution.
        B: Normalization factor ensuring sum of weights equals unity.
        NS: Number of strands (discretized into units).

    Returns:
        Weight distribution array w_i for each strand component, normalized such
        that Σw_i = 1 and Σw_i*Df_i = P.
    """
    return (
        Qs
        * phi
        / (
            NS
            + (Qs - NS)
            * B
            * jnp.exp(-LL * Df - Df**2 / 2.0 / kappa + Df / kappa * P)
        )
    )


def afun(x, NS):
    """Constraint equations for alignment parameter P and normalization B.

    Defines the system of self-consistency equations that determine P and B for
    the polydisperse smooth strand model. The first equation enforces normalization
    (Σw_i = 1), while the second ensures the mean differential free energy equals P.

    Args:
        x: Array [P, B] containing alignment parameter and normalization factor.
        NS: Number of strands (discretized into units).

    Returns:
        Residual array [Σw_i - 1, Σw_i*df_i - P]. Both components should be zero
        when P and B satisfy the self-consistency conditions.
    """
    P = x[0]
    B = x[1]
    wi = wfun(phi, df, P, B, NS)
    sum1 = jnp.sum(wi)
    sum2 = jnp.sum(wi * df)
    return jnp.array([sum1 - 1.0, sum2 - P])


def Free2(Ns, NT):
    """Calculate total free energy for smooth polydisperse strand with bending stiffness.

    Computes the Helmholtz free energy for a polydisperse smooth strand model
    incorporating bending elasticity (via kappa), component mixing entropy (theta
    and w distributions), and surface energy. Solves self-consistently for the
    alignment parameter P and normalization B at fixed NS and NT.

    Args:
        Ns: Array containing number of strands NS as first element (for optimizer interface).
        NT: Total number of monomers in the strand system.

    Returns:
        Total dimensionless free energy FF including mixing entropy of bound (w)
        and unbound (theta) components, bending stiffness contributions, strand
        stretching entropy, surface energy, and binding energy. Lower values
        indicate more stable states.
    """
    global LL, Qs, kappa, Pprevious, Bprevious
    NS = Ns[0]
    LL = NT / NS
    Qs = Qs0 * NS
    kappa = Kappa0 + 1.0 / LL**2

    sol = optimize.root(
        afun,
        jnp.array([max(0.00001, Pprevious), Bprevious]),
        args=(NS,),
        method="hybr",
        jac=False,
    )
    P = sol.x[0]
    B = sol.x[1]
    Pprevious = float(P)
    Bprevious = float(B)

    w = wfun(phi, df, P, B, NS)
    theta = jnp.maximum(
        thetaMin, Qs / (Qs - NS) * phi - NS / (Qs - NS) * w
    )
    sum1 = jnp.sum(theta * jnp.log(theta))
    sum2 = jnp.sum(phi * jnp.log(phi))
    sum3 = jnp.sum(w * jnp.log(w))
    sum4 = jnp.sum(w * df)
    sum5 = jnp.sum(w * df**2)

    # print w
    # print theta

    FF = (
        (Qs - NS) * sum1
        - Qs * sum2
        + NS * sum3
        - 0.5 * (NS - 1) * jnp.log(2.0 * jnp.pi / kappa)
        + 0.5 * jnp.log(NS)
        - NT * sum4
        - 0.5 * NS / kappa * (sum5 - sum4**2)
        - E0 * NT
    )
    # surface terms
    aspect = NS**3 / NT**2 / arsq
    aspect_val = float(aspect)
    if aspect_val < 1:
        ep = jnp.sqrt(1.0 - aspect)
        Stil = 2.0 * NS + 2.0 * ar * NT * jnp.arcsin(ep) / ep / jnp.sqrt(NS)
    elif aspect_val > 1:
        eps0 = jnp.sqrt(1.0 - 1.0 / aspect)
        Stil = (
            2.0 * NS
            + arsq * NT**2 * jnp.log((1.0 + eps0) / (1.0 - eps0)) / eps0 / NS**2
        )
    else:
        Stil = 2.0 * NS + 2.0 * ar * NT / jnp.sqrt(NS)
    FF += mus * Stil
    return float(FF)


def Free1(NT):
    """Find minimum free energy for smooth polydisperse strand at fixed NT.

    Determines the equilibrium number of strands NS by minimizing the total free
    energy Free2(NS, NT) using Nelder-Mead simplex optimization. This accounts
    for the complex coupling between bending stiffness, polydispersity, and
    surface energy in the smooth strand model.

    Args:
        NT: Total number of monomers in the strand system (fixed).

    Returns:
        Minimum free energy value at optimal NS. The optimization balances
        bending elasticity, mixing entropy, and surface contributions.
    """
    x0 = NSprevious
    res = optimize.minimize(Free2, x0, method="Nelder-Mead", args=(NT))
    # res=scipy.optimize.minimize_scalar(Free2, bounds=(1,0.999999*NT), args=(NT), method='bounded')
    print(res)
    return float(res.fun)


def Freefluc(NT):
    """Calculate smooth polydisperse free energy with Gaussian fluctuations.

    Computes the free energy including fluctuation corrections for the smooth
    polydisperse strand model. After finding the optimal NS, calculates the
    second derivative to include Gaussian fluctuation entropy. Updates the
    global NSprevious to warm-start subsequent optimizations.

    Args:
        NT: Total number of monomers in the strand system (fixed).

    Returns:
        Free energy with fluctuation correction: F_min + log(d²F/dNS² / 2π).
        Accounts for thermal fluctuations around the mean-field minimum in
        the complex smooth polydisperse strand model.
    """
    global NSprevious
    x0 = NSprevious
    res = optimize.minimize(Free2, x0, method="Nelder-Mead", args=(NT))
    # print res
    # res=scipy.optimize.minimize_scalar(Free2, bounds=(1,0.999999*NT), args=(NT), method='bounded')
    nsmid = float(res.x[0])
    NSprevious = nsmid
    ##second derivative
    d2fdn2 = (
        Free2([nsmid + 0.1], NT) + Free2([nsmid - 0.1], NT) - 2 * Free2([nsmid], NT)
    ) / 0.01
    fNT = float(res.fun) + float(jnp.log(d2fdn2 / 2 / jnp.pi))
    return float(fNT)


def findDfStar(params):
    """Find maximum activation barrier for smooth polydisperse strand nucleation.

    Searches for the peak nucleation barrier ΔF* by marching through NT values
    and evaluating the fluctuation-corrected free energy. The barrier height
    controls nucleation kinetics in the smooth polydisperse strand model. The
    search terminates early if the barrier drops by more than 1 kBT from the
    maximum, indicating the peak has been passed.

    Args:
        params: Dictionary containing:
            - phi: Volume fractions of strand length components
            - df: Differential free energies of components
            - epsilonB: Binding energy per monomer (E0)
            - muS: Surface energy per unit area (mus)
            - Kappa0: Curvature penalty for bending stiffness
            - Qs0: Reference value for bound segment density
            - maxNT: Maximum NT to search

    Returns:
        Tuple (BestBarrier, NT) where BestBarrier is the maximum nucleation
        barrier height and NT is the final evaluated monomer number. Higher
        barriers suppress strand nucleation more strongly.
    """
    global E0, mus, Kappa0, Qs0, maxNT, phi, df, arsq, ar, numc, NSprevious, Pprevious, Bprevious, thetaMin

    # Extract params
    phi = jnp.asarray(params["phi"])
    df = jnp.asarray(params["df"])
    numc = int(phi.size)
    E0 = float(params["epsilonB"])
    mus = float(params["muS"])
    Kappa0 = float(params["Kappa0"])
    Qs0 = float(params["Qs0"])
    maxNT = params["maxNT"]

    # Initialise variables
    # a_r
    arsq = 9.0 / 16.0 * jnp.pi
    thetaMin = 1e-300
    ar = float(jnp.sqrt(arsq))

    # March up the barrier
    NTlist = []
    Flist = []
    Fluclist = []
    Barrierlist = []

    BestBarrier = -10000.0
    NTlist = []
    Flist = []
    Fluclist = []
    Barrierlist = []
    NSprevious = 1.1
    Pprevious = 0.5 * float(jnp.max(df) - jnp.min(df))
    Bprevious = 1.0

    ratio = 2.0
    for i in range(int((maxNT - 1) / ratio)):
        NT = 2.0 + i * ratio
        NTlist.append(NT)
        ans = Freefluc(1.0 * NT)
        Fluclist.append(ans)

        if ans > BestBarrier:
            BestBarrier = ans
            nStar = NT

        if BestBarrier - ans > 1.0:
            break

    return BestBarrier, NT

    # res=scipy.optimize.minimize_scalar(FreeTrue, bounds=(3.0,maxNT), method='bounded')
    # return -res.fun


def findDfStar_Direct(params):
    """Find barrier via direct optimization near previous barrier location.

    Performs a local search for the nucleation barrier maximum by optimizing
    FreeTrue in a narrow window around the previously found barrier location
    (NTprevious). This refined search uses continuous optimization and is more
    efficient than the full marching search when a good initial estimate exists.
    Warm-starts the optimization using previous values of NS, P, and B.

    Args:
        params: Dictionary containing:
            - phi: Volume fractions of strand length components
            - df: Differential free energies of components
            - epsilonB: Binding energy per monomer (E0)
            - muS: Surface energy per unit area (mus)
            - Kappa0: Curvature penalty for bending stiffness
            - Qs0: Reference value for bound segment density
            - NTprevious: Previous barrier location for local search
            - NSprevious: Previous optimal NS for warm-start
            - Pprevious: Previous alignment parameter for warm-start
            - Bprevious: Previous normalization factor for warm-start

    Returns:
        Tuple (barrier, NT, NSprevious, Pprevious, Bprevious) containing the
        refined barrier height, its NT location, and updated warm-start values.
    """
    global E0, mus, Kappa0, Qs0, maxNT, phi, df, arsq, ar, numc, NSprevious, Pprevious, Bprevious, thetaMin

    # Extract params
    phi = jnp.asarray(params["phi"])
    df = jnp.asarray(params["df"])
    numc = int(phi.size)
    E0 = float(params["epsilonB"])
    mus = float(params["muS"])
    Kappa0 = float(params["Kappa0"])
    Qs0 = float(params["Qs0"])
    NTprevious = params["NTprevious"]
    NSprevious = params["NSprevious"]
    Pprevious = params["Pprevious"]
    Bprevious = params["Bprevious"]

    # Initialise variables
    # a_r
    arsq = 9.0 / 16.0 * jnp.pi
    thetaMin = 1e-300
    ar = float(jnp.sqrt(arsq))

    search_frac = 0.1
    upper = NTprevious * (1 + search_frac / 100.0)
    lower = NTprevious * (1 - search_frac)
    res = _minimize_scalar(FreeTrue, bounds=(lower, upper), args=())
    return -float(res.fun), float(res.x[0]), NSprevious, Pprevious, Bprevious


def FreeTrue(NT):
    """Return negative of fluctuation-corrected free energy for maximization.

    Wrapper function that returns the negative of the fluctuation-corrected free
    energy. Used with minimization algorithms to find the maximum nucleation
    barrier. The 1.0 * NT cast ensures NT is treated as a float.

    Args:
        NT: Total number of monomers (can be non-integer for smooth optimization).

    Returns:
        Negative of fluctuation-corrected free energy: -Freefluc(NT). Minimizing
        this function locates the maximum barrier height.
    """
    return -Freefluc(1.0 * NT)
