# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:15:57 2018

@author: phydjr
"""

from __future__ import annotations

import jax.numpy as jnp

from RepTate.core.fitting import nlsq_optimize as optimize

# import pandas as pa


def afun(A, LL):
    """Compute constraint equation for strand alignment parameter A.

    This function calculates the residual of the self-consistency equation for the
    alignment parameter A in the polydisperse strand model. The equation balances
    the contribution of different strand components weighted by their volume fractions
    and differential free energies.

    Args:
        A: Alignment parameter that couples strand orientation to differential free energy.
        LL: Strand stretch ratio (NT/NS), representing the extension of the polymer strand.

    Returns:
        Residual of the constraint equation LL * sum1/sum2 - 1. Zero when A satisfies
        the self-consistency condition.
    """
    tem = 1.0 - A * edf
    sum1 = jnp.sum(phi * edf / tem)
    sum2 = jnp.sum(phi * edf / (tem**2))
    return LL * sum1 / sum2 - 1.0


def _solve_for_A(LL):
    lower = 0.0
    upper = iedfmax
    x0 = jnp.array([(lower + upper) / 2.0])
    result = optimize.root(lambda x: afun(x[0], LL), x0, bounds=(lower, upper))
    return float(result.x[0])


def _minimize_scalar(fun, bounds, args=()):
    lower, upper = bounds
    x0 = jnp.array([(lower + upper) / 2.0])
    result = optimize.minimize(lambda x: fun(x[0], *args), x0, bounds=bounds)
    return result


def Free2(NS, NT):
    """Calculate total free energy for polydisperse strand with fixed NS and NT.

    Computes the Helmholtz free energy of a polydisperse polymer strand system
    including entropic contributions from strand stretching, mixing of different
    components, and surface energy from the strand-matrix interface. The calculation
    accounts for the distribution of strand lengths via volume fractions phi and
    differential free energies df.

    Args:
        NS: Number of strands (discretized into units).
        NT: Total number of monomers in the strand system.

    Returns:
        Total dimensionless free energy FF including bulk entropic terms and
        surface contributions (mus * Stil). Lower values indicate more stable states.
    """
    LL = NT / NS
    A = _solve_for_A(LL)
    tem = 1.0 - A * edf
    sum1 = jnp.sum(phi * edf / tem)
    AB = 1.0 / sum1
    w = AB * phi * edf / tem
    v = w / tem / LL
    # now, put together free energy terms
    logw = jnp.log(w)
    logv = jnp.log(v)
    logc = jnp.log(v - w / LL)
    sum1 = jnp.sum(
        w * (2 * logw - logphi) / LL
        - v * logv
        + (v - w / LL) * logc
        - v * df
    )
    FF = NT * sum1 - NS * jnp.log(LL) - NT * E0
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


def Freequi(NS, NT):
    """Calculate quiescent free energy for monodisperse strand with fixed NS and NT.

    Computes the Helmholtz free energy for a quiescent (undeformed) monodisperse
    polymer strand system. This simpler model assumes all strands have identical
    properties and serves as a baseline reference state for comparing with the
    polydisperse case.

    Args:
        NS: Number of strands (discretized into units).
        NT: Total number of monomers in the strand system.

    Returns:
        Total dimensionless free energy FF for the quiescent state including
        entropic stretching and surface energy contributions.
    """
    LL = NT / NS
    FF = -NS * jnp.log(LL) - NT * E0 + (NT - NS) * jnp.log(1.0 - 1.0 / LL)
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
    """Find minimum free energy for polydisperse strand at fixed NT by optimizing over NS.

    Determines the thermodynamically favorable number of strands NS by minimizing
    the total free energy Free2(NS, NT) over the physically allowed range of NS
    values. This represents the mean-field equilibrium state.

    Args:
        NT: Total number of monomers in the strand system (fixed).

    Returns:
        Minimum free energy value achieved at optimal NS.
    """
    res = _minimize_scalar(Free2, bounds=(1, 0.999999 * NT), args=(NT,))
    return float(res.fun)


def Freefluc(NT):
    """Calculate free energy including Gaussian fluctuations around minimum for polydisperse strand.

    Computes the free energy accounting for thermal fluctuations in the number of
    strands NS around the mean-field minimum. The fluctuation correction uses a
    Gaussian approximation based on the second derivative (curvature) of Free2 at
    the minimum, representing entropic contributions from fluctuations.

    Args:
        NT: Total number of monomers in the strand system (fixed).

    Returns:
        Free energy including fluctuation correction: F_min + log(d²F/dNS² / 2π).
        The correction is positive, slightly raising the free energy.
    """
    res = _minimize_scalar(Free2, bounds=(1, 0.999999 * NT), args=(NT,))
    nsmid = float(res.x[0])
    ##second derivative
    d2fdn2 = (
        Free2(nsmid + 0.1, NT) + Free2(nsmid - 0.1, NT) - 2 * Free2(nsmid, NT)
    ) / 0.01
    fNT = float(res.fun) + float(jnp.log(d2fdn2 / 2 / jnp.pi))
    return float(fNT)


def Free1qui(NT):
    """Find minimum quiescent free energy at fixed NT by optimizing over NS.

    Determines the equilibrium number of strands NS for the quiescent (monodisperse)
    system by minimizing Freequi(NS, NT). This provides a reference state for
    comparison with the polydisperse case.

    Args:
        NT: Total number of monomers in the strand system (fixed).

    Returns:
        Minimum quiescent free energy value at optimal NS.
    """
    res = _minimize_scalar(Freequi, bounds=(1, 0.999999 * NT), args=(NT,))
    return float(res.fun)


def Freeflucqui(NT):
    """Calculate quiescent free energy including Gaussian fluctuations around minimum.

    Computes the quiescent free energy with fluctuation corrections using a Gaussian
    approximation. The second derivative of Freequi at the minimum quantifies the
    entropic cost of fluctuations in NS around the equilibrium value.

    Args:
        NT: Total number of monomers in the strand system (fixed).

    Returns:
        Quiescent free energy with fluctuation correction: F_min + log(d²F/dNS² / 2π).
    """
    res = _minimize_scalar(Freequi, bounds=(1, 0.999999 * NT), args=(NT,))
    nsmid = float(res.x[0])
    ##second derivative
    d2fdn2 = (
        Freequi(nsmid + 0.1, NT) + Freequi(nsmid - 0.1, NT) - 2 * Freequi(nsmid, NT)
    ) / 0.01
    fNT = float(res.fun) + float(jnp.log(d2fdn2 / 2 / jnp.pi))
    return float(fNT)


def Freesum(NT):
    """Calculate polydisperse free energy via exact partition function summation over all NS.

    Computes the exact free energy by summing the Boltzmann factors exp(-Free2(NS, NT))
    over all possible strand numbers NS from 1 to NT-1. This accounts for all
    thermally accessible states beyond the Gaussian approximation. The result is
    F = -log(Z) where Z is the partition function.

    Args:
        NT: Total number of monomers in the strand system (fixed).

    Returns:
        Exact free energy from full partition function sum: -log(Σ exp(-F(NS))).
    """
    total = 0.0
    NS = 1
    while NS < int(NT):
        total = total + float(jnp.exp(-Free2(NS, NT)))
        NS = NS + 1
    fren = -float(jnp.log(total))
    return float(fren)


def Freesumqui(NT):
    """Calculate quiescent free energy via exact partition function summation over all NS.

    Computes the exact quiescent free energy by summing Boltzmann factors
    exp(-Freequi(NS, NT)) over all possible strand numbers. This provides the
    reference partition function for the monodisperse/quiescent system.

    Args:
        NT: Total number of monomers in the strand system (fixed).

    Returns:
        Exact quiescent free energy: -log(Σ exp(-F_qui(NS))).
    """
    total = 0.0
    NS = 1
    while NS < int(NT):
        total = total + float(jnp.exp(-Freequi(NS, NT)))
        NS = NS + 1
    fren = -float(jnp.log(total))
    return float(fren)


def findDfStar(params):
    """Find maximum activation barrier for strand nucleation via optimization.

    Searches for the peak nucleation barrier ΔF* by optimizing over total monomer
    number NT. Unlike the discrete search in GOpolySTRAND, this version uses
    continuous optimization with the FreeTrue function that interpolates the
    true quiescent landscape. Returns the maximum barrier height.

    Args:
        params: Dictionary containing:
            - landscape: Array of true quiescent free energies vs NT
            - phi: Volume fractions of strand length components
            - df: Differential free energies of components
            - epsilonB: Binding energy per monomer (E0)
            - muS: Surface energy per unit area (mus)

    Returns:
        Maximum nucleation barrier height -min(FreeTrue(NT)) obtained via
        continuous optimization. Higher values suppress strand nucleation.
    """
    # Extract params
    global E0, mus, phi, df, edf, edfmax, logphi, iedfmax, arsq, ar, numc, trueQuiescent

    trueQuiescent = params["landscape"]
    phi = jnp.asarray(params["phi"])
    df = jnp.asarray(params["df"])
    numc = int(phi.size)
    E0 = float(params["epsilonB"])
    mus = float(params["muS"])
    maxNT = trueQuiescent.size

    # a_r
    arsq = 9.0 / 16.0 * jnp.pi
    ar = float(jnp.sqrt(arsq))

    # setting up some parameters
    edf = jnp.exp(df)
    logphi = jnp.log(phi)
    edfmax = float(jnp.max(edf))
    iedfmax = 0.999999999999999 / edfmax

    res = _minimize_scalar(FreeTrue, bounds=(3.0, maxNT), args=())
    return -float(res.fun)


def FreeTrue(NT):
    """Compute negative total barrier using interpolated true quiescent landscape.

    Evaluates the nucleation barrier at a given NT by linearly interpolating the
    true quiescent free energy landscape and combining it with fluctuation corrections.
    Returns the negative barrier (for minimization to find maximum barrier).

    Args:
        NT: Total number of monomers (can be non-integer for smooth optimization).

    Returns:
        Negative of the total nucleation barrier: -(F_true_qui + F_fluc - F_fluc_qui).
        Minimizing this function finds the maximum barrier height.
    """
    NTlow = int(jnp.floor(NT))
    NThigh = NTlow + 1
    trueQ = trueQuiescent[NTlow] + (trueQuiescent[NThigh] - trueQuiescent[NTlow]) * (
        NT - NTlow
    )
    #!3return -(Freefluc(NT)-Freeflucqui(NT))
    return -(trueQ + Freefluc(NT) - Freeflucqui(NT))
