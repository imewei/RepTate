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
    res = _minimize_scalar(Free2, bounds=(1, 0.999999 * NT), args=(NT,))
    return float(res.fun)


def Freefluc(NT):
    res = _minimize_scalar(Free2, bounds=(1, 0.999999 * NT), args=(NT,))
    nsmid = float(res.x[0])
    ##second derivative
    d2fdn2 = (
        Free2(nsmid + 0.1, NT) + Free2(nsmid - 0.1, NT) - 2 * Free2(nsmid, NT)
    ) / 0.01
    fNT = float(res.fun) + float(jnp.log(d2fdn2 / 2 / jnp.pi))
    return float(fNT)


def Free1qui(NT):
    res = _minimize_scalar(Freequi, bounds=(1, 0.999999 * NT), args=(NT,))
    return float(res.fun)


def Freeflucqui(NT):
    res = _minimize_scalar(Freequi, bounds=(1, 0.999999 * NT), args=(NT,))
    nsmid = float(res.x[0])
    ##second derivative
    d2fdn2 = (
        Freequi(nsmid + 0.1, NT) + Freequi(nsmid - 0.1, NT) - 2 * Freequi(nsmid, NT)
    ) / 0.01
    fNT = float(res.fun) + float(jnp.log(d2fdn2 / 2 / jnp.pi))
    return float(fNT)


def Freesum(NT):
    total = 0.0
    NS = 1
    while NS < int(NT):
        total = total + float(jnp.exp(-Free2(NS, NT)))
        NS = NS + 1
    fren = -float(jnp.log(total))
    return float(fren)


def Freesumqui(NT):
    total = 0.0
    NS = 1
    while NS < int(NT):
        total = total + float(jnp.exp(-Freequi(NS, NT)))
        NS = NS + 1
    fren = -float(jnp.log(total))
    return float(fren)


def findDfStar(params):
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
    NTlow = int(jnp.floor(NT))
    NThigh = NTlow + 1
    trueQ = trueQuiescent[NTlow] + (trueQuiescent[NThigh] - trueQuiescent[NTlow]) * (
        NT - NTlow
    )
    #!3return -(Freefluc(NT)-Freeflucqui(NT))
    return -(trueQ + Freefluc(NT) - Freeflucqui(NT))
