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
    P = x[0]
    B = x[1]
    wi = wfun(phi, df, P, B, NS)
    sum1 = jnp.sum(wi)
    sum2 = jnp.sum(wi * df)
    return jnp.array([sum1 - 1.0, sum2 - P])


def Free2(Ns, NT):
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
    x0 = NSprevious
    res = optimize.minimize(Free2, x0, method="Nelder-Mead", args=(NT))
    # res=scipy.optimize.minimize_scalar(Free2, bounds=(1,0.999999*NT), args=(NT), method='bounded')
    print(res)
    return float(res.fun)


def Freefluc(NT):
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
    return -Freefluc(1.0 * NT)
