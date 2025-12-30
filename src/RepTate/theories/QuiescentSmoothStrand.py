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


def wholeLandscape(NT, E0_param, mus_param, Kappa0_param):
    global E0, mus, Kappa0, thetaMin, arsq, ar

    E0 = E0_param
    mus = mus_param
    Kappa0 = Kappa0_param

    thetaMin = 1e-300

    # a_r
    arsq = 9.0 / 16.0 * jnp.pi
    ar = float(jnp.sqrt(arsq))

    return Freeflucqui(NT)


def Freequi(NS, NT):
    LL = NT / NS
    kappa = Kappa0 + 1.0 / LL**2

    FF = (
        -0.5 * (NS - 1) * jnp.log(2.0 * jnp.pi / kappa) + 0.5 * jnp.log(NS) - NT * E0
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


def Free1qui(NT):
    res = _minimize_scalar(Freequi, bounds=(0.00001, 0.999999 * NT), args=(NT,))
    # print res
    return float(res.fun)


def Freeflucqui(NT):
    res = _minimize_scalar(Freequi, bounds=(0.0000001, 0.999999 * NT), args=(NT,))
    nsmid = float(res.x[0])
    ##second derivative
    d2fdn2 = (
        Freequi(nsmid + 0.1, NT) + Freequi(nsmid - 0.1, NT) - 2 * Freequi(nsmid, NT)
    ) / 0.01
    fNT = float(res.fun) + float(jnp.log(d2fdn2 / 2 / jnp.pi))
    return float(fNT)
