import numpy as np
import jax.numpy as jnp
from jax.experimental.ode import odeint
from interpax import interp1d as jinterp1d


def intSchneider(t, Ndot, Ndot0, N_0, G_C):
    global N_dot_func

    t = jnp.asarray(t, dtype=jnp.float64)
    Ndot = jnp.asarray(Ndot, dtype=jnp.float64)
    N_scale = N_0 + Ndot0 * t[-1]
    Ndot = Ndot / N_scale

    # Prepend an initial datapoint at t=0
    t = jnp.concatenate([jnp.array([0.0]), t])
    Ndot = jnp.concatenate([jnp.array([Ndot0], dtype=jnp.float64), Ndot])

    # Append a final data point
    t2 = jnp.concatenate([t, jnp.array([t[-1] * 5.0])])
    Ndot = jnp.concatenate([Ndot, jnp.array([Ndot[-1]])])

    N_dot_func = jinterp1d(t2, Ndot, kind="cubic")

    # Solve Schneider ODEs
    phiSc0 = jnp.array([0.0, 0.0, 0.0, 8.0 * jnp.pi * N_0 / N_scale])
    sol = odeint(Schneider, phiSc0, t, G_C)

    sol = np.asarray(sol[1:])  # Remove row containing t=0
    sol = sol * N_scale
    return sol

def Schneider(phiSc, t, G_C):
    return jnp.array(
        [
            G_C * phiSc[1],
            G_C * phiSc[2],
            G_C * phiSc[3],
            8.0 * jnp.pi * jnp.abs(N_dot_func(t)),
        ]
    )
