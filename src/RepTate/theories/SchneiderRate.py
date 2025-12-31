import numpy as np
import jax.numpy as jnp
from jax.experimental.ode import odeint
from interpax import interp1d as jinterp1d


def intSchneider(t, Ndot, Ndot0, N_0, G_C):
    """Integrate Schneider rate equations for chain scission dynamics.

    Solves the Schneider model ODEs to compute the evolution of molecular weight
    distribution moments during polymer degradation or chain scission processes.
    The model tracks how chain breaking affects the number of entanglements.

    Args:
        t (np.ndarray): Time values at which scission rate is evaluated (s)
        Ndot (np.ndarray): Chain scission rate dN/dt at each time point (1/s)
        Ndot0 (float): Initial scission rate at t=0 (1/s)
        N_0 (float): Initial number of entanglements per chain
        G_C (float): Characteristic rate parameter for stress relaxation (1/s)

    Returns:
        np.ndarray: Solution array (len(t) x 4) containing time evolution of:
            - phi_0: Zeroth moment (total chain concentration)
            - phi_1: First moment (number-average chain length)
            - phi_2: Second moment (weight-average chain length)
            - phi_3: Third moment (related to entanglement density)
    """
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
    """Compute time derivatives for Schneider chain scission model ODEs.

    Defines the right-hand side of the Schneider differential equations that
    describe how molecular weight distribution moments evolve during polymer
    degradation. Each moment couples to higher moments and the scission rate.

    Args:
        phiSc (jnp.ndarray): State vector [phi_0, phi_1, phi_2, phi_3] containing
                             the four moments of the molecular weight distribution
        t (float): Current time value (s)
        G_C (float): Characteristic rate parameter for stress relaxation (1/s)

    Returns:
        jnp.ndarray: Time derivatives [dphi_0/dt, dphi_1/dt, dphi_2/dt, dphi_3/dt]
                     where each derivative couples to the next higher moment and
                     the scission rate interpolated from N_dot_func
    """
    return jnp.array(
        [
            G_C * phiSc[1],
            G_C * phiSc[2],
            G_C * phiSc[3],
            8.0 * jnp.pi * jnp.abs(N_dot_func(t)),
        ]
    )
