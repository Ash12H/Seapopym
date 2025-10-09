"""This module contains the compiled (JIT) functions used by the biomass generator."""

from __future__ import annotations

import numpy as np
from numba import jit


@jit()
def biomass_euler_explicite(
    recruited: np.ndarray,
    mortality: np.ndarray,
    initial_conditions: np.ndarray | None,
    delta_time: np.floating | np.integer,
) -> np.ndarray:
    """Euler explicite approximating of dB/dt = R - lambda B."""
    initial_conditions = (
        np.zeros(recruited[:, 0, ...].shape, dtype=np.float64) if initial_conditions is None else initial_conditions
    )
    biomass = np.zeros(recruited.shape)
    biomass[:, 0, ...] = (
        initial_conditions + delta_time * recruited[:, 0, ...] - delta_time * mortality[:, 0, ...] * initial_conditions
    )

    for timestep in range(1, recruited.shape[1]):
        biomass[:, timestep, ...] = (
            biomass[:, timestep - 1, ...]
            + delta_time * recruited[:, timestep, ...]
            - delta_time * mortality[:, timestep, ...] * biomass[:, timestep - 1, ...]
        )
    return biomass


@jit()
def biomass_euler_implicite(
    recruited: np.ndarray,
    mortality: np.ndarray,
    initial_conditions: np.ndarray | None,
    delta_time: np.floating | np.integer,
) -> np.ndarray:
    """Euler implicite approximating of dB/dt = R - lambda B."""
    initial_conditions = (
        np.zeros(recruited[:, 0, ...].shape, dtype=np.float64) if initial_conditions is None else initial_conditions
    )
    biomass = np.zeros(recruited.shape)
    biomass[:, 0, ...] = (initial_conditions + delta_time * recruited[:, 0, ...]) / (
        1 + delta_time * mortality[:, 0, ...]
    )

    for timestep in range(1, recruited.shape[1]):
        biomass[:, timestep, ...] = (biomass[:, timestep - 1, ...] + delta_time * recruited[:, timestep, ...]) / (
            1 + delta_time * mortality[:, timestep, ...]
        )
    return biomass
