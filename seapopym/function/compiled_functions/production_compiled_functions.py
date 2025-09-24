"""This module contains the compiled (JIT) functions used by the production generator."""

from __future__ import annotations

import numpy as np
from numba import jit


@jit
def expand_dims(data: np.ndarray, dim_len: int) -> np.ndarray:
    """
    Expand array with a new cohort dimension, initializing first cohort with input data.

    Creates a new array with an additional dimension of length `dim_len` representing
    cohorts. The input data is placed in the first cohort (index 0), while all other
    cohorts are initialized to zero. This is used to convert primary production data
    into cohort-structured data for age-based modeling.

    Parameters
    ----------
    data : np.ndarray
        Input data to expand, typically primary production for a single timestep.
        Shape: [X, Y] (spatial dimensions)
    dim_len : int
        Length of the new cohort dimension to add.

    Returns
    -------
    expanded_data : np.ndarray
        Expanded data with new cohort dimension.
        Shape: [X, Y, Cohort] where Cohort has length `dim_len`

    Notes
    -----
    Only the first cohort (index 0) contains the input data. All other cohorts
    are initialized to 0.0.

    """
    expanded_data = np.full((*data.shape, dim_len), 0.0, dtype=np.float64)
    expanded_data[..., 0] = data
    return expanded_data


@jit
def ageing(production: np.ndarray, nb_timestep_by_cohort: np.ndarray) -> np.ndarray:
    """
    Age production across cohorts by transferring fractions to next age classes.

    Implements cohort aging by moving a fraction of production from each cohort to
    the next age class. The transfer fraction is 1/nb_timestep_by_cohort, representing
    the proportion that graduates to the next cohort at each timestep. The remaining
    production stays in the current cohort.

    Parameters
    ----------
    production : np.ndarray
        Production data to age across cohorts.
        Shape: [..., Cohort] where last dimension represents age classes.
    nb_timestep_by_cohort : np.ndarray
        Number of timesteps each cohort spans before aging to the next.
        Shape: [Cohort]. Higher values mean slower aging.

    Returns
    -------
    aged_production : np.ndarray
        Production after aging process, same shape as input.
        Each cohort contains: staying production + incoming production from younger cohort.

    Notes
    -----
    - The last (oldest) cohort receives transfers but doesn't transfer out
    - Transfer coefficient = 1.0 / nb_timestep_by_cohort[cohort]
    - First cohort only receives new production (no incoming transfers)

    """
    coefficient_except_last = 1.0 / nb_timestep_by_cohort[:-1]
    production_except_last = production[..., :-1]
    first_as_zero = np.zeros((*production.shape[:-1], 1), dtype=np.float64)
    growing = np.concatenate((first_as_zero, production_except_last * coefficient_except_last), axis=-1)
    staying_except_last = production_except_last * (1 - coefficient_except_last)
    staying = np.concatenate((staying_except_last, production[..., -1:]), axis=-1)
    return growing + staying


@jit
def production(
    primary_production: np.ndarray,
    mask_temperature: np.ndarray,
    timestep_number: np.ndarray,
    initial_production: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute total recruited production from primary production with cohort aging.

    Simulates production recruitment over time by processing primary production through
    cohort-based aging dynamics. At each timestep, production either gets recruited
    (based on recruitment mask) or ages into older cohorts. The function returns the
    total recruited production summed across all cohorts.

    Parameters
    ----------
    primary_production : np.ndarray
        Input primary production for each timestep.
        Shape: [T, X, Y] where T is time, X, Y are spatial dimensions.
    mask_temperature : np.ndarray
        Recruitment mask determining when production can be recruited.
        Shape: [T, X, Y, Cohort]. True values allow recruitment.
    timestep_number : np.ndarray
        Number of timesteps each cohort spans.
        Shape: [Cohort]. Controls aging rate between cohorts.
    initial_production : np.ndarray | None, default=None
        Pre-existing production in cohorts from previous simulation.
        Shape: [X, Y, Cohort]. If None, starts with zero initial conditions.

    Returns
    -------
    total_recruited : np.ndarray
        Total recruited production summed across all cohorts.
        Shape: [T, X, Y].

    Notes
    -----
    This version stores full cohort data during computation then sums at the end.
    For memory-optimized computation, use `production_space_optimized`.

    """
    output_recruited = np.empty(mask_temperature.shape)
    next_prepoduction = np.zeros(mask_temperature.shape[1:]) if initial_production is None else initial_production
    for timestep in range(primary_production.shape[0]):
        pre_production = expand_dims(primary_production[timestep], timestep_number.size)
        pre_production = pre_production + next_prepoduction
        if timestep < primary_production.shape[0] - 1:
            not_recruited = np.where(np.logical_not(mask_temperature[timestep]), pre_production, 0)
            next_prepoduction = ageing(not_recruited, timestep_number)
        recruited = np.where(mask_temperature[timestep], pre_production, 0)
        output_recruited[timestep] = recruited
    return np.sum(output_recruited, axis=-1)


@jit
def production_space_optimized(
    primary_production: np.ndarray,
    mask_temperature: np.ndarray,
    timestep_number: np.ndarray,
    initial_production: np.ndarray | None = None,
) -> np.ndarray:
    """
    Memory-optimized computation of total recruited production with cohort aging.

    Performs the same calculation as `production` but with reduced memory footprint.
    Instead of storing full cohort data then summing, this function computes the
    cohort sum at each timestep, reducing memory usage by a factor equal to the
    number of cohorts.

    Parameters
    ----------
    primary_production : np.ndarray
        Input primary production for each timestep.
        Shape: [T, X, Y] where T is time, X, Y are spatial dimensions.
    mask_temperature : np.ndarray
        Recruitment mask determining when production can be recruited.
        Shape: [T, X, Y, Cohort]. True values allow recruitment.
    timestep_number : np.ndarray
        Number of timesteps each cohort spans.
        Shape: [Cohort]. Controls aging rate between cohorts.
    initial_production : np.ndarray | None, default=None
        Pre-existing production in cohorts from previous simulation.
        Shape: [X, Y, Cohort]. If None, starts with zero initial conditions.

    Returns
    -------
    total_recruited : np.ndarray
        Total recruited production summed across all cohorts.
        Shape: [T, X, Y].

    Notes
    -----
    - Memory usage: O(T*X*Y) instead of O(T*X*Y*Cohort)
    - Performance: Slightly faster due to immediate summation and better cache usage
    - Functionally equivalent to `production` but more memory efficient

    """
    output_recruited = np.empty(mask_temperature.shape[:-1])
    next_prepoduction = np.zeros(mask_temperature.shape[1:]) if initial_production is None else initial_production
    for timestep in range(primary_production.shape[0]):
        pre_production = expand_dims(primary_production[timestep], timestep_number.size)
        pre_production = pre_production + next_prepoduction
        if timestep < primary_production.shape[0] - 1:
            not_recruited = np.where(np.logical_not(mask_temperature[timestep]), pre_production, 0)
            next_prepoduction = ageing(not_recruited, timestep_number)
        recruited = np.where(mask_temperature[timestep], pre_production, 0)
        output_recruited[timestep] = np.sum(recruited, axis=-1)
    return output_recruited


@jit
def production_export_initial(
    primary_production: np.ndarray,
    mask_temperature: np.ndarray,
    timestep_number: np.ndarray,
    initial_production: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute recruited production and export final pre-production state for continuity.

    Performs the same recruitment calculation as `production` but additionally returns
    the unrecruited production state at the end of the simulation. This final state
    can be used as initial conditions for subsequent simulation runs, enabling
    seamless chaining of simulations.

    Parameters
    ----------
    primary_production : np.ndarray
        Input primary production for each timestep.
        Shape: [T, X, Y] where T is time, X, Y are spatial dimensions.
    mask_temperature : np.ndarray
        Recruitment mask determining when production can be recruited.
        Shape: [T, X, Y, Cohort]. True values allow recruitment.
    timestep_number : np.ndarray
        Number of timesteps each cohort spans.
        Shape: [Cohort]. Controls aging rate between cohorts.
    initial_production : np.ndarray | None, default=None
        Pre-existing production in cohorts from previous simulation.
        Shape: [X, Y, Cohort]. If None, starts with zero initial conditions.

    Returns
    -------
    total_recruited : np.ndarray
        Total recruited production summed across all cohorts.
        Shape: [T, X, Y].
    final_preproduction : np.ndarray
        Unrecruited production state at the end of the simulation.
        Shape: [X, Y, Cohort]. Can be used as initial_production for next run.

    Notes
    -----
    The final pre-production state contains production that accumulated but wasn't
    recruited by the end of the simulation. This is essential for simulation
    continuity when running multi-year simulations in chunks.

    """
    output_recruited = np.empty(mask_temperature.shape)
    next_prepoduction = np.zeros(mask_temperature.shape[1:]) if initial_production is None else initial_production

    for timestep in range(primary_production.shape[0]):
        pre_production = expand_dims(primary_production[timestep], timestep_number.size)
        pre_production = pre_production + next_prepoduction
        if timestep < primary_production.shape[0] - 1:
            not_recruited = np.where(np.logical_not(mask_temperature[timestep]), pre_production, 0)
            next_prepoduction = ageing(not_recruited, timestep_number)
        recruited = np.where(mask_temperature[timestep], pre_production, 0)
        output_recruited[timestep] = recruited
    return (np.sum(output_recruited, axis=-1), next_prepoduction)


@jit
def production_export_preproduction(
    primary_production: np.ndarray,
    mask_temperature: np.ndarray,
    timestep_number: np.ndarray,
    initial_production: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute recruited production and export full pre-production time series.

    Performs recruitment calculation while tracking and exporting the complete
    pre-production (unrecruited) state at every timestep. This provides detailed
    diagnostic information about production dynamics, including cohort-specific
    accumulation and aging processes throughout the simulation.

    Parameters
    ----------
    primary_production : np.ndarray
        Input primary production for each timestep.
        Shape: [T, X, Y] where T is time, X, Y are spatial dimensions.
    mask_temperature : np.ndarray
        Recruitment mask determining when production can be recruited.
        Shape: [T, X, Y, Cohort]. True values allow recruitment.
    timestep_number : np.ndarray
        Number of timesteps each cohort spans.
        Shape: [Cohort]. Controls aging rate between cohorts.
    initial_production : np.ndarray | None, default=None
        Pre-existing production in cohorts from previous simulation.
        Shape: [X, Y, Cohort]. If None, starts with zero initial conditions.

    Returns
    -------
    total_recruited : np.ndarray
        Total recruited production summed across all cohorts.
        Shape: [T, X, Y].
    preproduction_timeseries : np.ndarray
        Complete pre-production state for all timesteps and cohorts.
        Shape: [T, X, Y, Cohort]. Includes final aged state at last timestep.

    Notes
    -----
    - Memory intensive: stores full [T, X, Y, Cohort] pre-production data
    - Useful for detailed analysis of recruitment timing and cohort dynamics
    - The last timestep contains the final aged pre-production state

    """
    output_recruited = np.empty(mask_temperature.shape)
    output_preproduction = np.empty(mask_temperature.shape, dtype=np.float64)
    next_prepoduction = np.zeros(mask_temperature.shape[1:]) if initial_production is None else initial_production

    for timestep in range(primary_production.shape[0]):
        pre_production = expand_dims(primary_production[timestep], timestep_number.size)
        pre_production = pre_production + next_prepoduction
        # if timestep < primary_production.shape[0] - 1:
        not_recruited = np.where(np.logical_not(mask_temperature[timestep]), pre_production, 0)
        next_prepoduction = ageing(not_recruited, timestep_number)
        recruited = np.where(mask_temperature[timestep], pre_production, 0)

        output_recruited[timestep] = recruited
        output_preproduction[timestep] = pre_production
    output_preproduction[-1, ...] = next_prepoduction
    return (np.sum(output_recruited, axis=-1), output_preproduction)
