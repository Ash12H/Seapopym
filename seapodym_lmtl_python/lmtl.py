"""This file contains the main class for the LMTL model."""

from dask.distributed import Client

from seapodym_lmtl_python.config import model_configuration, parse_configuration_file
from seapodym_lmtl_python.independent import (
    apply_coefficient_to_primary_production,
    average_temperature_by_fgroup,
    compute_cell_area,
    compute_daylength,
    compute_mortality_field,
    landmask_by_fgroup,
    mask_temperature_by_cohort_by_functional_group,
    min_temperature_by_cohort,
)


def main():

    # --------
    # 1. PARSE
    parameters = parse_configuration_file()

    # -----------------------------
    # 2. GENERATE THE CONFIGURATION
    configuration = model_configuration(parameters)

    # -------------------------------------
    # 3. INITIALIZE THE DASK CLIENT/CLUSTER
    # NOTE(Jules):
    # ? So many arguments can be passed to the Client class. This can be setup in the configuration file or with CLI.
    # CLI will override the configuration file.
    # ! But I am about to create a function rather than a script so all can be passed through the function arguments
    # as a class/dict/etc...
    client = Client()

    # ------------------------------
    # 4. RUN THE INDEPENDENT PROCESS
    # All None arguments are comming from the configuration dataset
    # ! This can be done in a function and return a dataset that contains all the computed forcings.
    landmask = client.submit(landmask_by_fgroup, None, None, None)
    day_length = client.submit(compute_daylength, None, None, None)
    avg_temperature = client.submit(
        average_temperature_by_fgroup,
        day_length,
        landmask,
        None,
        None,
        None,
    )
    pre_production = client.submit(
        apply_coefficient_to_primary_production, None, None, None
    )
    min_temperature = client.submit(min_temperature_by_cohort, None, None, None)
    mask_temperature = client.submit(
        mask_temperature_by_cohort_by_functional_group,
        min_temperature,
        avg_temperature,
    )
    cell_area = client.submit(compute_cell_area, None, None)
    mortality = client.submit(compute_mortality_field, avg_temperature, None, None)

    # ----------------------------------------
    # 5. RUN THE DEPENDENT PROCESS (map_block)

    # ------------------
    # 6. COMPUTE BIOMASS

    # -------------------
    # 7. SAVE THE OUTPUTS

    # -------------------
    # 8. CLOSE THE CLIENT
