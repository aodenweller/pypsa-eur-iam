# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build hourly water heat demand time series from daily heat demand.

This uses the intraday profiles from BDEW. However, here we also use the space heating profile for water heating,
because the water heating profile is just flat, which is not realistic. 

The hourly water heat demand profile is distributed by population to clustered onshore regions.
"""
# %%
import logging
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr

from scripts._helpers import (
    configure_logging,
    generate_periodic_profiles,
    get_snapshots,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

# %%

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_hourly_water_heat_demand_REMIND",
            clusters=4,
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )

    intraday_profiles = pd.read_csv(snakemake.input.heat_profile, index_col=0)
    
    clustered_pop_layout = pd.read_csv(
        snakemake.input.clustered_pop_layout, index_col=0
    )

    # Create weekly profile from weekday and weekend profiles of residential and services space heating
    weekday = list(intraday_profiles["residential space weekday"] + intraday_profiles["services space weekday"] / 2)
    weekend = list(intraday_profiles["residential space weekend"] + intraday_profiles["services space weekend"] / 2)
    
    # Create weekly profile
    weekly_profile = weekday * 5 + weekend * 2
    
    # Make periodic yearly profile
    intraday_year_profile = generate_periodic_profiles(
        snapshots.tz_localize("UTC"),
        nodes=clustered_pop_layout.index,
        weekly_profile=weekly_profile,
    )
    
    # Weight by clustered_pop_layout["fraction"]
    intraday_year_profile = (
        intraday_year_profile * clustered_pop_layout["fraction"]
    )

    # Export
    ds = intraday_year_profile
    ds.index.name = "snapshots"
    ds = ds.to_xarray()
    ds.to_netcdf(snakemake.output.water_heat_demand)
