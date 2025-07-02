# -*- coding: utf-8 -*-

# %%

import logging

import pandas as pd
from _helpers import (
    configure_logging,
    get_region_mapping,
    mock_snakemake,
    read_remind_data,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_load",
            scenario="TEST",
            iteration="1",
            year="2050",
            # configfiles="resources/PyPSA_PkBudg1000_DEU_rm350_pypsa202504_EV_heatingExport_2025-06-20_14.49.59/i1/config.remind_scenario.yaml",
        )

    configure_logging(snakemake)

    # %%
    # Load original load timeseries from PyPSA-EUR
    load = pd.read_csv(snakemake.input["load_timeseries"], index_col=0)

    # Load REMIND sectoral load
    demand_sector = read_remind_data(
        snakemake.input["remind_data"],
        "p32_load_sector",
        rename_columns={"ttot": "year", "all_regi": "region", "loadPy32": "sector"},
    ).query("year == @snakemake.wildcards.year")
    demand_sector["value"] *= 1e6 * 8760  # Convert from TWa to MWh

    # Add sectoral loads to AC if sector coupling is not enabled
    sc_settings = snakemake.params["sector_coupling"]

    if not sc_settings["EVs"]["enable"]:
        demand_sector.loc[
            demand_sector["sector"] == "AC", "value"
        ] += demand_sector.loc[demand_sector["sector"] == "EVs", "value"].values[0]
        demand_sector.query("sector != 'EVs'", inplace=True)

    if not sc_settings["heating"]["enable"]:
        demand_sector.loc[demand_sector["sector"] == "AC", "value"] += (
            demand_sector.loc[demand_sector["sector"] == "heatpump", "value"]
            + demand_sector.loc[demand_sector["sector"] == "resistive", "value"]
        )
        demand_sector.query(
            "sector not in ['heatpump', 'resistive']",
            inplace=True,
        )

    # If any sectoral load is negative set to zero and write warning mentioning the sector
    if (demand_sector["value"] < 0).any():
        negative_sectors = demand_sector.query("value < 0")["sector"].unique()
        logger.warning(
            f"Negative sectoral demand detected for sectors {negative_sectors}. "
            "Setting negative values to zero."
        )
        # Set negative values to zero
        demand_sector.loc[demand_sector["value"] < 0, "value"] = 0

    # Save sectoral demand to CSV
    demand_sector.to_csv(snakemake.output["sectoral_load"])

    # Get region mapping
    region_mapping = (
        pd.DataFrame.from_dict(
            get_region_mapping(
                snakemake.input["region_mapping"],
                source="PyPSA-EUR",
                target="REMIND-EU",
            ),
            "index",
        )
        .reset_index()
        .rename(columns={"index": "PyPSA-EUR", 0: "REMIND-EU"})
    )

    # Calculate load from PyPSA-EUR load-timeseries for REMIND-EU regions
    regional_load = (
        region_mapping.join(load.sum(axis="rows").rename("annual_load"), on="PyPSA-EUR")
        .groupby("REMIND-EU")["annual_load"]
        .sum()
    )
    regional_load = regional_load[regional_load > 0]

    # Factor by which PyPSA-EUR loads have to be scaled to match REMIND-EU demand
    load_scaling_factor = (
        (demand_sector.query("sector == 'AC'").value.values[0] / regional_load).rename(
            "load_scaling_factor"
        )
    ).fillna(0)
    load_scaling_factor = region_mapping.join(
        load_scaling_factor,
        on="REMIND-EU",
        how="inner",
    ).set_index("PyPSA-EUR")["load_scaling_factor"]

    load_scaling_factor.to_csv(snakemake.output["load_scaling_factor"])

# %%
