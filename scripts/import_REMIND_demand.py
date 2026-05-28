
"""
Read sectoral electricity demand from REMIND and export it for use in PyPSA-Eur.

Reads the REMIND variable ``v32_load_sector`` (unit: TWa), converts it to annual
MWh per (year, region, sector), and filters to REMIND regions that overlap with the
configured PyPSA-Eur countries.

Outputs
-------
- ``sectoral_load.csv``: long-format table with columns year, region, sector, value (MWh_el).
"""

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
        snakemake = mock_snakemake(
            "import_REMIND_demand",
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)

    logger.info("Loading REMIND demand data...")
    demand = read_remind_data(
        snakemake.input["remind_data"],
        "v32_load_sector",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "loadPy32": "sector",
            "level": "value",
        },
    )

    demand = demand[["year", "region", "sector", "value"]].copy()

    # Convert TWa to annual MWh.
    demand["value"] = demand["value"] * 1e6 * 8760
    demand["unit"] = "MWh_el"

    # Restrict to REMIND regions that overlap with configured PyPSA-Eur countries.
    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"],
        source="PyPSA-EUR",
        target="REMIND-EU",
    )
    mapped_regions = (
        pd.Series(region_mapping, name="region")
        .explode()
        .dropna()
        .unique()
    )
    demand = demand[demand["region"].isin(mapped_regions)]

    # Use one row per (year, region, sector) to keep downstream handling deterministic.
    demand = (
        demand.groupby(
            ["year", "region", "sector", "unit"],
            as_index=False,
            observed=False,
        )["value"]
        .sum()
        .sort_values(["year", "region", "sector"])
    )

    demand.to_csv(snakemake.output["sectoral_load"], index=False)
    logger.info(
        "Wrote %s rows of REMIND demand data to %s",
        len(demand),
        snakemake.output["sectoral_load"],
    )
