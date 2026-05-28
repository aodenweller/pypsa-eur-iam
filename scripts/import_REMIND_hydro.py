
"""
Read REMIND hydro capacity and generation targets and export them for use in PyPSA-Eur.

Reads ``p32_hydroCapacity`` (TW -> MW) and ``p32_hydroGeneration`` (TWa -> MWh/year)
from the REMIND GDX file, merges them, derives a capacity factor, and filters to
REMIND regions that overlap with the configured PyPSA-Eur countries.

Outputs
-------
- ``hydro_targets.csv``: table with columns year, region, hydro_capacity_mw,
  hydro_generation_mwh, hydro_capacity_factor.
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
            "import_REMIND_hydro",
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)

    logger.info("Loading REMIND hydro capacity and generation data...")

    hydro_capacity = read_remind_data(
        snakemake.input["remind_data"],
        "p32_hydroCapacity",
        rename_columns={"ttot": "year", "all_regi": "region"},
    )
    hydro_generation = read_remind_data(
        snakemake.input["remind_data"],
        "p32_hydroGeneration",
        rename_columns={"ttot": "year", "all_regi": "region"},
    )

    # Aggregate defensively in case additional dimensions are present.
    hydro_capacity = (
        hydro_capacity.groupby(["year", "region"], as_index=False, observed=False)[
            "value"
        ]
        .sum()
        .rename(columns={"value": "hydro_capacity_mw"})
    )
    hydro_generation = (
        hydro_generation.groupby(["year", "region"], as_index=False, observed=False)[
            "value"
        ]
        .sum()
        .rename(columns={"value": "hydro_generation_mwh"})
    )

    # Unit conversions from REMIND outputs.
    hydro_capacity["hydro_capacity_mw"] *= 1e6  # TW -> MW
    hydro_generation["hydro_generation_mwh"] *= 1e6 * 8760  # TWa -> MWh/year

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

    hydro = hydro_capacity.merge(hydro_generation, on=["year", "region"], how="outer")
    hydro = hydro[hydro["region"].isin(mapped_regions)].copy()

    hydro["hydro_capacity_mw"] = hydro["hydro_capacity_mw"].fillna(0.0)
    hydro["hydro_generation_mwh"] = hydro["hydro_generation_mwh"].fillna(0.0)

    denom = hydro["hydro_capacity_mw"] * 8760
    hydro["hydro_capacity_factor"] = 0.0
    valid = denom > 0
    hydro.loc[valid, "hydro_capacity_factor"] = (
        hydro.loc[valid, "hydro_generation_mwh"] / denom[valid]
    )

    hydro = hydro.sort_values(["year", "region"]).reset_index(drop=True)
    hydro.to_csv(snakemake.output["hydro_targets"], index=False)

    logger.info(
        "Wrote %s rows of REMIND hydro targets to %s",
        len(hydro),
        snakemake.output["hydro_targets"],
    )
