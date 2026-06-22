"""
Read REMIND hydro capacity and generation targets and export them for use in PyPSA-Eur.

Reads ``hydro_capacity`` and ``hydro_generation`` from the REMIND output (GDX or IAMC .mif,
auto-detected) via the central symbol config — GDX: ``p32_hydroCapacity`` (TW) /
``p32_hydroGeneration`` (TWa); IAMC: ``Cap|Electricity|Hydro`` (GW) / ``SE|Electricity|Hydro``
(EJ/yr). Both are converted to MW / MWh-per-year by ``load_frame``. Merges them, derives a
capacity factor, and filters to REMIND regions that overlap with the configured PyPSA-Eur
countries.
"""

import logging

import pandas as pd
from _helpers import (
    configure_logging,
    mock_snakemake,
)
from rpycpl.io import RemindLoader
from rpycpl.io.remind_symbols import load_frame, load_symbol_specs
from rpycpl.transforms.mapping import read_region_map as get_region_mapping

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

    logger.info(
        "Loading REMIND hydro capacity and generation data from the rpycpl symbol config..."
    )

    loader = RemindLoader(snakemake.input["remind_data"])
    symbols = load_symbol_specs(backend=loader.backend)

    hydro_capacity = load_frame(loader, symbols["hydro_capacity"])  # TW -> MW
    hydro_generation = load_frame(
        loader, symbols["hydro_generation"]
    )  # TWa -> MWh/year

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

    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
    )
    mapped_regions = (
        pd.Series(region_mapping, name="region").explode().dropna().unique()
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
