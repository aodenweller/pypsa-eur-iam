"""
Read REMIND hydro capacity and generation targets and export them for use in PyPSA-Eur.

Hydro capacity is taken from the general ``capacity`` symbol filtered to the hydro technology
(GDX: ``vm_cap``; IAMC: ``Cap|Electricity|Hydro``) — there is no dedicated hydro-capacity symbol.
Generation comes from ``hydro_generation`` (GDX: ``p32_prodSeHydro`` in TWa; IAMC:
``SE|Electricity|Hydro`` in EJ/yr), converted to MWh-per-year by the symbol layer. Merges them,
derives a capacity factor, and filters to REMIND regions overlapping the configured countries.
"""

import logging

import pandas as pd
from _helpers import (
    configure_logging,
    mock_snakemake,
)
from iampypsa.couplers.remind import read_region_map as get_region_mapping
from iampypsa.io import RemindLoader
from iampypsa.io.remind_symbols import load_frame, load_spec, load_symbol_specs

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
        "Loading REMIND hydro capacity and generation data from the iampypsa symbol config..."
    )

    loader = RemindLoader(snakemake.input["remind_data"])
    symbols = load_symbol_specs(backend=loader.backend)

    # Hydro capacity: the `capacity` symbol (vm_cap / Cap|Electricity|Hydro), filtered to hydro.
    # The defensive groupby below sums out any extra dims (e.g. GDX vm_cap's `rlf` grades).
    capacity = load_spec(loader, symbols["capacity"])
    hydro_capacity = capacity[capacity["technology"] == "hydro"]
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

    region_mapping = get_region_mapping(source="country", target="model_region")
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
