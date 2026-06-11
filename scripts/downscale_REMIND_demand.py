"""Disaggregate REMIND regional demand to country-level demand.

Stage 2 of the demand pipeline. Thin wrapper over ``RemindEurAdapter.downscale_country_demand``:
splits each (year, region, sector) row (Stage-1 output of ``import_REMIND_demand``) across
constituent countries using a sector-weighted blend of SSP population and GDP shares
(single-country regions are a no-op). Demand attributed to unconfigured countries is excluded
(warned above 1% in the rpycpl function).
"""

import logging

import pandas as pd
from _helpers import configure_logging, mock_snakemake
from remind.adapter_remind_eur import RemindEurAdapter
from rpycpl.io import RemindLoader
from rpycpl.io.remind_symbols import load_symbol_specs
from rpycpl.transforms.mapping import read_region_map as get_region_mapping

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "downscale_REMIND_demand",
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)

    sectoral_load = pd.read_csv(snakemake.input.sectoral_load)
    pop = pd.read_csv(snakemake.input.population).set_index(["iso2", "year"])
    gdp = pd.read_csv(snakemake.input.gdp).set_index(["iso2", "year"])
    region_to_countries = get_region_mapping(
        snakemake.input.region_mapping, source="REMIND-EU", target="PyPSA-EUR"
    )
    configured_countries = set(snakemake.params.countries)

    years = snakemake.params.years
    logger.info("Disaggregating demand for %d scenario years via the rpycpl adapter ...", len(years))

    # The loader is unused for Stage 2 (regional demand is passed in via ``regional=``), but the
    # adapter binds the downscaling inputs (region map, SSP proxies, sector weights, horizons).
    adapter = RemindEurAdapter(
        loader=RemindLoader(snakemake.input["remind_data"]),
        symbols=load_symbol_specs(),
        region_map=region_to_countries,
        config={
            "sector_weights": snakemake.params.sector_weights,
            "countries": list(configured_countries),
            "planning_horizons": list(years),
        },
        remind_regions=sorted(region_to_countries),
        reference_data={"population": pop, "gdp": gdp},
    )
    result = adapter.downscale_country_demand(regional=sectoral_load)

    if missing := sorted(configured_countries - set(result["region"].unique())):
        country_to_region = {c: r for r, members in region_to_countries.items() for c in members}
        missing_regions = sorted({country_to_region.get(c, "unknown") for c in missing})
        raise ValueError(
            f"No demand data for countries {missing}. "
            f"REMIND regions {missing_regions} missing from the GDX export."
        )

    result.to_csv(snakemake.output.sectoral_load_country, index=False)
    logger.info("Wrote %d country-level demand rows to %s", len(result), snakemake.output.sectoral_load_country)
