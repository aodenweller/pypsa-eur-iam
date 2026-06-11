"""Disaggregate REMIND regional demand to country-level demand.

Thin wrapper over ``rpycpl.downscale.disaggregate_demand_to_country``: splits each
(year, region, sector) row across constituent countries using a sector-weighted blend of
SSP population and GDP shares (single-country regions are a no-op). Demand attributed to
unconfigured countries is excluded (warned above 1% in the rpycpl function).
"""

import logging

import pandas as pd
from _helpers import configure_logging, get_region_mapping, mock_snakemake
from rpycpl.downscale.demand import disaggregate_demand_to_country

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
    sectoral_load = sectoral_load[sectoral_load["year"].isin(years)]
    logger.info("Disaggregating demand for %d scenario years via rpycpl ...", len(years))

    result = disaggregate_demand_to_country(
        sectoral_load,
        region_to_countries,
        pop,
        gdp,
        snakemake.params.sector_weights,
        configured_countries,
    )

    if missing := sorted(configured_countries - set(result["region"].unique())):
        country_to_region = {c: r for r, members in region_to_countries.items() for c in members}
        missing_regions = sorted({country_to_region.get(c, "unknown") for c in missing})
        raise ValueError(
            f"No demand data for countries {missing}. "
            f"REMIND regions {missing_regions} missing from the GDX export."
        )

    result.to_csv(snakemake.output.sectoral_load_country, index=False)
    logger.info("Wrote %d country-level demand rows to %s", len(result), snakemake.output.sectoral_load_country)
