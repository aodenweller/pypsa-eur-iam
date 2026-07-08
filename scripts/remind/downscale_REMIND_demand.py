"""
Disaggregate REMIND regional demand to country-level demand.

Stage 2 of the demand pipeline. Splits each (year, region, sector) row (Stage-1 output of
``import_REMIND_demand``) across constituent countries using a sector-weighted blend of SSP
population and GDP shares (single-country regions are a no-op). Sectors without configured
weights are folded into ``AC`` first (see ``fold_unconfigured_sectors_into_ac``). Demand
attributed to unconfigured countries is excluded (warned above 1% in the iampypsa function).
"""

import logging

import pandas as pd
from _helpers import configure_logging, mock_snakemake
from iampypsa.downscale.demand import disaggregate_demand_to_country
from iampypsa.transforms.mapping import read_region_map as get_region_mapping

logger = logging.getLogger(__name__)


def fold_unconfigured_sectors_into_ac(loads: pd.DataFrame, sector_weights: dict) -> pd.DataFrame:
    """Re-label any sector absent from ``sector_weights`` as ``AC`` and re-aggregate.

    A new REMIND FE sector without configured weights would otherwise crash downscaling or be
    silently dropped later — falls back to the general AC pool instead.
    """
    unconfigured = sorted(set(loads["sector"]) - set(sector_weights))
    if not unconfigured:
        return loads
    logger.info(
        "Sector(s) %s not present in sector_weights; folding into AC.", unconfigured
    )
    loads = loads.copy()
    loads.loc[loads["sector"].isin(unconfigured), "sector"] = "AC"
    return (
        loads.groupby(["year", "region", "sector", "unit"], as_index=False)["value"]
        .sum()
        .sort_values(["year", "region", "sector"])
    )


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
        snakemake.input.region_mapping, source="model_region", target="country"
    )
    configured_countries = set(snakemake.params.countries)

    years = {int(y) for y in snakemake.params.years}
    logger.info(
        "Disaggregating demand for %d scenario years via SSP population/GDP shares ...",
        len(years),
    )

    loads = sectoral_load[sectoral_load["year"].isin(years)]
    loads = fold_unconfigured_sectors_into_ac(loads, snakemake.params.sector_weights)
    proxies = {"population": pop, "gdp": gdp}
    result = disaggregate_demand_to_country(
        loads,
        region_to_countries,
        proxies,
        snakemake.params.sector_weights,
        configured_countries,
    )

    if missing := sorted(configured_countries - set(result["region"].unique())):
        country_to_region = {
            c: r for r, members in region_to_countries.items() for c in members
        }
        missing_regions = sorted({country_to_region.get(c, "unknown") for c in missing})
        raise ValueError(
            f"No demand data for countries {missing}. "
            f"REMIND regions {missing_regions} missing from the GDX export."
        )

    result.to_csv(snakemake.output.sectoral_load_country, index=False)
    logger.info(
        "Wrote %d country-level demand rows to %s",
        len(result),
        snakemake.output.sectoral_load_country,
    )
