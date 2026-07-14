"""
Disaggregate REMIND regional demand to country-level demand.

Stage 2 of the demand pipeline. Splits each (year, region, sector) row (Stage-1 output of
``import_REMIND_demand``) across constituent countries using a sector-weighted blend of named
proxies (single-country regions are a no-op): SSP population and GDP shares for most sectors,
and an HDD-weighted ``heating_demand`` proxy (population × heating degree-days) for
``heatpump``/``resistive``. Sectors without configured weights are folded into ``AC`` first (see
``fold_unconfigured_sectors_into_ac``). Demand attributed to unconfigured countries is excluded
(warned above 1% in the iampypsa function).
"""

import logging

import pandas as pd
from _helpers import configure_logging, mock_snakemake
from iampypsa.couplers.remind import read_region_map as get_region_mapping
from iampypsa.downscale import build_demand_proxy_from_dd, disaggregate_demand_to_country
from iampypsa.io import read_degree_days

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
    region_to_countries = get_region_mapping(source="model_region", target="country")
    configured_countries = set(snakemake.params.countries)

    dd = snakemake.params.degree_days
    hdd = read_degree_days(
        snakemake.input.hdd, dd_type="HDD",
        tlim_setpoint=dd["tlim_setpoint_hdd"], rcp=dd["rcp"], ssp=dd["ssp"],
    ).set_index(["iso2", "year"])
    cdd = read_degree_days(
        snakemake.input.cdd, dd_type="CDD",
        tlim_setpoint=dd["tlim_setpoint_cdd"], rcp=dd["rcp"], ssp=dd["ssp"],
    ).set_index(["iso2", "year"])
    proxies = {
        "population": pop,
        "gdp": gdp,
        "heating_demand": build_demand_proxy_from_dd(hdd, pop),
        "cooling_demand": build_demand_proxy_from_dd(cdd, pop),
    }

    years = {int(y) for y in snakemake.params.years}
    logger.info(
        "Disaggregating demand for %d scenario years via SSP/HDD/CDD proxy shares ...",
        len(years),
    )

    loads = sectoral_load[sectoral_load["year"].isin(years)]
    loads = fold_unconfigured_sectors_into_ac(loads, snakemake.params.sector_weights)
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
