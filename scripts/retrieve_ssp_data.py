"""
Retrieve SSP population and GDP projections from the IIASA scenario database.

Uses the pyam package to download data from the IIASA SSP database and saves
country-level population and GDP|PPP time series as CSV files. These are used
by ``downscale_REMIND_demand`` to disaggregate REMIND regional demand to
individual countries.

Country names returned by pyam (English full names) are mapped to ISO-2 codes
using the ``country_converter`` package. Data is at 5-year intervals matching
REMIND's time resolution — no interpolation is applied.

Outputs
-------
- ``ssp_population.csv``: columns [iso2, year, value] (population in millions)
- ``ssp_gdp.csv``: columns [iso2, year, value] (GDP|PPP in billion USD 2005)
"""

import logging

import country_converter as coco
import pandas as pd
from _helpers import configure_logging, mock_snakemake
from pyam import read_ixmp4

logger = logging.getLogger(__name__)


def _retrieve_variable(
    database: str,
    model: str,
    scenario: str,
    variable: str,
    label: str,
) -> pd.DataFrame:
    """Download one variable from IIASA, return long DataFrame with [iso2, year, value]."""
    logger.info("Retrieving %s from IIASA (%s / %s)...", variable, model, scenario)
    df = read_ixmp4(
        database,
        model=model,
        scenario=scenario,
        variable=variable,
    )
    data = df.data[["region", "year", "value"]].copy()
    cc = coco.CountryConverter()
    data["iso2"] = cc.pandas_convert(
        pd.Series(data["region"]), to="ISO2", not_found=None
    )
    # Drop entries that could not be mapped to an ISO-2 country (regions/aggregates)
    n_before = len(data)
    data = data.dropna(subset=["iso2"])
    n_dropped = n_before - len(data)
    if n_dropped:
        logger.debug("Dropped %d non-country entries for %s", n_dropped, label)
    data = (
        data[["iso2", "year", "value"]]
        .groupby(["iso2", "year"], as_index=False)["value"]
        .sum()
        .sort_values(["iso2", "year"])
    )
    logger.info("Retrieved %s for %d country-year combinations.", label, len(data))
    return data


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "retrieve_ssp_data",
            configfiles="config/config.remind.yaml",
        )

    configure_logging(snakemake)

    params = snakemake.params
    database = "ssp"
    scenario = params.ssp_scenario

    population = _retrieve_variable(
        database,
        model=params.ssp_population_model,
        scenario=scenario,
        variable="Population",
        label="population",
    )
    population.to_csv(snakemake.output.population, index=False)
    logger.info("Wrote population data to %s", snakemake.output.population)

    gdp = _retrieve_variable(
        database,
        model=params.ssp_gdp_model,
        scenario=scenario,
        variable="GDP|PPP",
        label="GDP|PPP",
    )
    gdp.to_csv(snakemake.output.gdp, index=False)
    logger.info("Wrote GDP data to %s", snakemake.output.gdp)
