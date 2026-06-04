"""
Retrieve SSP population and GDP projections from the IIASA scenario database.

Calls the IIASA SSP REST API directly via httpx (no ixmp4/pyam dependency).
Saves country-level population and GDP|PPP time series as CSV files used by
``downscale_REMIND_demand`` to disaggregate REMIND regional demand.

Country names returned by the API (English full names) are mapped to ISO-2 codes
using the ``country_converter`` package. Data is at 5-year intervals matching
REMIND's time resolution — no interpolation is applied.

API pagination
--------------
The IIASA API caps responses at 25 000 rows per request. Without pagination this
silently truncates ~30 countries (GB, SE, CH, TR, …). The script pages through
results using ``limit``/``offset`` query parameters until it receives a page
smaller than the page size, indicating the last page.

Outputs
-------
- ``population.csv``: columns [iso2, year, value] (population in millions)
- ``gdp.csv``: columns [iso2, year, value] (GDP|PPP in billion USD 2005)
"""

import logging

import country_converter as coco
import httpx
import pandas as pd
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)

IIASA_URL = "https://ixmp4.ece.iiasa.ac.at/v1/ssp/iamc/datapoints/"
IIASA_PARAMS = {"join_parameters": "true", "join_runs": "true", "table": "true"}
_PAGE_SIZE = 20_000  # stay below the 25k hard cap


def _fetch_variable(variable: str) -> pd.DataFrame:
    """Download one IAMC variable from the IIASA SSP platform, paging through all results."""
    logger.info("Fetching '%s' from IIASA SSP API …", variable)

    all_data = []
    columns = None
    offset = 0

    while True:
        resp = httpx.patch(
            IIASA_URL,
            params={**IIASA_PARAMS, "limit": _PAGE_SIZE, "offset": offset},
            json={"variable": {"name": variable}},
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()["results"]

        if columns is None:
            columns = payload["columns"]

        page = payload["data"]
        all_data.extend(page)
        offset += len(page)
        logger.info("  page: %d rows (cumulative: %d)", len(page), offset)

        if len(page) < _PAGE_SIZE:
            break  # last or only page

    df = pd.DataFrame(all_data, columns=columns)
    logger.info("  total received: %d rows", len(df))
    return df


def _retrieve_variable(
    variable: str,
    model: str,
    scenario: str,
    label: str,
) -> pd.DataFrame:
    """Fetch variable, filter to model/scenario, map regions to ISO-2."""
    df = _fetch_variable(variable)
    df = df[(df["model"] == model) & (df["scenario"] == scenario)].copy()
    logger.info("After filtering to %s / %s: %d rows", model, scenario, len(df))

    cc = coco.CountryConverter()
    df["iso2"] = cc.pandas_convert(
        pd.Series(df["region"]), to="ISO2", not_found=None
    )
    n_before = len(df)
    df = df.dropna(subset=["iso2"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.debug("Dropped %d non-country entries for %s", n_dropped, label)

    result = (
        df[["iso2", "step_year", "value"]]
        .rename(columns={"step_year": "year"})
        .groupby(["iso2", "year"], as_index=False)["value"]
        .sum()
        .sort_values(["iso2", "year"])
        .reset_index(drop=True)
    )
    logger.info("Retrieved %s for %d country-year combinations.", label, len(result))
    return result


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "retrieve_ssp_data",
            configfiles="config/config.remind.yaml",
        )

    configure_logging(snakemake)

    params = snakemake.params
    scenario = params.ssp_scenario

    population = _retrieve_variable(
        variable="Population",
        model=params.ssp_population_model,
        scenario=scenario,
        label="population",
    )
    population.to_csv(snakemake.output.population, index=False)
    logger.info("Wrote population data to %s", snakemake.output.population)

    gdp = _retrieve_variable(
        variable="GDP|PPP",
        model=params.ssp_gdp_model,
        scenario=scenario,
        label="GDP|PPP",
    )
    gdp.to_csv(snakemake.output.gdp, index=False)
    logger.info("Wrote GDP data to %s", snakemake.output.gdp)
