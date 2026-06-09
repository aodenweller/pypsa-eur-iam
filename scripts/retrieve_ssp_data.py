"""
Retrieve SSP population and GDP projections from the IIASA scenario database.

Calls the IIASA SSP REST API directly via httpx (no ixmp4/pyam dependency).
Saves country-level population and GDP|PPP time series as CSV files used by
``downscale_REMIND_demand`` to disaggregate REMIND regional demand.

Country names returned by the API (English full names) are mapped to ISO-2 codes
using the ``country_converter`` package. Data is at 5-year intervals matching
REMIND's time resolution — no interpolation is applied.

API endpoint
------------
Uses the IIASA ixmp4 ``/iamc/datapoints/tabulate`` PATCH endpoint with JSON
filters. Pagination via ``limit``/``offset`` query parameters is supported but
rarely needed (GDP|PPP and Population return well under 25 000 rows each).

Outputs
-------
- ``population.csv``: columns [iso2, year, value] (population in millions)
- ``gdp.csv``: columns [iso2, year, value] (GDP|PPP in billion USD 2010)
"""

import logging

import country_converter as coco
import httpx
import pandas as pd
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)

IIASA_URL = "https://ixmp4.ece.iiasa.ac.at/v1/ssp/iamc/datapoints/tabulate"
_PAGE_SIZE = 20_000


def _fetch_variable(variable: str, model: str, scenario: str) -> pd.DataFrame:
    """Download one IAMC variable from the IIASA SSP platform, paging through all results."""
    logger.info(
        "Fetching '%s' (%s / %s) from IIASA SSP API …", variable, model, scenario
    )

    body = {
        "join_parameters": True,
        "join_runs": True,
        "join_run_id": False,
        "variable": {"name__like": variable},
        "run": {"default_only": True},
        "model": {"name__like": model},
        "scenario": {"name__like": scenario},
    }

    all_rows = []
    columns = None
    offset = 0

    while True:
        resp = httpx.patch(
            IIASA_URL,
            params={"limit": _PAGE_SIZE, "offset": offset},
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()

        results = payload["results"]
        if columns is None:
            columns = results["columns"]

        page = results["data"]
        all_rows.extend(page)
        offset += len(page)
        total = payload.get("total", "?")
        logger.info("  page: %d rows (cumulative: %d / %s)", len(page), offset, total)

        if len(page) < _PAGE_SIZE:
            break

    df = pd.DataFrame(all_rows, columns=columns)
    logger.info("  total received: %d rows", len(df))
    return df


def _retrieve_variable(
    variable: str,
    model: str,
    scenario: str,
    label: str,
) -> pd.DataFrame:
    """Fetch variable, filter to model/scenario, map regions to ISO-2."""
    df = _fetch_variable(variable, model, scenario)
    logger.info("After filtering to %s / %s: %d rows", model, scenario, len(df))

    cc = coco.CountryConverter()
    df["iso2"] = cc.pandas_convert(pd.Series(df["region"]), to="ISO2", not_found=None)
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
