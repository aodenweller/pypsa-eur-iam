"""Retrieve SSP population and GDP projections from the IIASA scenario database.

Thin wrapper over ``iampypsa.io.ssp.fetch_ssp_data``: pulls country-level Population and GDP|PPP
time series for the configured SSP scenario/models from the IIASA SSP REST API and saves them
as CSVs used by ``downscale_REMIND_demand`` to disaggregate REMIND regional demand. Data is at
5-year intervals matching REMIND's time resolution — no interpolation is applied.

Outputs
-------
- ``population.csv``: columns [iso2, year, value] (population in millions)
- ``gdp.csv``: columns [iso2, year, value] (GDP|PPP in billion USD 2010)
"""

import logging

from _helpers import configure_logging, mock_snakemake
from iampypsa.io.ssp import fetch_ssp_data

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "retrieve_ssp_data",
            configfiles="config/config.remind.yaml",
        )

    configure_logging(snakemake)

    params = snakemake.params
    logger.info("Fetching SSP population + GDP|PPP (%s) from the IIASA SSP API ...", params.ssp_scenario)
    population, gdp = fetch_ssp_data(
        scenario=params.ssp_scenario,
        population_model=params.ssp_population_model,
        gdp_model=params.ssp_gdp_model,
    )

    population.to_csv(snakemake.output.population, index=False)
    logger.info("Wrote %d population rows to %s", len(population), snakemake.output.population)

    gdp.to_csv(snakemake.output.gdp, index=False)
    logger.info("Wrote %d GDP rows to %s", len(gdp), snakemake.output.gdp)
