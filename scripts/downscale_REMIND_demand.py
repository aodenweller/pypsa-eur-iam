"""
Disaggregate REMIND regional demand to country-level demand.

REMIND exports sectoral electricity demand at the resolution of its own regions
(e.g. DEU, FRA, EWN). PyPSA-Eur operates at country level. This script splits
each regional demand value across the constituent countries using a weighted
combination of SSP population and GDP shares.

Weights are sector-specific (configured in ``remind_coupling.demand_downscaling.
sector_weights``). For single-country REMIND regions (e.g. DEU = DE only), the
step is a no-op: the value is preserved and the region label is replaced with
the ISO-2 country code.

The ``_compute_weights`` function is the single extension point for future
per-sector algorithms. Additional inputs (e.g. heating degree days) can be
passed via ``extra_inputs`` without changing the main loop.

Inputs
------
- ``sectoral_load``: long-format CSV from ``import_REMIND_demand`` with columns
  [year, region (REMIND-EU code), sector, unit, value]
- ``population``: SSP population CSV with columns [iso2, year, value]
- ``gdp``: SSP GDP|PPP CSV with columns [iso2, year, value]
- ``region_mapping``: ``config/regionmapping_21_EU11.csv``

Outputs
-------
- ``sectoral_load_country``: same schema as input but ``region`` contains ISO-2
  country codes and values are disaggregated per country.
"""

import logging

import pandas as pd
from _helpers import configure_logging, get_region_mapping, mock_snakemake

logger = logging.getLogger(__name__)


def _normalize(s: pd.Series) -> pd.Series:
    """Normalize a non-negative Series to sum to 1; return uniform weights if total is zero."""
    s = s.astype(float).clip(lower=0.0)
    total = s.sum()
    if total <= 0.0:
        return pd.Series(1.0 / len(s), index=s.index) if len(s) else s
    return s / total


def _compute_weights(
    countries: list[str],
    year: int,
    sector: str,
    pop_data: pd.DataFrame,
    gdp_data: pd.DataFrame,
    sector_weights: dict,
    **extra_inputs,
) -> dict[str, float]:
    """
    Return {iso2: share} for a given set of countries, year, and sector.

    This is the single extension point for future algorithms. Add new data
    sources (e.g. heating degree days) as keyword arguments in ``extra_inputs``
    and branch on ``sector`` here.

    Parameters
    ----------
    countries:
        ISO-2 codes of the countries in this REMIND region.
    year:
        Reference year for the SSP data lookup.
    sector:
        REMIND sector name (e.g. 'AC', 'heatpump').
    pop_data:
        DataFrame indexed by (iso2, year) with a 'value' column.
    gdp_data:
        DataFrame indexed by (iso2, year) with a 'value' column.
    sector_weights:
        Nested dict from config: sector → {'gdp': float, 'population': float}.
    **extra_inputs:
        Reserved for future inputs such as heating degree day data.
    """
    w = sector_weights.get(
        sector, sector_weights.get("AC", {"gdp": 0.6, "population": 0.4})
    )
    gdp_w = w["gdp"]
    pop_w = w["population"]

    pop = pop_data.reindex(
        pd.MultiIndex.from_product(
            [[c for c in countries], [year]], names=["iso2", "year"]
        )
    )["value"].fillna(0.0)
    gdp = gdp_data.reindex(
        pd.MultiIndex.from_product(
            [[c for c in countries], [year]], names=["iso2", "year"]
        )
    )["value"].fillna(0.0)

    pop.index = pop.index.get_level_values("iso2")
    gdp.index = gdp.index.get_level_values("iso2")

    weights = gdp_w * _normalize(gdp) + pop_w * _normalize(pop)
    weights = _normalize(weights)
    return weights.to_dict()


def _disaggregate(
    sectoral_load: pd.DataFrame,
    region_to_countries: dict,
    pop_data: pd.DataFrame,
    gdp_data: pd.DataFrame,
    sector_weights: dict,
) -> pd.DataFrame:
    """Split each (year, region, sector) row into per-country rows."""
    rows = []
    for _, row in sectoral_load.iterrows():
        remind_region = row["region"]
        countries = region_to_countries.get(remind_region)
        if not countries:
            logger.warning(
                "REMIND region '%s' not found in region mapping — skipping.",
                remind_region,
            )
            continue

        if len(countries) == 1:
            # Single-country region: relabel, value unchanged
            rows.append({**row.to_dict(), "region": countries[0]})
        else:
            year = int(row["year"])
            weights = _compute_weights(
                countries,
                year,
                row["sector"],
                pop_data,
                gdp_data,
                sector_weights,
            )
            for country, share in weights.items():
                rows.append(
                    {**row.to_dict(), "region": country, "value": row["value"] * share}
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "downscale_REMIND_demand",
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)

    # Load inputs
    sectoral_load = pd.read_csv(snakemake.input.sectoral_load)
    pop_raw = pd.read_csv(snakemake.input.population).set_index(["iso2", "year"])
    gdp_raw = pd.read_csv(snakemake.input.gdp).set_index(["iso2", "year"])

    # REMIND region → list of ISO-2 countries
    region_to_countries = get_region_mapping(
        snakemake.input.region_mapping,
        source="REMIND-EU",
        target="PyPSA-EUR",
    )

    sector_weights = snakemake.params.sector_weights
    logger.info(
        "Disaggregating demand for %d (year, region, sector) combinations...",
        len(sectoral_load),
    )

    result = _disaggregate(
        sectoral_load, region_to_countries, pop_raw, gdp_raw, sector_weights
    )
    result = (
        result.groupby(["year", "region", "sector", "unit"], as_index=False)["value"]
        .sum()
        .sort_values(["year", "region", "sector"])
    )

    result.to_csv(snakemake.output.sectoral_load_country, index=False)
    logger.info(
        "Wrote %d country-level demand rows to %s",
        len(result),
        snakemake.output.sectoral_load_country,
    )
