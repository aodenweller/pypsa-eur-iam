"""
Disaggregate REMIND regional demand to country-level demand.

Description
-----------
REMIND exports sectoral electricity demand at the resolution of its own regions
(e.g. DEU, FRA, EWN). This script splits each regional value across constituent
countries using a weighted combination of SSP population and GDP shares.
Weights are sector-specific (configured in
``remind_coupling.demand_downscaling.sector_weights``).

For single-country regions (e.g. DEU = DE only) the step is a no-op. Demand
attributed to unconfigured countries is excluded from the model; a warning is
emitted when this exceeds 1 % of regional demand (e.g. Turkey in NES).
"""

import logging

import pandas as pd
from _helpers import configure_logging, get_region_mapping, mock_snakemake

logger = logging.getLogger(__name__)


def _normalize(s: pd.Series) -> pd.Series:
    """Normalize to sum to 1; return uniform weights if total is zero."""
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
    configured_countries: set[str],
    **extra_inputs,
) -> dict[str, float]:
    """
    Return {iso2: share} for a given set of countries, year, and sector.

    Weights are computed over *all* region members including unconfigured ones.
    Missing SSP data for configured countries raises an error; unconfigured
    countries with missing data receive zero weight.
    """
    w = sector_weights.get(
        sector, sector_weights.get("AC", {"gdp": 0.6, "population": 0.4})
    )

    # SSP data ends at 2100; clamp to the last available year rather than
    # falling back to zero (which would produce spurious uniform weights).
    available_years = pop_data.index.get_level_values("year").unique()
    lookup_year = min(year, available_years.max())
    if lookup_year != year:
        logger.debug(
            "SSP data unavailable for year %d — using %d weights instead.",
            year,
            lookup_year,
        )

    idx = pd.MultiIndex.from_product([countries, [lookup_year]], names=["iso2", "year"])
    pop = pop_data.reindex(idx)["value"]
    gdp = gdp_data.reindex(idx)["value"]

    for label, series in [("population", pop), ("GDP", gdp)]:
        missing = [
            c
            for c in series[series.isna()].index.get_level_values("iso2")
            if c in configured_countries
        ]
        if missing:
            raise ValueError(
                f"SSP {label} data missing for {missing} in year {lookup_year}."
            )

    pop = pop.fillna(0.0)
    gdp = gdp.fillna(0.0)
    pop.index = pop.index.get_level_values("iso2")
    gdp.index = gdp.index.get_level_values("iso2")

    weights = w["gdp"] * _normalize(gdp) + w["population"] * _normalize(pop)
    return _normalize(weights).to_dict()


def _disaggregate(
    sectoral_load: pd.DataFrame,
    region_to_countries: dict,
    pop_data: pd.DataFrame,
    gdp_data: pd.DataFrame,
    sector_weights: dict,
    configured_countries: set[str],
) -> pd.DataFrame:
    """Split each (year, region, sector) row into per-country rows."""
    rows = []
    warned_regions: set[str] = set()

    for _, row in sectoral_load.iterrows():
        remind_region = row["region"]
        all_members = region_to_countries.get(remind_region)
        if not all_members:
            logger.warning(
                "REMIND region '%s' not found in region mapping — skipping.",
                remind_region,
            )
            continue

        configured = [c for c in all_members if c in configured_countries]
        if not configured:
            continue

        if len(all_members) == 1:
            rows.append({**row.to_dict(), "region": configured[0]})
        else:
            year = int(row["year"])
            weights = _compute_weights(
                all_members,
                year,
                row["sector"],
                pop_data,
                gdp_data,
                sector_weights,
                configured_countries=configured_countries,
            )

            unconfigured = [c for c in all_members if c not in configured_countries]
            if unconfigured and remind_region not in warned_regions:
                frac = sum(weights.get(c, 0.0) for c in unconfigured)
                logger.warning(
                    "REMIND region '%s' has unconfigured countries %s accounting for "
                    "%.1f%% of regional demand — this demand is excluded from the model.",
                    remind_region,
                    unconfigured,
                    frac * 100,
                )
                warned_regions.add(remind_region)

            for country in configured:
                rows.append(
                    {
                        **row.to_dict(),
                        "region": country,
                        "value": row["value"] * weights.get(country, 0.0),
                    }
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

    sectoral_load = pd.read_csv(snakemake.input.sectoral_load)
    pop_raw = pd.read_csv(snakemake.input.population).set_index(["iso2", "year"])
    gdp_raw = pd.read_csv(snakemake.input.gdp).set_index(["iso2", "year"])

    region_to_countries = get_region_mapping(
        snakemake.input.region_mapping,
        source="REMIND-EU",
        target="PyPSA-EUR",
    )
    configured_countries = set(snakemake.params.countries)

    years = snakemake.params.years
    sectoral_load = sectoral_load[sectoral_load["year"].isin(years)]
    logger.info(
        "Filtered sectoral_load to %d scenario years: %s.",
        len(years),
        sorted(years),
    )

    sector_weights = snakemake.params.sector_weights
    logger.info(
        "Disaggregating demand for %d (year, region, sector) combinations...",
        len(sectoral_load),
    )

    result = _disaggregate(
        sectoral_load,
        region_to_countries,
        pop_raw,
        gdp_raw,
        sector_weights,
        configured_countries,
    )
    result = (
        result.groupby(["year", "region", "sector", "unit"], as_index=False)["value"]
        .sum()
        .sort_values(["year", "region", "sector"])
    )

    if missing_countries := sorted(configured_countries - set(result["region"].unique())):
        country_to_region = {
            c: r for r, members in region_to_countries.items() for c in members
        }
        missing_regions = sorted(
            {country_to_region.get(c, "unknown") for c in missing_countries}
        )
        raise ValueError(
            f"No demand data for countries {missing_countries}. "
            f"The REMIND regions {missing_regions} are missing from the GDX export."
        )

    result.to_csv(snakemake.output.sectoral_load_country, index=False)
    logger.info(
        "Wrote %d country-level demand rows to %s",
        len(result),
        snakemake.output.sectoral_load_country,
    )
