# -*- coding: utf-8 -*-

import logging

import pandas as pd
from _helpers import (
    configure_logging,
    get_region_mapping,
    get_technology_mapping,
    mock_snakemake,
    read_remind_data,
)

logger = logging.getLogger(__name__)

LINK_TECHNOLOGIES_INPUT_CAPACITY = {"elh2", "h2turb", "btin"}


def load_remind_capacities(fp_remind_data: str) -> pd.DataFrame:
    """Load REMIND capacities and convert to MW/MWh units."""
    capacities = read_remind_data(
        fp_remind_data,
        "p32_capAvg",
        rename_columns={
            "ttot": "year",
            "all_regi": "region_REMIND",
            "all_te": "remind_technology",
        },
    )
    capacities = capacities[["year", "region_REMIND", "remind_technology", "value"]].copy()
    capacities["value"] *= 1e6
    return capacities


def adjust_link_capacities_to_input(
    capacities: pd.DataFrame,
    fp_remind_data: str,
) -> pd.DataFrame:
    """Convert output-based REMIND capacities to input capacities for link-like techs."""
    efficiencies = read_remind_data(
        fp_remind_data,
        "pm_eta_conv",
        rename_columns={
            "tall": "year",
            "all_regi": "region_REMIND",
            "all_te": "remind_technology",
            "value": "efficiency",
        },
    )
    efficiencies = efficiencies[["year", "region_REMIND", "remind_technology", "efficiency"]]

    merged = capacities.merge(
        efficiencies,
        on=["year", "region_REMIND", "remind_technology"],
        how="left",
    )

    is_link_tech = merged["remind_technology"].isin(LINK_TECHNOLOGIES_INPUT_CAPACITY)
    missing_eta = is_link_tech & merged["efficiency"].isna()
    zero_eta = is_link_tech & (merged["efficiency"] == 0)

    if missing_eta.any():
        logger.warning(
            "Missing efficiency values for %s rows of link technologies; keeping original values.",
            int(missing_eta.sum()),
        )
    if zero_eta.any():
        logger.warning(
            "Zero efficiency values for %s rows of link technologies; keeping original values.",
            int(zero_eta.sum()),
        )

    valid_eta = is_link_tech & merged["efficiency"].notna() & (merged["efficiency"] != 0)
    merged.loc[valid_eta, "value"] = merged.loc[valid_eta, "value"] / merged.loc[valid_eta, "efficiency"]

    return merged.drop(columns=["efficiency"])


def map_to_pypsa_technology_groups(
    capacities: pd.DataFrame,
    fp_technology_mapping: str,
) -> pd.DataFrame:
    """Map REMIND technologies to PyPSA technology groups and aggregate capacities."""
    technology_mapping = get_technology_mapping(
        fp_technology_mapping,
        group_technologies=True,
    )
    technology_mapping = technology_mapping[["REMIND-EU", "technology_group"]].drop_duplicates()

    mapped = capacities.merge(
        technology_mapping,
        left_on="remind_technology",
        right_on="REMIND-EU",
        how="left",
    )

    unmapped = mapped["technology_group"].isna().sum()
    if unmapped > 0:
        logger.warning(
            "Dropping %s rows with unmapped REMIND technologies.",
            int(unmapped),
        )

    mapped = mapped.dropna(subset=["technology_group"])

    grouped = (
        mapped.groupby(["year", "region_REMIND", "technology_group"], as_index=False, observed=False)["value"]
        .sum()
        .round(2)
    )
    grouped = grouped[grouped["value"] > 0].rename(columns={"value": "p_nom_min"})

    return grouped.sort_values(["year", "region_REMIND", "technology_group"]).reset_index(drop=True)


def filter_to_modeled_regions(capacities: pd.DataFrame, fp_region_mapping: str) -> pd.DataFrame:
    """Restrict output to REMIND regions overlapping PyPSA-Eur regions."""
    region_mapping = get_region_mapping(
        fp_region_mapping,
        source="PyPSA-EUR",
        target="REMIND-EU",
    )
    remind_regions = pd.Series(region_mapping).explode().dropna().unique()
    return capacities[capacities["region_REMIND"].isin(remind_regions)].copy()


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "import_REMIND_capacities",
            scenario="TEST",
            iteration="1",
            configfiles="config/config.remind.yaml",
        )

    configure_logging(snakemake)

    logger.info("Loading REMIND capacities...")
    capacities = load_remind_capacities(snakemake.input["remind_data"])
    capacities = filter_to_modeled_regions(capacities, snakemake.input["region_mapping"])

    logger.info("Adjusting capacities for link technologies to input-capacity convention...")
    capacities = adjust_link_capacities_to_input(
        capacities,
        snakemake.input["remind_data"],
    )

    logger.info("Mapping REMIND technologies to PyPSA technology groups...")
    capacities = map_to_pypsa_technology_groups(
        capacities,
        snakemake.input["technology_cost_mapping"],
    )

    logger.info("Exporting data to %s", snakemake.output["capacities"])
    capacities.to_csv(snakemake.output["capacities"], index=False)
