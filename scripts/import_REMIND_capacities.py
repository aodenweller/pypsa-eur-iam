"""
Read installed-capacity targets from REMIND and export them for use as lower bounds in PyPSA-Eur.

Reads ``p32_capAvg`` (TW → MW), adjusts link-like technologies from output- to
input-capacity convention by dividing by efficiency, maps REMIND technology names to
PyPSA-Eur carrier names via the technology mapping CSV, and filters to configured regions.
"""

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

LINK_TECHNOLOGIES_INPUT_CAPACITY = {"elh2", "h2turb", "btin", "elh2VRE", "h2turbVRE"}

# VRE-coupled variants map to the same carrier as their primary counterpart.
# Merged before PyPSA-Eur mapping so the groupby().sum() aggregates them correctly.
_VRE_TO_PRIMARY = {"elh2VRE": "elh2", "h2turbVRE": "h2turb"}

# REMIND battery storage technologies (solar-coupled, onshore-wind-coupled, offshore-wind-coupled).
# p32_capAvg for these techs needs exogenous scaling before being treated as battery charger capacity:
# After scaling they are renamed to "btin" so map_to_pypsa_carriers groups them with any
# conventional btin capacity and maps the sum to "battery inverter".
# Applied *before* adjust_link_capacities_to_input so all btin rows are η-corrected consistently.
_BATTERY_SCALING_FACTORS: dict[str, float] = {
    "storspv": 4.0,
    "storwindon": 1.2,
    "storwindoff": 1.2,
}


def _merge_vre_technologies(capacities: pd.DataFrame) -> pd.DataFrame:
    """Rename elh2VRE→elh2 and h2turbVRE→h2turb so groupby().sum() aggregates them naturally."""
    capacities = capacities.copy()
    # Cast to str first to avoid CategoricalDtype replace silently no-oping
    tech = capacities["remind_technology"].astype(str)
    capacities["remind_technology"] = tech.map(lambda t: _VRE_TO_PRIMARY.get(t, t))
    return capacities


def _scale_and_merge_battery_technologies(capacities: pd.DataFrame) -> pd.DataFrame:
    """
    Scale storspv/storwindon/storwindoff and rename them to btin as a fallback when btin is absent.

    Must run before adjust_link_capacities_to_input so all btin rows receive the same η correction.
    """
    capacities = capacities.copy()
    tech = capacities["remind_technology"].astype(str)
    btin_available = ((tech == "btin") & (capacities["value"] > 0)).any()
    is_stor = tech.isin(_BATTERY_SCALING_FACTORS)
    if btin_available:
        return capacities[~is_stor].copy()
    scale = tech.map(_BATTERY_SCALING_FACTORS)
    capacities.loc[scale.notna(), "value"] *= scale[scale.notna()]
    capacities["remind_technology"] = tech.map(lambda t: "btin" if t in _BATTERY_SCALING_FACTORS else t)
    return capacities


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


def map_to_pypsa_carriers(
    capacities: pd.DataFrame,
    fp_technology_mapping: str,
) -> pd.DataFrame:
    """Map REMIND technologies to PyPSA-Eur carrier names (1:1)."""
    technology_mapping = get_technology_mapping(fp_technology_mapping)
    # Use only one row per REMIND-EU tech (hydro auto-adds ror; we keep only hydro here)
    remind_to_carrier = (
        technology_mapping[["REMIND-EU", "PyPSA-Eur"]]
        .drop_duplicates(subset="REMIND-EU", keep="first")
    )

    mapped = capacities.merge(
        remind_to_carrier,
        left_on="remind_technology",
        right_on="REMIND-EU",
        how="left",
    )

    unmapped = mapped["PyPSA-Eur"].isna().sum()
    if unmapped > 0:
        logger.warning(
            "Dropping %s rows with unmapped REMIND technologies.",
            int(unmapped),
        )

    mapped = mapped.dropna(subset=["PyPSA-Eur"]).rename(columns={"PyPSA-Eur": "carrier"})

    grouped = (
        mapped.groupby(["year", "region_REMIND", "carrier"], as_index=False, observed=False)["value"]
        .sum()
        .round(2)
    )
    grouped = grouped[grouped["value"] > 0].rename(columns={"value": "p_nom_min"})

    return grouped.sort_values(["year", "region_REMIND", "carrier"]).reset_index(drop=True)


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
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)

    logger.info("Loading REMIND capacities...")
    capacities = load_remind_capacities(snakemake.input["remind_data"])
    capacities = filter_to_modeled_regions(capacities, snakemake.input["region_mapping"])

    logger.info("Merging VRE-coupled technology variants (elh2VRE→elh2, h2turbVRE→h2turb)...")
    capacities = _merge_vre_technologies(capacities)

    logger.info(
        "Scaling and merging VRE-coupled battery technologies "
        "(storspv×4, storwindon×1.2, storwindoff×1.2 → btin)..."
    )
    capacities = _scale_and_merge_battery_technologies(capacities)

    logger.info("Adjusting capacities for link technologies to input-capacity convention...")
    capacities = adjust_link_capacities_to_input(
        capacities,
        snakemake.input["remind_data"],
    )

    logger.info("Mapping REMIND technologies to PyPSA carrier names...")
    capacities = map_to_pypsa_carriers(
        capacities,
        snakemake.input["technology_cost_mapping"],
    )

    logger.info("Exporting data to %s", snakemake.output["capacities"])
    capacities.to_csv(snakemake.output["capacities"], index=False)
