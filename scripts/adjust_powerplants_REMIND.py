"""
Align the powerplantmatching database with REMIND capacity targets before the network build.

Assigns a REMIND-compatible carrier name to each plant using the technology mapping CSV,
filters out plants not yet built or already decommissioned in the target year, computes
per-(REMIND region, carrier) scaling factors to reduce aggregate PyPSA-Eur capacity to
the REMIND target wherever PyPSA-Eur exceeds it, and overwrites each plant's Fueltype
with the carrier name so that downstream scripts receive consistent carrier labels.
"""

import logging

import pandas as pd
from _helpers import configure_logging, get_region_mapping, mock_snakemake

logger = logging.getLogger(__name__)


def assign_carriers_from_mapping(ppl: pd.DataFrame, fp_mapping: str) -> pd.Series:
    """
    Assign a carrier name to each powerplant row using ppm_ columns in the mapping CSV.

    Matching is applied in specificity order so that more-specific rules take
    precedence over broader ones:
      1. (fueltype + set)        e.g. Natural Gas + CHP  → gas-chp
      2. (fueltype + technology) e.g. Natural Gas + CCGT → ccgt
      3. (fueltype only)         e.g. Hard Coal;Lignite  → coal-pc

    Unmatched plants get NaN.
    """
    df = pd.read_csv(fp_mapping)
    unique = (
        df[["PyPSA-Eur technology", "ppm_fueltype", "ppm_technology", "ppm_set"]]
        .drop_duplicates()
        .fillna("")
    )
    # Only keep rows that have at least one ppm column filled.
    has_ppm = unique[unique[["ppm_fueltype", "ppm_technology", "ppm_set"]].ne("").any(axis=1)]

    carrier = pd.Series(index=ppl.index, dtype=object)

    # 1. Fueltype + Set rules (e.g. gas-chp)
    fs_rows = has_ppm[has_ppm["ppm_fueltype"].ne("") & has_ppm["ppm_set"].ne("") & has_ppm["ppm_technology"].eq("")]
    for _, row in fs_rows.iterrows():
        fueltypes = [ft.strip() for ft in row["ppm_fueltype"].split(";")]
        mask = ppl["Fueltype"].isin(fueltypes) & (ppl["Set"] == row["ppm_set"]) & carrier.isna()
        carrier[mask] = row["PyPSA-Eur technology"]

    # 2. Fueltype + Technology rules (e.g. ccgt, ocgt)
    ft_rows = has_ppm[has_ppm["ppm_fueltype"].ne("") & has_ppm["ppm_technology"].ne("") & has_ppm["ppm_set"].eq("")]
    for _, row in ft_rows.iterrows():
        fueltypes = [ft.strip() for ft in row["ppm_fueltype"].split(";")]
        mask = ppl["Fueltype"].isin(fueltypes) & (ppl["Technology"] == row["ppm_technology"]) & carrier.isna()
        carrier[mask] = row["PyPSA-Eur technology"]

    # 3. Fueltype-only rules (e.g. coal-pc, nuclear, oil)
    f_rows = has_ppm[has_ppm["ppm_fueltype"].ne("") & has_ppm["ppm_technology"].eq("") & has_ppm["ppm_set"].eq("")]
    for _, row in f_rows.iterrows():
        fueltypes = [ft.strip() for ft in row["ppm_fueltype"].split(";")]
        mask = ppl["Fueltype"].isin(fueltypes) & carrier.isna()
        carrier[mask] = row["PyPSA-Eur technology"]

    return carrier


def filter_decommissioned_powerplants(ppl: pd.DataFrame, year: int) -> pd.DataFrame:
    """Drop plants that are not yet built or already decommissioned in the given year."""
    # Keep hydro assets and everywhere_powerplants (Capacity == 0) regardless of dates.
    filtered = ppl.query(
        "(Fueltype == 'Hydro') or (Capacity == 0) or (DateIn <= @year and (DateOut >= @year or DateOut.isna()))"
    ).copy()
    return filtered


def build_scaling_factors(
    ppl: pd.DataFrame,
    capacities: pd.DataFrame,
    region_mapping: dict,
    year: int,
    fp_mapping: str,
) -> pd.DataFrame:
    """Compute per-(region, carrier) scaling factors to reduce PyPSA capacity to REMIND targets where needed."""
    ppl = ppl.copy()
    ppl["carrier"] = assign_carriers_from_mapping(ppl, fp_mapping)
    ppl["region_REMIND"] = ppl["Country"].map(region_mapping)

    ppl_grouped = (
        ppl.dropna(subset=["carrier", "region_REMIND"])
        .groupby(["region_REMIND", "carrier"], observed=False, as_index=False)["Capacity"]
        .sum()
        .rename(columns={"Capacity": "capacity_pypsa"})
    )

    caps_y = capacities.loc[capacities["year"] == year].copy()
    caps_y = caps_y.rename(columns={"p_nom_min": "capacity_remind"})

    compare = ppl_grouped.merge(
        caps_y[["region_REMIND", "carrier", "capacity_remind"]],
        on=["region_REMIND", "carrier"],
        how="left",
    )

    compare["capacity_remind"] = compare["capacity_remind"].fillna(compare["capacity_pypsa"])
    compare["scaling_factor"] = 1.0

    mask = (compare["capacity_pypsa"] > 0) & (compare["capacity_remind"] < compare["capacity_pypsa"])
    compare.loc[mask, "scaling_factor"] = (
        compare.loc[mask, "capacity_remind"] / compare.loc[mask, "capacity_pypsa"]
    )

    return compare[["region_REMIND", "carrier", "capacity_pypsa", "capacity_remind", "scaling_factor"]]


def apply_scaling(ppl: pd.DataFrame, scaling: pd.DataFrame, region_mapping: dict, fp_mapping: str) -> pd.DataFrame:
    """Scale plant capacities by the computed factors and overwrite Fueltype with the REMIND carrier name."""
    out = ppl.copy()
    out["carrier"] = assign_carriers_from_mapping(out, fp_mapping)
    out["region_REMIND"] = out["Country"].map(region_mapping)

    out = out.merge(
        scaling[["region_REMIND", "carrier", "scaling_factor"]],
        on=["region_REMIND", "carrier"],
        how="left",
    )
    out["scaling_factor"] = out["scaling_factor"].fillna(1.0)
    out["Capacity"] = out["Capacity"] * out["scaling_factor"]

    # Overwrite Fueltype with REMIND carrier name so load_and_aggregate_powerplants
    # in add_electricity_sector_REMIND.py gets the correct carrier directly.
    # to_pypsa_names() just lowercases, so writing "gas-chp" here gives "gas-chp" there.
    matched = out["carrier"].notna()
    out.loc[matched, "Fueltype"] = out.loc[matched, "carrier"]

    return out.drop(columns=["carrier", "region_REMIND", "scaling_factor"])


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "adjust_powerplants_REMIND",
            scenario="TEST",
            iteration="1",
            year="2030",
            clusters="4",
            configfiles="config/config.remind.yaml",
        )

    configure_logging(snakemake)

    year = int(snakemake.wildcards["year_REMIND"])
    ppl = pd.read_csv(snakemake.input["powerplants"], index_col=0)
    capacities = pd.read_csv(snakemake.input["capacities"])
    fp_mapping = snakemake.input["technology_mapping"]

    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"],
        source="PyPSA-EUR",
        target="REMIND-EU",
        flatten=True,
    )

    ppl = ppl.loc[ppl["Country"].isin(snakemake.params["countries"])].copy()
    ppl = filter_decommissioned_powerplants(ppl, year)

    scaling = build_scaling_factors(ppl, capacities, region_mapping, year, fp_mapping)
    ppl_adjusted = apply_scaling(ppl, scaling, region_mapping, fp_mapping)

    reductions = scaling.loc[scaling["scaling_factor"] < 1].copy()
    if reductions.empty:
        logger.info("No capacity downscaling required for year %s", year)
    else:
        for _, row in reductions.iterrows():
            logger.info(
                "Scaled %s in %s from %.2f MW to %.2f MW",
                row["carrier"],
                row["region_REMIND"],
                row["capacity_pypsa"],
                row["capacity_remind"],
            )

    logger.info("Exporting adjusted powerplants to %s", snakemake.output["powerplants_adjusted"])
    ppl_adjusted.reset_index(drop=True).to_csv(snakemake.output["powerplants_adjusted"])
