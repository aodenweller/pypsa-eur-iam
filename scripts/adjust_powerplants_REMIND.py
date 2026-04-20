# -*- coding: utf-8 -*-

import logging

import pandas as pd
from _helpers import configure_logging, get_region_mapping, mock_snakemake

logger = logging.getLogger(__name__)


FUELTYPE_TO_GROUP = {
    "Lignite": "coal & lignite",
    "Hard Coal": "coal & lignite",
    "Bioenergy": "biomass",
    "Nuclear": "nuclear",
    "Oil": "oil",
}

TECH_TO_GROUP = {
    "CCGT": "CCGT",
    "OCGT": "OCGT",
}


def filter_decommissioned_powerplants(ppl: pd.DataFrame, year: int) -> pd.DataFrame:
    # Keep hydro assets regardless of decommissioning year handling.
    filtered = ppl.query(
        "(Fueltype == 'Hydro') or (DateIn <= @year and (DateOut >= @year or DateOut.isna()))"
    ).copy()
    return filtered


def build_scaling_factors(
    ppl: pd.DataFrame,
    capacities: pd.DataFrame,
    region_mapping: dict,
    year: int,
) -> pd.DataFrame:
    ppl = ppl.copy()
    ppl["technology_group"] = ppl["Fueltype"].map(FUELTYPE_TO_GROUP)
    ppl["technology_group"] = ppl["technology_group"].fillna(
        ppl["Technology"].map(TECH_TO_GROUP)
    )
    ppl["region_REMIND"] = ppl["Country"].map(region_mapping)

    ppl_grouped = (
        ppl.dropna(subset=["technology_group", "region_REMIND"])
        .groupby(["region_REMIND", "technology_group"], observed=False, as_index=False)["Capacity"]
        .sum()
        .rename(columns={"Capacity": "capacity_pypsa"})
    )

    caps_y = capacities.loc[capacities["year"] == year].copy()
    caps_y = caps_y.rename(columns={"p_nom_min": "capacity_remind"})

    compare = ppl_grouped.merge(
        caps_y[["region_REMIND", "technology_group", "capacity_remind"]],
        on=["region_REMIND", "technology_group"],
        how="left",
    )

    compare["capacity_remind"] = compare["capacity_remind"].fillna(compare["capacity_pypsa"])
    compare["scaling_factor"] = 1.0

    mask = (compare["capacity_pypsa"] > 0) & (compare["capacity_remind"] < compare["capacity_pypsa"])
    compare.loc[mask, "scaling_factor"] = (
        compare.loc[mask, "capacity_remind"] / compare.loc[mask, "capacity_pypsa"]
    )

    return compare[["region_REMIND", "technology_group", "capacity_pypsa", "capacity_remind", "scaling_factor"]]


def apply_scaling(ppl: pd.DataFrame, scaling: pd.DataFrame, region_mapping: dict) -> pd.DataFrame:
    out = ppl.copy()
    out["technology_group"] = out["Fueltype"].map(FUELTYPE_TO_GROUP)
    out["technology_group"] = out["technology_group"].fillna(
        out["Technology"].map(TECH_TO_GROUP)
    )
    out["region_REMIND"] = out["Country"].map(region_mapping)

    out = out.merge(
        scaling[["region_REMIND", "technology_group", "scaling_factor"]],
        on=["region_REMIND", "technology_group"],
        how="left",
    )
    out["scaling_factor"] = out["scaling_factor"].fillna(1.0)
    out["Capacity"] = out["Capacity"] * out["scaling_factor"]

    return out.drop(columns=["technology_group", "region_REMIND", "scaling_factor"])


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

    year = int(snakemake.wildcards["year"])
    ppl = pd.read_csv(snakemake.input["powerplants"], index_col=0)
    capacities = pd.read_csv(snakemake.input["capacities"])

    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"],
        source="PyPSA-EUR",
        target="REMIND-EU",
        flatten=True,
    )

    ppl = ppl.loc[ppl["Country"].isin(snakemake.params["countries"])].copy()
    ppl = filter_decommissioned_powerplants(ppl, year)

    scaling = build_scaling_factors(ppl, capacities, region_mapping, year)
    ppl_adjusted = apply_scaling(ppl, scaling, region_mapping)

    reductions = scaling.loc[scaling["scaling_factor"] < 1].copy()
    if reductions.empty:
        logger.info("No capacity downscaling required for year %s", year)
    else:
        for _, row in reductions.iterrows():
            logger.info(
                "Scaled %s in %s from %.2f MW to %.2f MW",
                row["technology_group"],
                row["region_REMIND"],
                row["capacity_pypsa"],
                row["capacity_remind"],
            )

    logger.info("Exporting adjusted powerplants to %s", snakemake.output["powerplants_adjusted"])
    ppl_adjusted.reset_index(drop=True).to_csv(snakemake.output["powerplants_adjusted"])
