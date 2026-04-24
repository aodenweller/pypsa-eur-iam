# -*- coding: utf-8 -*-

import logging

import pandas as pd
import pypsa
import yaml
from _helpers import configure_logging, get_region_mapping, read_remind_data
import scripts.process_cost_data as process_cost_data
from scripts.process_cost_data import prepare_costs

logger = logging.getLogger(__name__)


def load_region_mapping(fp_region_mapping: str, countries: list[str]) -> pd.DataFrame:
    region_mapping = (
        pd.DataFrame.from_dict(
            get_region_mapping(
                fp_region_mapping,
                source="PyPSA-EUR",
                target="REMIND-EU",
            )
        )
        .T.reset_index()
        .rename(columns={"index": "PyPSA-EUR", 0: "REMIND-EU"})
    )
    return region_mapping.query("`PyPSA-EUR`.isin(@countries)")


def load_technology_mapping(fp_mapping: str) -> pd.DataFrame:
    technology_mapping = pd.read_csv(fp_mapping)
    technology_mapping["reference"] = technology_mapping["reference"].apply(
        lambda x: yaml.safe_load(x) if isinstance(x, str) else x
    )
    return technology_mapping.explode("reference")


def extract_remind_parameter_data(snakemake, region_mapping: pd.DataFrame) -> pd.DataFrame:
    year = str(snakemake.wildcards["year_REMIND"])

    costs = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_capCostwAdjCost",
        rename_columns={"ttot": "year", "all_regi": "region", "all_te": "technology"},
    ).query("year == @year")
    costs["value"] *= 1e6
    costs["parameter"] = "investment"
    costs["unit"] = "USD/MW"
    costs.loc[costs["technology"].isin(["h2stor", "btstor"]), "unit"] = "USD/MWh"

    lifetime = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_data",
        rename_columns={"all_regi": "region", "all_te": "technology"},
    ).query("char == 'lifetime'")
    lifetime["parameter"] = "lifetime"
    lifetime["unit"] = "years"

    fom = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_data",
        rename_columns={"all_regi": "region", "all_te": "technology"},
    ).query("char == 'omf'")
    fom["value"] *= 100
    fom["parameter"] = "FOM"
    fom["unit"] = "%/year"

    vom = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_data",
        rename_columns={"all_regi": "region", "all_te": "technology"},
    ).query("char == 'omv'")
    vom["value"] *= 1e6 / 8760
    vom["parameter"] = "VOM"
    vom["unit"] = "USD/MWh"

    co2_intensity = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_emifac",
        rename_columns={
            "tall_0": "year",
            "all_regi_1": "region",
            "all_enty_2": "from_carrier",
            "all_enty_3": "to_carrier",
            "all_te_4": "technology",
            "all_enty_5": "emission_type",
        },
    ).query("to_carrier == 'seel' & emission_type == 'co2' & year == @year")
    co2_intensity["value"] *= 1e9 * ((2 * 16 + 12) / 12) / 8760 / 1e6
    co2_intensity["parameter"] = "CO2 intensity"
    co2_intensity["unit"] = "t_CO2/MWh_th"

    efficiency = pd.concat(
        [
            read_remind_data(
                file_path=snakemake.input["remind_data"],
                variable_name="pm_eta_conv",
                rename_columns={"tall": "year", "all_regi": "region", "all_te": "technology"},
            ),
            read_remind_data(
                file_path=snakemake.input["remind_data"],
                variable_name="pm_dataeta",
                rename_columns={"tall": "year", "all_regi": "region", "all_te": "technology"},
            ),
        ]
    ).query("year == @year")
    efficiency["parameter"] = "efficiency"
    efficiency["unit"] = "p.u."
    efficiency.loc[efficiency["technology"].isin(["fnrs", "tnrs"]), "value"] *= 8760 / 1e6
    efficiency.loc[efficiency["technology"].isin(["fnrs", "tnrs"]), "unit"] = "MWh/g_U"
    efficiency.loc[efficiency["technology"] == "btin", "value"] **= 2

    fuel_costs = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_PEPriceAvg",
        rename_columns={"ttot": "year", "all_regi": "region", "all_enty": "technology"},
    ).query("year == @year")
    fuel_costs["parameter"] = "fuel"
    fuel_costs.loc[~(fuel_costs["technology"] == "peur"), "value"] *= 1e6 / 8760
    fuel_costs["unit"] = "USD/MWh_th"
    fuel_costs.loc[fuel_costs["technology"] == "peur", "unit"] = "USD/g_U"

    df = pd.concat([costs, lifetime, fom, vom, co2_intensity, efficiency, fuel_costs])[
        ["region", "technology", "parameter", "value", "unit"]
    ].rename(columns={"technology": "reference"})

    return df.query("region.isin(@region_mapping['REMIND-EU'])", engine="python")


def build_generation_weighted_overrides(
    snakemake,
    technology_mapping: pd.DataFrame,
    region_mapping: pd.DataFrame,
    remind_df: pd.DataFrame,
) -> pd.DataFrame:
    mapped = technology_mapping.query(
        "`couple to` == 'mapping generation weighted to reference REMIND-EU technology'"
    ).drop(columns=["unit"])

    year = str(snakemake.wildcards["year_REMIND"])
    weights = pd.concat(
        [
            read_remind_data(
                file_path=snakemake.input["remind_data"],
                variable_name="p32_weightGen",
                rename_columns={"ttot": "year", "all_regi": "region", "all_te": "technology", "value": "weight"},
            ),
            read_remind_data(
                file_path=snakemake.input["remind_data"],
                variable_name="p32_weightPEprice",
                rename_columns={"ttot": "year", "all_regi": "region", "all_enty": "technology", "value": "weight"},
            ),
            read_remind_data(
                file_path=snakemake.input["remind_data"],
                variable_name="p32_weightStor",
                rename_columns={"ttot": "year", "all_regi": "region", "all_te": "technology", "value": "weight"},
            ),
        ]
    ).query("`region`.isin(@region_mapping['REMIND-EU']) and year == @year", engine="python")[["region", "technology", "weight"]]

    merged = remind_df.merge(
        weights,
        left_on=["region", "reference"],
        right_on=["region", "technology"],
        how="left",
    )[["region", "reference", "parameter", "value", "unit", "weight"]]

    merged = mapped.merge(merged, on=["reference", "parameter"], how="left")
    merged["weight"] = merged["weight"].fillna(0.0) + 1e-6 / 8760

    def weighted_value(group: pd.DataFrame) -> pd.Series:
        units = group["unit"].dropna().unique()
        if len(units) != 1:
            raise ValueError(
                "Multiple units detected for weighted aggregation: "
                f"{group[['PyPSA-EUR technology', 'parameter']].drop_duplicates()}"
            )
        return pd.Series(
            {
                "value": (group["value"] * group["weight"]).sum(skipna=False)
                / group["weight"].sum(),
                "unit": units[0],
            }
        )

    out = (
        merged.groupby(["PyPSA-EUR technology", "parameter"], observed=False)
        .apply(weighted_value)
        .reset_index()
        .rename(columns={"PyPSA-EUR technology": "technology"})
    )
    out["source"] = "REMIND-EU"
    out["further description"] = "Extracted from REMIND-EU model in import_REMIND_costs.py"
    return out


def build_pypsa_default_overrides(
    technology_mapping: pd.DataFrame,
    baseline_raw: pd.DataFrame,
) -> pd.DataFrame:
    pypsa = technology_mapping.query(
        "`couple to` == 'mapping to PyPSA-EUR default values'"
    ).drop(columns=["unit"])
    pypsa = pypsa.merge(
        baseline_raw,
        left_on=["PyPSA-EUR technology", "parameter"],
        right_on=["technology", "parameter"],
        how="left",
        validate="one_to_one",
    )
    pypsa["source"] = "PyPSA-EUR"
    pypsa["further description"] = "Default parameter from PyPSA-EUR baseline cost file"
    return pypsa[["technology", "parameter", "value", "unit", "source", "further description"]]


def build_set_value_overrides(technology_mapping: pd.DataFrame, mapping_file: str) -> pd.DataFrame:
    set_df = technology_mapping.query("`couple to` == 'setting to reference value'").rename(
        columns={
            "PyPSA-EUR technology": "technology",
            "reference": "value",
            "comment": "further description",
        }
    )[["technology", "parameter", "value", "unit", "further description"]]
    set_df["source"] = f"Set via configuration file: {mapping_file}"
    set_df["further description"] = set_df["further description"].fillna("")
    return set_df


def add_discount_rate(snakemake, costs: pd.DataFrame) -> pd.DataFrame:
    year = str(snakemake.wildcards["year_REMIND"])
    with_discount = costs.loc[costs["parameter"] == "discount rate", "technology"]
    no_discount = costs.loc[~costs["technology"].isin(with_discount)][["technology"]].drop_duplicates()

    discount_rate = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_discountRate",
        rename_columns={"ttot": "year"},
    ).query("year == @year")

    if discount_rate.shape[0] != 1:
        raise ValueError("Expected a single discount rate value from REMIND")

    dr = pd.Series(
        {
            "parameter": "discount rate",
            "value": discount_rate["value"].item(),
            "unit": "p.u.",
            "source": "REMIND-EU",
            "further description": "p32_discountRate",
        }
    ).to_frame().T
    dr = dr.merge(no_discount, how="cross")
    return pd.concat([costs, dr], ignore_index=True)


def apply_special_corrections(costs: pd.DataFrame) -> pd.DataFrame:
    costs = costs.copy()
    for tech in ["electrolysis", "battery inverter"]:
        inv_mask = (costs["technology"] == tech) & (costs["parameter"] == "investment")
        eff_mask = (costs["technology"] == tech) & (costs["parameter"] == "efficiency")
        if inv_mask.any() and eff_mask.any():
            costs.loc[inv_mask, "value"] /= costs.loc[eff_mask, "value"].values
            logger.info("Corrected investment costs for %s from output to input capacity basis.", tech)
    return costs


def merge_overrides_into_baseline(
    baseline_raw: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    base = baseline_raw.set_index(["technology", "parameter"]).copy()
    ov = overrides.set_index(["technology", "parameter"]).copy()

    if ov.index.duplicated().any():
        raise ValueError(
            "Duplicate overrides for (technology, parameter): "
            f"{ov.index[ov.index.duplicated()].tolist()}"
        )

    extra_idx = ov.index.difference(base.index)
    if len(extra_idx) > 0:
        base = pd.concat([base, ov.loc[extra_idx, base.columns.intersection(ov.columns)]])

    for col in ["value", "unit", "source", "further description"]:
        if col in ov.columns:
            base.loc[ov.index.intersection(base.index), col] = ov.loc[
                ov.index.intersection(base.index), col
            ]

    merged = base.reset_index()
    if merged.duplicated(subset=["technology", "parameter"]).any():
        dups = merged[merged.duplicated(subset=["technology", "parameter"], keep=False)]
        raise ValueError(f"Duplicates after merge: {dups}")
    return merged


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_costs",
            scenario="TEST",
            iteration="1",
            year="2030",
            configfiles="config/config.remind.yaml",
        )

    configure_logging(snakemake)
    year = str(snakemake.wildcards["year_REMIND"])
    logger.info("Building REMIND-adjusted costs for year %s", year)

    region_mapping = load_region_mapping(snakemake.input["region_mapping"], snakemake.config["countries"])
    technology_mapping = load_technology_mapping(snakemake.input["technology_cost_mapping"])
    mapped_technologies = set(technology_mapping["PyPSA-EUR technology"].dropna().unique())

    remind_long = extract_remind_parameter_data(snakemake, region_mapping)
    baseline_raw = pd.read_csv(snakemake.input["original_costs"])

    mapped_overrides = build_generation_weighted_overrides(
        snakemake,
        technology_mapping,
        region_mapping,
        remind_long,
    )
    pypsa_overrides = build_pypsa_default_overrides(technology_mapping, baseline_raw)
    set_overrides = build_set_value_overrides(
        technology_mapping,
        snakemake.input["technology_cost_mapping"],
    )

    overrides = pd.concat([mapped_overrides, pypsa_overrides, set_overrides], ignore_index=True)
    overrides = add_discount_rate(snakemake, overrides)
    overrides = apply_special_corrections(overrides)

    merged_raw = merge_overrides_into_baseline(baseline_raw, overrides)
    merged_raw_mapped = merged_raw.loc[merged_raw["technology"].isin(mapped_technologies)].copy()
    logger.info(
        "Keeping %d overwritten raw cost rows across %d mapped technologies",
        len(merged_raw_mapped),
        merged_raw_mapped["technology"].nunique(),
    )

    logger.info(
        "Exporting overwritten raw costs to %s",
        snakemake.output["costs_raw_overwritten"],
    )
    merged_raw_mapped.to_csv(snakemake.output["costs_raw_overwritten"], index=False)

    n = pypsa.Network(snakemake.input["network"])
    nyears = n.snapshot_weightings.generators.sum() / 8760.0
    # `prepare_costs` currently resolves `snakemake` and `planning_horizon`
    # from module-level globals in `scripts.process_cost_data`. We set them
    # here to keep `process_cost_data.py` unchanged while calling it from REMIND.
    process_cost_data.snakemake = snakemake
    process_cost_data.planning_horizon = year
    costs_processed = prepare_costs(
        costs=merged_raw.set_index(["technology", "parameter"]),
        config=snakemake.params["costs"],
        max_hours=snakemake.params["max_hours"],
        nyears=nyears,
        custom_costs_fn=snakemake.input.get("custom_costs"),
    )
    costs_processed = costs_processed.loc[
        costs_processed.index.isin(mapped_technologies)
    ].copy()
    logger.info(
        "Keeping %d processed cost rows across %d mapped technologies",
        len(costs_processed),
        costs_processed.index.nunique(),
    )

    required_cols = ["capital_cost", "marginal_cost"]
    missing_required = [c for c in required_cols if c not in costs_processed.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in processed costs: {missing_required}")
    if costs_processed[required_cols].isna().any().any():
        nan_cols = list(costs_processed[required_cols].columns[costs_processed[required_cols].isna().any()])
        raise ValueError(f"NaN values in required processed cost columns: {nan_cols}")

    logger.info(
        "Exporting processed costs to %s",
        snakemake.output["costs_processed"],
    )
    costs_processed.to_csv(snakemake.output["costs_processed"])
