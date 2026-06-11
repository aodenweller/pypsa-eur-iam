
"""
Build REMIND-adjusted technology costs for PyPSA-Eur.

Reads investment costs, fixed/variable O&M, lifetime, efficiency, CO2 intensity, and
fuel costs from the REMIND GDX file, maps them to PyPSA-Eur carrier names via the
technology mapping CSV, and merges the result as overrides on top of the PyPSA-Eur
baseline cost CSV. Investment costs for electrolysis and battery inverter are converted
from output-capacity to input-capacity basis. Per-region discount rates from REMIND are
used, and PyPSA-Eur's ``prepare_costs`` function computes annualised capital costs and
marginal costs — called once per mapped REMIND region.

Outputs
-------
- ``costs_raw_overwritten.csv``: raw cost table restricted to mapped technologies, with REMIND overrides applied; one block per region (region column is the first column).
- ``costs_processed.csv``: processed cost table (capital_cost, marginal_cost, etc.) ready for the network build; indexed by (region, technology) MultiIndex.
"""

import logging

import pandas as pd
import pypsa
from _helpers import configure_logging
from remind.adapter_remind_eur import LINK_TECHS, RemindEurAdapter
from rpycpl.io import RemindLoader
from rpycpl.io.remind_symbols import load_symbol_specs
from rpycpl.transforms.costs import (
    add_discount_rate,
    build_cost_overrides,
    convert_investment_to_input_capacity_basis,
    merge_cost_overrides_into_baseline,
)
from rpycpl.transforms.mapping import read_region_map as get_region_mapping

logger = logging.getLogger(__name__)


def build_adapter(remind_data_path: str, mapped_regions: set[str]) -> RemindEurAdapter:
    """Construct the shared ``RemindEurAdapter`` bound to the REMIND GDX + central symbol config.

    The adapter is the single entry point to the coupling package: it owns the loader and the
    resolved symbol map, exposes ``extract_cost_parameters`` (REMIND cost semantics), and is
    reused below to read the discount-rate symbol.
    """
    return RemindEurAdapter(
        loader=RemindLoader(remind_data_path),
        symbols=load_symbol_specs(),
        region_map={},
        config={"link_techs": LINK_TECHS},
        remind_regions=sorted(mapped_regions),
    )


def build_mapped_overrides(
    technology_mapping: pd.DataFrame,
    remind_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map REMIND parameter values onto PyPSA-Eur carriers via rpycpl's ``build_cost_overrides``.

    ``build_cost_overrides`` does the 1:1 (region, technology, parameter) lookup, drops mapped
    references that are absent from the GDX (those keep the PyPSA-Eur baseline on merge), and
    raises on duplicates. Here we only log the dropped references and tag the provenance columns
    that ``merge_cost_overrides_into_baseline`` propagates.
    """
    overrides = build_cost_overrides(technology_mapping, remind_df)

    present = set(zip(remind_df["reference"], remind_df["parameter"]))
    mapped = technology_mapping.query("`source` == 'REMIND'")
    for _, row in mapped.iterrows():
        if (row["reference"], row["parameter"]) not in present:
            logger.warning(
                "REMIND reference '%s' (→ '%s', parameter '%s') not found in GDX "
                "— falling back to PyPSA-Eur baseline value.",
                row["reference"],
                row["PyPSA-Eur technology"],
                row["parameter"],
            )

    overrides["source"] = "REMIND-EU"
    overrides["further description"] = "Extracted from REMIND-EU model in import_REMIND_costs.py"
    return overrides


def build_pypsa_default_overrides(
    technology_mapping: pd.DataFrame,
    baseline_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Pull parameter values from the PyPSA-Eur baseline costs for rows marked source=PyPSA-Eur."""
    df = technology_mapping.query(
        "`source` == 'PyPSA-Eur'"
    ).drop(columns=["unit"])
    df = df.merge(
        baseline_raw,
        left_on=["PyPSA-Eur technology", "parameter"],
        right_on=["technology", "parameter"],
        how="left",
        validate="one_to_one",
    )
    df["source"] = "PyPSA-EUR"
    df["further description"] = "Default parameter from PyPSA-EUR baseline cost file"
    return df[["technology", "parameter", "value", "unit", "source", "further description"]]


def build_set_value_overrides(technology_mapping: pd.DataFrame, mapping_file: str) -> pd.DataFrame:
    """Return overrides for rows marked source=fixed, converting the reference column to a numeric value."""
    set_df = technology_mapping.query("`source` == 'fixed'").rename(
        columns={
            "PyPSA-Eur technology": "technology",
            "reference": "value",
            "comment": "further description",
        }
    )[["technology", "parameter", "value", "unit", "further description"]].copy()
    set_df["value"] = pd.to_numeric(set_df["value"], errors="raise")
    set_df["source"] = f"Set via configuration file: {mapping_file}"
    set_df["further description"] = set_df["further description"].fillna("")
    return set_df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # When run directly, Python adds scripts/ to sys.path, not the repo root.
    # scripts.process_cost_data must be imported as a package from the repo root,
    # so we insert it explicitly. Not needed under Snakemake (which sets up sys.path correctly).
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import scripts.process_cost_data as process_cost_data
    from scripts.process_cost_data import prepare_costs

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_costs",
            scen_REMIND="PkBudg1000_EU",
            iter_REMIND="1",
            year_REMIND="2050",
            configfiles="config/config.remind_europe.yaml",
        )

    configure_logging(snakemake)
    year = str(snakemake.wildcards["year_REMIND"])
    logger.info("Building REMIND-adjusted costs for year %s", year)

    countries = set(snakemake.config["countries"])
    full_mapping = get_region_mapping(snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU")
    mapped_regions = {r for c, rs in full_mapping.items() if c in countries for r in rs if r}

    technology_mapping = pd.read_csv(snakemake.input["technology_cost_mapping"])
    mapped_technologies = set(technology_mapping["PyPSA-Eur technology"].dropna().unique())

    adapter = build_adapter(snakemake.input["remind_data"], mapped_regions)
    remind_long = adapter.extract_cost_parameters(int(year))
    baseline_raw = pd.read_csv(snakemake.input["original_costs"])

    # REMIND-derived overrides keep their region dimension; non-regional overrides are
    # the same for every region (PyPSA-Eur baseline values and fixed-value entries).
    regional_mapped_overrides = build_mapped_overrides(technology_mapping, remind_long)
    pypsa_overrides = build_pypsa_default_overrides(technology_mapping, baseline_raw)
    set_overrides = build_set_value_overrides(
        technology_mapping,
        snakemake.input["technology_cost_mapping"],
    )
    non_regional_overrides = pd.concat([pypsa_overrides, set_overrides], ignore_index=True)

    discount_rates = adapter.discount_rates(int(year))
    logger.info("Regional REMIND discount rates for year %s: %s", year, discount_rates.round(4).to_dict())

    n = pypsa.Network(snakemake.input["network"])
    nyears = n.snapshot_weightings.generators.sum() / 8760.0
    # `prepare_costs` currently resolves `snakemake` and `planning_horizon`
    # from module-level globals in `scripts.process_cost_data`. We set them
    # here to keep `process_cost_data.py` unchanged while calling it from REMIND.
    process_cost_data.snakemake = snakemake
    process_cost_data.planning_horizon = year

    all_raw = []
    all_processed = []

    for region in sorted(mapped_regions):
        region_overrides = regional_mapped_overrides[
            regional_mapped_overrides["region"] == region
        ].drop(columns="region")

        combined = pd.concat([region_overrides, non_regional_overrides], ignore_index=True)
        combined = add_discount_rate(combined, discount_rates[region], source="REMIND-EU", reference="p_r")
        combined = convert_investment_to_input_capacity_basis(combined)

        merged_raw = merge_cost_overrides_into_baseline(baseline_raw, combined)

        merged_raw_mapped = merged_raw.loc[merged_raw["technology"].isin(mapped_technologies)].copy()
        merged_raw_mapped.insert(0, "region", region)
        all_raw.append(merged_raw_mapped)

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
        costs_processed.index = pd.MultiIndex.from_tuples(
            [(region, t) for t in costs_processed.index], names=["region", "technology"]
        )
        all_processed.append(costs_processed)

    raw_combined = pd.concat(all_raw, ignore_index=True)
    processed_combined = pd.concat(all_processed)

    logger.info(
        "Keeping %d raw cost rows across %d regions × %d mapped technologies",
        len(raw_combined),
        raw_combined["region"].nunique(),
        raw_combined["technology"].nunique(),
    )
    logger.info(
        "Keeping %d processed cost rows across %d regions × %d mapped technologies",
        len(processed_combined),
        processed_combined.index.get_level_values("region").nunique(),
        processed_combined.index.get_level_values("technology").nunique(),
    )

    required_cols = ["capital_cost", "marginal_cost"]
    missing_required = [c for c in required_cols if c not in processed_combined.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in processed costs: {missing_required}")
    if processed_combined[required_cols].isna().any().any():
        nan_cols = list(processed_combined[required_cols].columns[processed_combined[required_cols].isna().any()])
        raise ValueError(f"NaN values in required processed cost columns: {nan_cols}")

    logger.info(
        "Exporting overwritten raw costs to %s",
        snakemake.output["costs_raw_overwritten"],
    )
    raw_combined.to_csv(snakemake.output["costs_raw_overwritten"], index=False)

    logger.info(
        "Exporting processed costs to %s",
        snakemake.output["costs_processed"],
    )
    processed_combined.to_csv(snakemake.output["costs_processed"])

    # Region-averaged flat costs: single-index by technology, used by prepare_network.py
    # which expects the upstream load_costs() format (index_col=0 → technology index).
    costs_flat = processed_combined.groupby(level="technology").mean()
    logger.info(
        "Exporting flat (region-averaged) processed costs to %s",
        snakemake.output["costs_processed_flat"],
    )
    costs_flat.to_csv(snakemake.output["costs_processed_flat"])
