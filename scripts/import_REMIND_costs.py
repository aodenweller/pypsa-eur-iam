# -*- coding: utf-8 -*-
# %%
# There are four steps for importing REMIND costs followed in this script:
# 1. Technologies which get their values from REMIND, weighted by the electricity generation of the related REMIND technology
# 2. Technologies where values are taken from PyPSA-EUR default values
# 3. Technologies where original PyPSA-Eur values are scaled based on a REMIND proxy technology
# 4. Technologies where values are set in the technology mapping config file
# 5. Add discount rate for all technologies where not discount rate is set in the technology mapping config file

import logging

import pandas as pd
import yaml
from _helpers import configure_logging, get_region_mapping, read_remind_data

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_costs",
            scenario="PyPSA_PkBudg1000_DEU_genRCL_4nodes_remind_v350_2025-04-14_20.03.01",
            iteration="1",
            year="2030",
        )

    configure_logging(snakemake)

    # Load region mapping
    logger.info("Loading region mapping ... ")
    region_mapping = (
        pd.DataFrame.from_dict(
            get_region_mapping(
                snakemake.input["region_mapping"],
                source="PyPSA-EUR",
                target="REMIND-EU",
            )
        )
        .T.reset_index()
        .rename(columns={"index": "PyPSA-EUR", 0: "REMIND-EU"})
    )

    # Limit to regions & countries PyPSA-EUR is configured for
    region_mapping = region_mapping.query(
        f"`PyPSA-EUR`.isin({snakemake.config['countries']})"
    )

    # Read new technology mapping
    logger.info("Loading technology mapping ... ")
    technology_mapping = pd.read_csv(
        snakemake.input["technology_cost_mapping"],
    )

    # Convert list-like entries to real lists and explode for 1:1 mapping per row entry between PyPSA-EUR and REMIND-EU technologies
    technology_mapping["reference"] = technology_mapping["reference"].apply(
        lambda x: yaml.safe_load(x) if isinstance(x, str) else x
    ).to_list()
    technology_mapping = technology_mapping.explode("reference")

    #%%
    # +++ 1. Technologies which get their values from REMIND-EU, weighted by the electricity generation of the related REMIND-EU technology +++
    mapped_technologies = technology_mapping.query(
        "`couple to` == 'mapping generation weighted to reference REMIND-EU technology'"
    ).drop(columns=["unit"])

    ## Load REMIND data into long format
    # investment costs
    logger.info("... extracting investment costs")
    costs = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_capCostwAdjCost",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "all_te": "technology",
        },
    ).query("year == '{}'".format(snakemake.wildcards["year"]))
    costs["value"] *= 1e6  # Unit conversion from TUSD/TW to USD/MW (or TUSD/TWh to USD/MWh)
    costs["parameter"] = "investment"
    costs["unit"] = "USD/MW"
    # Storage technologies in USD/MWh
    costs.loc[costs["technology"] == "h2stor", "unit"] = "USD/MWh"
    costs.loc[costs["technology"] == "btstor", "unit"] = "USD/MWh"

    # lifetime
    logger.info("... extracting lifetime")
    lifetime = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_data",
        rename_columns={
            "all_regi": "region",
            "all_te": "technology",
        },
    ).query("char == 'lifetime'")
    lifetime["parameter"] = "lifetime"
    lifetime["unit"] = "years"

    # fixed O&M
    logger.info("... extracting FOM")
    fom = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_data",
        rename_columns={
            "all_regi": "region",
            "all_te": "technology",
        },
    ).query("char == 'omf'")
    fom["value"] *= 100  # Unit conversion from p.u. to %
    fom["parameter"] = "FOM"
    fom["unit"] = "%/year"

    # variable O&M
    logger.info("... extracting VOM")
    vom = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_data",
        rename_columns={
            "all_regi": "region",
            "all_te": "technology",
        },
    ).query("char == 'omv'")
    vom["value"] *= 1e6 / 8760  # Unit conversion from TUSD/TWa to USD/MWh
    vom["parameter"] = "VOM"
    vom[
        "unit"
    ] = "USD/MWh"

    # CO2 intensities
    logger.info("... extracting CO2 intensities")
    co2_intensity = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name = "pm_emifac",
        rename_columns={
            "tall_0": "year",
            "all_regi_1": "region",
            "all_enty_2": "from_carrier",
            "all_enty_3": "to_carrier",
            "all_te_4": "technology",
            "all_enty_5": "emission_type",
        },
    ).query("to_carrier == 'seel' & emission_type == 'co2'").query("year == '{}'".format(snakemake.wildcards["year"]))

    # Unit conversion from Gt_C/TWa to t_CO2/MWh
    co2_intensity["value"] *= 1e9 * ((2 * 16 + 12) / 12) / 8760 / 1e6
    co2_intensity["parameter"] = "CO2 intensity"
    co2_intensity["unit"] = "t_CO2/MWh_th"  # TODO check correct unit

    # Efficiencies
    # Values are split across two different variables in REMIND (constant & year-dependent)
    logger.info("... extracting efficiencies")
    efficiency = pd.concat(
        [
            read_remind_data(
                file_path=snakemake.input["remind_data"],
                variable_name="pm_eta_conv",
                rename_columns={
                    "tall": "year",
                    "all_regi": "region",
                    "all_te": "technology",
                },
            ),
            read_remind_data(
                file_path=snakemake.input["remind_data"],
                variable_name="pm_dataeta",
                rename_columns={
                    "tall": "year",
                    "all_regi": "region",
                    "all_te": "technology",
                },
            ),
        ]
    ).query("year == '{}'".format(snakemake.wildcards["year"]))
    efficiency["parameter"] = "efficiency"
    efficiency["unit"] = "p.u."  # TODO check correct unit
    # Special treatment for nuclear: Efficiencies are in TWa/Mt=8760 TWh/Tg_U -> convert to MWh/g_U to match with fuel costs in USD/g_U
    efficiency.loc[efficiency["technology"].isin(["fnrs", "tnrs"]), "value"] *= 8760 / 1e6
    efficiency.loc[efficiency["technology"].isin(["fnrs", "tnrs"]), "unit"] = "MWh/g_U"
    # Special treatment for battery: Efficiencies in costs.csv should be roundtrip
    efficiency.loc[efficiency["technology"] == "btin", "value"] **= 2

    # Fuel costs
    logger.info("... extracting fuel costs")
    fuel_costs = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_PEPriceAvg",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "all_enty": "technology",
        },
    ).query("year == '{}'".format(snakemake.wildcards["year"]))
    fuel_costs["parameter"] = "fuel"
    # Unit conversion from TUSD/TWa to USD/MWh
    # Special treatment for nuclear fuel uranium (peur): Fuel costs are originally in TUSD/Mt = USD/g_U (TUSD/Tg) -> adjust unit
    fuel_costs.loc[~(fuel_costs["technology"] == "peur"), "value"] *= 1e6 / 8760
    fuel_costs[
        "unit"
    ] = "USD/MWh_th"  # TODO check correct unit (should be per MWh_th input)
    fuel_costs.loc[fuel_costs["technology"] == "peur", "unit"] = "USD/g_U"

    # Combine all technology data for further processing
    df = pd.concat([costs, lifetime, fom, vom, co2_intensity, efficiency, fuel_costs])[
        ["region", "technology", "parameter", "value", "unit"]
    ].rename(columns={"technology": "reference"})

    # Limit to regions & countries REMIND-EU is configured for
    df = df.query("region.isin(@region_mapping['REMIND-EU'])", engine="python")

    # To calculate weighted values for technologies / other cost related parameters,
    # first load the weights calculated in REMIND-EU
    weights = pd.concat(
        [
            read_remind_data(
                file_path=snakemake.input["remind_data"],
                variable_name="p32_weightGen",
                rename_columns={
                    "ttot": "year",
                    "all_regi": "region",
                    "all_te": "technology",
                    "value": "weight",
                },
            ),
            read_remind_data(
                file_path=snakemake.input["remind_data"],
                variable_name="p32_weightPEprice",
                rename_columns={
                    "ttot": "year",
                    "all_regi": "region",
                    "all_enty": "technology",
                    "value": "weight",
                },
            ),
            read_remind_data(
                file_path=snakemake.input["remind_data"],
                variable_name="p32_weightStor",
                rename_columns={
                    "ttot": "year",
                    "all_regi": "region",
                    "all_te": "technology",
                    "value": "weight",
                },
            ),
        ]
    ).query(
        "`region`.isin(@region_mapping['REMIND-EU']) and year == '{}'".format(
            snakemake.wildcards["year"]
        ),
        engine="python",
    )[
        ["region", "technology", "weight"]
    ]

    # Merge weights to REMIND-EU technology data
    df = df.merge(
        weights,
        left_on=["region", "reference"],
        right_on=["region", "technology"],
        how="left",
    )[["region", "reference", "parameter", "value", "unit", "weight"]]


    # Create new technology data for all technologies which are mapped and generation weighted
    mapped_technologies = mapped_technologies.merge(
        df, on=["reference", "parameter"], how="left"
    )

    # Some parameters are not reported by REMIND-EU if they are not build, e.g. efficiency.
    # After merge these values will be NaN, fill here with 0
    mapped_technologies["weight"] = mapped_technologies["weight"].fillna(0.0)

    # Generation is reported in TWa and reported as 0 for technologies which are not built;
    # by adding a small value, we can avoid NaN values in the weighted aggregation
    # Value added is 1 MWh = 1e-6/8760 TWa
    # This step also ensures that weights are available for all technologies
    mapped_technologies["weight"] += 1e-6 / 8760


    # Helper function used together with pd.DataFrame.apply:
    # * Calculate weighted value per technology and parameter
    # * Determine unit (which should be identical for all aggregated values)
    def calculate_weighted_value(x):
        result = {
            "weighted_value": (x["value"] * x["weight"]).sum(skipna=False)
            / x["weight"].sum(),
            "unit": x["unit"].unique()[0],
        }

        assert (
            len(x["unit"].unique()) == 1
        ), f'Multiple units per parameter detected. Check: {x[["PyPSA-EUR technology", "parameter"]]}'

        return pd.Series(result, index=["weighted_value", "unit"])


    # Calculate electricity-generation weighted technology parameter (weighted across regions and different REMIND-EU technologies per single PyPSA-EUR technology)
    mapped_technologies = mapped_technologies.groupby(
        ["PyPSA-EUR technology", "parameter"]
    ).apply(calculate_weighted_value)
    mapped_technologies = mapped_technologies.reset_index().rename(
        columns={"weighted_value": "value", "PyPSA-EUR technology": "technology"}
    )
    mapped_technologies["source"] = "REMIND-EU"
    mapped_technologies[
        "further description"
    ] = "Extracted from REMIND-EU model in 'import_REMIND_costs.py' script"

    #%%
    # +++ 2. Technologies where values are taken from PyPSA-EUR default values
    pypsa_technologies = technology_mapping.query(
        "`couple to` == 'mapping to PyPSA-EUR default values'"
    ).drop(columns=["unit"])

    # Load original costs used by standard PyPSA-EUR model used as calculation basis
    original_cost = pd.read_csv(snakemake.input["original_costs"])

    # Merge with original costs
    pypsa_technologies = pypsa_technologies.merge(
        original_cost,
        left_on=["PyPSA-EUR technology", "parameter"],
        right_on=["technology", "parameter"],
        how="left",
        validate="one_to_one",
    )

    pypsa_technologies["source"] = "PyPSA-EUR"
    pypsa_technologies[
        "further description"
    ] = "Default parameter from PyPSA-EUR model in 'import_REMIND_costs.py' script"

    # Only keep relevant columns
    pypsa_technologies = pypsa_technologies[
        ["technology", "parameter", "value", "unit", "source", "further description"]
        ]

    #%%
    # +++ 3. Technologies where values are scaled based on a proxy technology +++
    # In 2025 the original cost assumptions from PyPSA-EUR are used
    # In later years the costs are scaled based on the cost developmentin REMIND-EU for a reference technology
    scaled_technologies = technology_mapping.query(
        "`couple to` == 'scaling original values based on reference PyPSA-EUR technology'"
    ).drop(columns=["unit"])

    # Assert that the parameter column is only investment, otherwise raise an error
    assert (
        scaled_technologies["parameter"] == "investment"
    ).all(), "Only investment costs can be scaled based on the reference technology"

    # Get costs of the reference technology in REMIND-EU from remind_data
    reference_cost_improvement = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_capCostwAdjCost",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "all_te": "technology",
        }
    # Only include the first_year or wildcard year in order to calculate the cost decrease of the reference technology
    )

    # Get first year of REMIND coupling data
    years_coupled = reference_cost_improvement.year.unique().tolist()
    first_year = int(min(years_coupled))

    reference_cost_improvement = reference_cost_improvement.query("year in ['{}', '{}']".format(first_year, snakemake.wildcards["year"]))

    # Load original costs in first year
    if first_year == 2025:
        original_cost = pd.read_csv(snakemake.input["original_costs_2025"])
    elif first_year == 2030:
        original_cost = pd.read_csv(snakemake.input["original_costs_2030"])
    else:
        raise ValueError(f"Year {first_year} not supported for original costs")

    # Get costs of the reference technology in REMIND-EU from remind_data
    reference_cost_improvement = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_capCostwAdjCost",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "all_te": "technology",
        }
    # Only include the first_year or wildcard year in order to calculate the cost decrease of the reference technology
    ).query("year in ['{}', '{}']".format(first_year, snakemake.wildcards["year"]))

    # Calculate the cost decrease of the reference technology from 2025 to the year of interest
    reference_cost_improvement = reference_cost_improvement.pivot_table(
        index="technology", columns="year", values="value", observed=False
    ).reset_index()
    reference_cost_improvement["cost_decrease"] = (
        reference_cost_improvement[snakemake.wildcards["year"]] / reference_cost_improvement[str(first_year)]
    )

    # Drop the year columns which are not needed anymore
    reference_cost_improvement = reference_cost_improvement.set_index("technology")[["cost_decrease"]]

    # Use original costs, scale them by the cost decrease of the reference technology and use them as new costs
    df = scaled_technologies.merge(
        original_cost,
        left_on=["PyPSA-EUR technology", "parameter"],
        right_on=["technology", "parameter"],
        how="left",
        validate="one_to_one"
    ).rename(
        columns={"value": "original_value", "unit": "original_unit"}
    )

    # Merge with the cost decrease of the reference technology
    df = df.merge(
        reference_cost_improvement,
        left_on="reference",
        right_on="technology",
        how="left",
        validate="many_to_one"
    )

    # Calculate the new value
    df["value"] = df["original_value"] * df["cost_decrease"]

    # Add further description
    df["source"] = "REMIND-EU and PyPSA-EUR"
    df["further description"] = (
        "Original value from PyPSA-EUR in " 
        + str(first_year)
        + " scaled by REMIND cost development until "
        + snakemake.wildcards["year"]
        + " for technology: " + df["reference"]
    )

    # Only keep relevant columns
    scaled_technologies = df[
        ["PyPSA-EUR technology", "parameter", "value", "original_unit", "source", "further description"]
    ].rename(
        columns={"PyPSA-EUR technology": "technology", "original_unit": "unit"}
    )

    #%%
    # +++ 4. Technologies where values are set in the technology mapping config file +++
    set_technologies = technology_mapping.query(
        "`couple to` == 'setting to reference value'"
    ).rename(
        columns={
            "PyPSA-EUR technology": "technology",
            "reference": "value",
            "comment": "further description",
        }
    )[
        ["technology", "parameter", "value", "unit", "further description"]
    ]
    set_technologies[
        "source"
    ] = f"Set via configuration file: {snakemake.input['technology_cost_mapping']}"
    set_technologies["further description"] = set_technologies[
        "further description"
    ].fillna("")

    #%%
    # Combine all technologies
    costs = pd.concat([mapped_technologies, pypsa_technologies, scaled_technologies, set_technologies])

    #%%
    # +++ 5. Add discount rate +++
    # Discount rate is calculated on REMIND-EU side and just needs to be added for all technologies
    # By adding the discount rate after the 4. step, we allow the discount rate to be overwritten in the technology mapping config file

    # Get technologies with and without "discount rate"
    discount_rate_technologies = costs.loc[
        costs["parameter"] == "discount rate", "technology"
    ]
    technologies_without_discount_rate = costs.loc[
        ~costs["technology"].isin(discount_rate_technologies)
    ]

    logger.info("... extracting discount rate")
    discount_rate = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_discountRate",
        rename_columns={
            "ttot": "year",
        },
    ).query("year == '{}'".format(snakemake.wildcards["year"]))

    assert (
        discount_rate.shape[0] == 1
    ), "Multiple discount rates instead of a single value found"

    # Construct dataframe with discount rate for all technologies by cartesian product
    discount_rate = (
        pd.Series(
            {
                "parameter": "discount rate",
                "value": discount_rate["value"].item(),
                "unit": "p.u.",
                "source": "REMIND-EU",
                "further description": "p32_discountRate",
            }
        )
        .to_frame()
        .T
    )

    discount_rate = discount_rate.merge(
        technologies_without_discount_rate[["technology"]].drop_duplicates(), how="cross"
    )

    # Add discount rate to costs
    costs = pd.concat([costs, discount_rate])

    #%%
    # Special case: Convert electrolysis capex from USD/MW_H2 (output, in REMIND) to USD/MW_el (input, in PyPSA)
    # Note: For hydrogen turbines, the correction from USD/MW_el (output, in REMIND) to USD/MW_H2 (input, in PyPSA)
    # is done in add_extra_components by default
    tech = "electrolysis"
    costs.loc[
        (costs["technology"] == tech) & (costs["parameter"] == "investment"), "value"
    ] /= costs.loc[
        (costs["technology"] == tech) & (costs["parameter"] == "efficiency"), "value"
    ].values
    logger.info(f"Corrected investment costs for {tech} from MW_H2 output to MW_el input.") 

    # Special case: Convert battery inverter capex from USD/MW_e (output, in REMIND) to USD/MW_e (input, in PyPSA)
    tech = "battery inverter"
    costs.loc[
        (costs["technology"] == tech) & (costs["parameter"] == "investment"), "value"
    ] /= costs.loc[
        (costs["technology"] == tech) & (costs["parameter"] == "efficiency"), "value"
    ].values
    logger.info(f"Corrected investment costs for {tech} from MW_e output to MW_e input.")

    # Output to file
    costs.to_csv(snakemake.output["costs"], index=False)

    # %%
    # list all rows in r with nan values inside
    if not (nan_rows := costs[costs.isna().any(axis=1)]).empty:
        raise ValueError(f"NaN values in costs detected: {nan_rows}")

    # Make sure no duplicates for (technology, parameter) exist
    if not (
        duplicates := costs.where(
            costs.duplicated(subset=["technology", "parameter"], keep=False)
        ).dropna()
    ).empty:
        raise ValueError(
            f"Duplicate values for (technology, parameter) detected: {duplicates}"
        )
