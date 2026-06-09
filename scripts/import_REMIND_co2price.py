"""Import a year-level CO2 price pathway from REMIND, per REMIND region."""

import logging

import pandas as pd
from _helpers import (
    configure_logging,
    get_region_mapping,
    mock_snakemake,
    read_remind_data,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "import_REMIND_co2price",
            scen_REMIND="PkBudg1000_EU",
            iter_REMIND="1",
            configfiles="config/config.remind_europe.yaml",
        )

    configure_logging(snakemake)
    logger.info("Building REMIND CO2 price pathway")

    countries = set(snakemake.config["countries"])
    full_mapping = get_region_mapping(snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU")
    mapped_regions = {r for c, rs in full_mapping.items() if c in countries for r in rs if r}

    co2_price = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p_priceCO2",
        rename_columns={
            "tall": "year",
            "all_regi": "region",
        },
    )

    # unit conversion from USD/tC to USD/tCO2
    co2_price["value"] *= 12 / (12 + 2 * 16)

    years_coupled = (
        read_remind_data(
            file_path=snakemake.input["remind_data"],
            variable_name="t",
            rename_columns={"ttot": "year"},
        )
        .year.unique()
        .tolist()
    )
    logger.info("Read coupled years from t set in GDX.")

    co2_price = co2_price.loc[co2_price["region"].isin(mapped_regions)].copy()
    co2_price["year"] = co2_price["year"].astype(int)
    co2_price = (
        co2_price.set_index(["region", "year"])["value"]
        .reindex(
            pd.MultiIndex.from_product(
                [sorted(mapped_regions), list(map(int, years_coupled))],
                names=["region", "year"],
            ),
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"value": "co2_price"})
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    logger.info(
        "CO2 prices per region for coupled years:\n%s",
        co2_price.pivot(index="year", columns="region", values="co2_price").round(1).to_string(),
    )

    logger.info("Exporting CO2 price pathway to %s", snakemake.output["co2_price"])
    co2_price.to_csv(snakemake.output["co2_price"], index=False)
