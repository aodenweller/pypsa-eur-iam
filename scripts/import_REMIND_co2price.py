
"""Import a year-level CO2 price pathway from REMIND."""

import logging

from _helpers import configure_logging, get_region_mapping, read_remind_data

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_co2price",
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
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
            variable_name="tPy32",
            rename_columns={"ttot": "year"},
        )
        .year.unique()
        .tolist()
    )

    # Calculate mean CO2 price across all regions overlapping between REMIND and PyPSA-EUR countries for each year
    # TODO: Implement regional prices in PyPSA!
    co2_price = (
        co2_price.loc[co2_price["region"].isin(mapped_regions)]
        .groupby("year", observed=False)["value"]
        .mean()
    )

    co2_price.index = co2_price.index.astype(int)
    co2_price = (
        co2_price.reindex(list(map(int, years_coupled)), fill_value=0)
        .to_frame("co2_price")
        .reset_index()
    )

    co2_price = co2_price.sort_values("year").reset_index(drop=True)

    logger.info("Exporting CO2 price pathway to %s", snakemake.output["co2_price"])
    co2_price.to_csv(snakemake.output["co2_price"], index=False)
