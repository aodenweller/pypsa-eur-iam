"""Import a year-level CO2 price pathway from REMIND, per REMIND region.

Thin wrapper over rpycpl: reads the CO2-price symbol (resolved from the central symbol
config), converts tC->tCO2, filters to mapped REMIND regions and reindexes to the coupled
years (missing filled with 0). Output identical to the previous standalone implementation.
"""

import logging

from _helpers import configure_logging, get_region_mapping, mock_snakemake
from rpycpl.io import RemindLoader
from rpycpl.symbols import load_frame, load_symbol_specs
from rpycpl.transforms.co2_prices import convert_co2_prices, extract_co2_prices

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
    logger.info("Building REMIND CO2 price pathway via rpycpl")

    countries = set(snakemake.config["countries"])
    full_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
    )
    mapped_regions = sorted({r for c, rs in full_mapping.items() if c in countries for r in rs if r})

    loader = RemindLoader(snakemake.input["remind_data"])
    symbols = load_symbol_specs()  # EUR default section

    raw = load_frame(loader, symbols["co2_price"])
    coupled_years = sorted(load_frame(loader, symbols["coupled_years"])["year"].astype(int).unique())

    co2_price = convert_co2_prices(
        extract_co2_prices(raw, regions=mapped_regions, years=coupled_years),
        currency_factor=1.0,
    )
    co2_price = (
        co2_price.rename(columns={"value": "co2_price"})[["region", "year", "co2_price"]]
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    logger.info("Exporting CO2 price pathway to %s", snakemake.output["co2_price"])
    co2_price.to_csv(snakemake.output["co2_price"], index=False)
