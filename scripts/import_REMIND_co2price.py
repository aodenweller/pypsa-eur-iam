"""Import a year-level CO2 price pathway from REMIND, per REMIND region.

Thin wrapper over ``RemindEurAdapter.build_co2_prices``: reads the CO2-price symbol (resolved
from the central symbol config), converts tC->tCO2, filters to the mapped REMIND regions and
reindexes to REMIND's coupled-year set (missing filled with 0). Output identical to the
previous standalone implementation.
"""

import logging

from _helpers import configure_logging, mock_snakemake
from remind.adapter_remind_eur import RemindEurAdapter
from rpycpl.io import RemindLoader
from rpycpl.io.remind_symbols import load_frame, load_symbol_specs
from rpycpl.transforms.mapping import read_region_map as get_region_mapping

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
    logger.info("Building REMIND CO2 price pathway via the rpycpl adapter")

    countries = set(snakemake.config["countries"])
    full_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
    )
    mapped_regions = sorted({r for c, rs in full_mapping.items() if c in countries for r in rs if r})

    loader = RemindLoader(snakemake.input["remind_data"])
    symbols = load_symbol_specs()

    # CO2 prices are reindexed to REMIND's coupled-year set (the `coupled_years`/`t` symbol);
    # the adapter uses ``config["planning_horizons"]`` as that year grid.
    coupled_years = sorted(load_frame(loader, symbols["coupled_years"])["year"].astype(int).unique())
    adapter = RemindEurAdapter(
        loader=loader,
        symbols=symbols,
        region_map={},
        config={"planning_horizons": coupled_years, "currency_factor": 1.0},
        remind_regions=mapped_regions,
    )

    co2_price = adapter.build_co2_prices()
    co2_price = (
        co2_price.rename(columns={"value": "co2_price"})[["region", "year", "co2_price"]]
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    logger.info("Exporting CO2 price pathway to %s", snakemake.output["co2_price"])
    co2_price.to_csv(snakemake.output["co2_price"], index=False)
