"""
Import a year-level CO2 price pathway from REMIND, per REMIND region.

Reads the CO2-price symbol (resolved from the central symbol config; tC->tCO2 applied by
``load_frame`` via the symbol's ``to_unit``), filters to the mapped REMIND regions and reindexes
to REMIND's coupled-year set (missing filled with 0). The extract/convert chain lives in the
coupling package via ``CouplingAdapter.build_co2_prices(years=...)``.
"""

import logging

from _helpers import configure_logging, mock_snakemake
from rpycpl.adapters import RemindGdxAdapter, RemindIamcAdapter
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
    logger.info("Building REMIND CO2 price pathway from the rpycpl symbol config")

    countries = set(snakemake.config["countries"])
    full_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
    )
    mapped_regions = sorted(
        {r for c, rs in full_mapping.items() if c in countries for r in rs if r}
    )

    loader = RemindLoader(snakemake.input["remind_data"])
    symbols = load_symbol_specs(backend=loader.backend)

    # GDX: reindex to REMIND's coupled-year set (symbol `t`).
    # IAMC: mif starts in 2005 so we can't infer coupling years from the data; use config instead.
    if "coupled_years" in symbols:
        coupled_years = sorted(
            load_frame(loader, symbols["coupled_years"])["year"].astype(int).unique()
        )
    else:
        coupled_years = sorted(int(y) for y in snakemake.config["remind_coupling"]["years"])

    # Build the pathway via the coupling package so the transform chain lives in one place.
    adapter_cls = RemindIamcAdapter if loader.backend == "iamc" else RemindGdxAdapter
    adapter = adapter_cls(
        loader, symbols, region_map={}, config={}, remind_regions=mapped_regions
    )
    co2_price = adapter.build_co2_prices(years=coupled_years)  # tC -> tCO2 applied here
    co2_price = (
        co2_price.rename(columns={"value": "co2_price"})[
            ["region", "year", "co2_price"]
        ]
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    logger.info("Exporting CO2 price pathway to %s", snakemake.output["co2_price"])
    co2_price.to_csv(snakemake.output["co2_price"], index=False)
