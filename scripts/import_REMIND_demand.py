"""Read sectoral electricity demand from REMIND and export it for use in PyPSA-Eur.

Stage 1 of the demand pipeline. Thin wrapper over ``RemindEurAdapter.build_regional_demand``:
reads the load-sector symbol (``v32_load_sector`` with ``p32_load_sector`` fallback, resolved
from the central symbol config), converts TWa to annual MWh, and filters to the REMIND regions
overlapping the configured countries. ``downscale_REMIND_demand`` then splits this to countries.
"""

import logging

from _helpers import configure_logging, mock_snakemake
from remind.adapter_remind_eur import RemindEurAdapter
from rpycpl.io import RemindLoader
from rpycpl.io.remind_symbols import load_symbol_specs
from rpycpl.transforms.mapping import read_region_map as get_region_mapping

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "import_REMIND_demand",
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)
    logger.info("Loading REMIND regional demand via the rpycpl adapter ...")

    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
    )
    mapped_regions = sorted({r for rs in region_mapping.values() for r in rs if r})

    adapter = RemindEurAdapter(
        loader=RemindLoader(snakemake.input["remind_data"]),
        symbols=load_symbol_specs(),
        region_map={},
        config={},
        remind_regions=mapped_regions,
    )
    demand = adapter.build_regional_demand()

    demand.to_csv(snakemake.output["sectoral_load"], index=False)
    logger.info("Wrote %s rows of REMIND demand to %s", len(demand), snakemake.output["sectoral_load"])
