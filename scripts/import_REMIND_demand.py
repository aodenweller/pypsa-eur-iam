"""Read sectoral electricity demand from REMIND and export it for use in PyPSA-Eur.

Thin wrapper over rpycpl: reads the load-sector symbol (``v32_load_sector`` with
``p32_load_sector`` fallback, resolved from the central symbol config), converts TWa to
annual MWh, and filters to REMIND regions overlapping the configured countries.
"""

import logging

from _helpers import configure_logging, get_region_mapping, mock_snakemake
from rpycpl.io import RemindLoader
from rpycpl.symbols import load_frame, load_symbol_specs
from rpycpl.transforms.loads import convert_loads

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
    logger.info("Loading REMIND demand data via rpycpl ...")

    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
    )
    mapped_regions = sorted({r for rs in region_mapping.values() for r in rs if r})

    loader = RemindLoader(snakemake.input["remind_data"])
    symbols = load_symbol_specs()

    raw = load_frame(loader, symbols["load_sector"])
    demand = convert_loads(raw, regions=mapped_regions)

    demand.to_csv(snakemake.output["sectoral_load"], index=False)
    logger.info("Wrote %s rows of REMIND demand to %s", len(demand), snakemake.output["sectoral_load"])
