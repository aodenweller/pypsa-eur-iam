"""
Read installed-capacity targets from REMIND and export them as PyPSA-Eur lower bounds.

Thin wrapper over ``iampypsa.transforms.capacities.build_capacity_targets``: reads the capacity
spec (unit conversion handled by the spec), applies any ``consolidation`` block declared on the
spec (VRE-variant merge, battery scaling, link output→input adjustment), and maps model tech
tokens to PyPSA-Eur carriers.

Output columns: [year, region_REMIND, carrier, value, unit]
  - value: input-basis capacity in MW (link-like techs already divided by efficiency)
  - unit: "MW" for all rows
"""

import logging

from _helpers import (
    configure_logging,
    get_technology_mapping,
    mock_snakemake,
)
from iampypsa.io import RemindLoader
from iampypsa.io.remind_symbols import load_symbol_specs
from iampypsa.transforms.capacities import build_capacity_targets
from iampypsa.transforms.mapping import read_region_map as get_region_mapping

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "import_REMIND_capacities",
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)
    logger.info("Building REMIND capacity targets via iampypsa.build_capacity_targets")

    countries = set(snakemake.params["countries"])
    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="country", target="model_region"
    )
    mapped_regions = sorted({r for c, rs in region_mapping.items() if c in countries for r in rs if r})
    tech_map = get_technology_mapping(snakemake.input["technology_mapping"])

    loader = RemindLoader(snakemake.input["remind_data"])
    symbols = load_symbol_specs(backend=loader.backend)

    years = snakemake.params["years"]

    capacities = build_capacity_targets(
        loader, symbols, mapped_regions, tech_map,
        map_tech_col="REMIND-EU", map_carrier_col="PyPSA-Eur",
    )
    capacities = (
        capacities.rename(columns={"region": "region_REMIND"})[
            ["year", "region_REMIND", "carrier", "value", "unit"]
        ]
        .query("year in @years and region_REMIND in @mapped_regions")
        .sort_values(["year", "region_REMIND", "carrier"])
        .reset_index(drop=True)
    )

    capacities.to_csv(snakemake.output["capacities"], index=False)
    logger.info(
        "Wrote %d capacity-target rows to %s",
        len(capacities),
        snakemake.output["capacities"],
    )
