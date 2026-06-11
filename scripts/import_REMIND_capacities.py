"""Read installed-capacity targets from REMIND and export them as PyPSA-Eur lower bounds.

Thin wrapper over ``RemindEurAdapter.determine_must_build_capacity``: reads the capacity symbol
(TW -> MW), merges VRE-coupled variants, scales battery techs, adjusts link-like techs to
input-capacity basis, and maps REMIND techs to PyPSA-Eur carriers. Output identical to the
previous standalone implementation: [year, region_REMIND, carrier, p_nom_min].
"""

import logging

from _helpers import (
    configure_logging,
    get_technology_mapping,
    mock_snakemake,
)
from remind.adapter_remind_eur import LINK_TECHS, RemindEurAdapter
from rpycpl.io import RemindLoader
from rpycpl.io.remind_symbols import load_symbol_specs
from rpycpl.transforms.mapping import read_region_map as get_region_mapping

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
    logger.info("Building REMIND capacity targets via rpycpl adapter")

    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
    )
    mapped_regions = sorted({r for rs in region_mapping.values() for r in rs if r})
    tech_map = get_technology_mapping(snakemake.input["technology_cost_mapping"])

    # Bind each argument to a named variable (no inline function calls) so the adapter
    # inputs can be inspected when debugging.
    loader = RemindLoader(snakemake.input["remind_data"])
    symbols = load_symbol_specs()
    coupling_config = {"link_techs": LINK_TECHS}
    adapter = RemindEurAdapter(
        loader=loader,
        symbols=symbols,
        region_map={},
        config=coupling_config,
        remind_regions=mapped_regions,
    )

    capacities = adapter.determine_must_build_capacity(tech_map)
    capacities = (
        capacities[capacities["region"].isin(mapped_regions)]
        .rename(columns={"region": "region_REMIND"})[["year", "region_REMIND", "carrier", "p_nom_min"]]
        .sort_values(["year", "region_REMIND", "carrier"])
        .reset_index(drop=True)
    )

    capacities.to_csv(snakemake.output["capacities"], index=False)
    logger.info("Wrote %d capacity-target rows to %s", len(capacities), snakemake.output["capacities"])
