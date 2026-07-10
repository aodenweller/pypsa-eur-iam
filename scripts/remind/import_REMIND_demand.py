"""
Read sectoral electricity demand from REMIND and export it for use in PyPSA-Eur.

Stage 1 of the demand pipeline. Reads regional sectoral electricity demand via the
``Coupler`` (backend-selected by ``RemindLoader``):

- GDX backend: reads ``load_sector`` symbol (``v32_load_sector`` / ``p32_load_sector``),
  converts TWa→MWh via the symbol spec.
- IAMC backend: derives demand from SE|Electricity, transmission losses, and FE sector
  variables, applying an implicit T&D efficiency and computing an AC residual for
  untracked loads.

``downscale_REMIND_demand`` splits this to countries (Stage 2, backend-agnostic).
"""

import logging

from _helpers import configure_logging, mock_snakemake
from iampypsa import RemindGdxCoupler, RemindIamcCoupler
from iampypsa.couplers.remind import read_region_map as get_region_mapping
from iampypsa.io import RemindLoader
from iampypsa.io.remind_symbols import load_symbol_specs

logger = logging.getLogger(__name__)

# Which coupler handles each REMIND output backend (selected from loader.backend below).
REMIND_COUPLERS = {"gdx": RemindGdxCoupler, "iamc": RemindIamcCoupler}


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "import_REMIND_demand",
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)
    logger.info("Loading REMIND regional demand ...")

    countries = set(snakemake.params["countries"])
    region_mapping = get_region_mapping(source="country", target="model_region")
    mapped_regions = sorted({r for c, rs in region_mapping.items() if c in countries for r in rs if r})

    loader = RemindLoader(snakemake.input["remind_data"])
    symbols = load_symbol_specs(backend=loader.backend)

    coupler_cls = REMIND_COUPLERS[loader.backend]
    coupler = coupler_cls(
        loader, symbols,
        region_map={},
        config={},
        model_regions=mapped_regions,
    )
    demand = coupler.build_regional_demand()

    # Drop demand_h2 (different unit, should not get folded into AC)
    demand = demand[demand["sector"] != "demand_h2"]

    years = snakemake.params["years"]
    demand = demand[demand["year"].isin(years) & demand["region"].isin(mapped_regions)]

    demand.to_csv(snakemake.output["sectoral_load"], index=False)
    logger.info(
        "Wrote %d rows of REMIND demand (%s backend) to %s",
        len(demand),
        loader.backend,
        snakemake.output["sectoral_load"],
    )
