"""
Export PyPSA-Eur optimisation results back to REMIND.

Reads all solved PyPSA networks for one coupling iteration and computes the
quantities REMIND uses to update its investment decisions in the next
iteration: optimal installed capacities, capacity factors, electricity prices,
markup metrics, peak residual loads, renewable potentials, and grid losses.

Results are aggregated to REMIND regions and technology codes and written to:
  - A GDX file (PyPSAEUR2REMIND.gdx) read by REMIND on its next solve
  - CSV files in coupling_parameters/ (identical content to GDX)
  - CSV files in reporting_parameters/ (additional diagnostics, not used by REMIND)
"""

import copy
import logging
import os

import gamspy as gt
import numpy as np
import pandas as pd
import pypsa
import yaml
from _helpers import configure_logging, get_region_mapping
from scipy.stats import zscore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GDX parameter metadata: {key: {name, description, dims}}
# dims must match column names in the accumulated DataFrames (after renaming).
# ---------------------------------------------------------------------------
GDX_PARAMS = {
    "capacity_factors": {
        "name": "p32_PyPSA_CF",
        "description": "Capacity factors of generators and links [1]",
        "dims": ["year", "region", "carrier"],
        "rename": {"remind_carrier": "carrier"},
    },
    "optimal_capacities": {
        "name": "p32_PyPSA_OptCap",
        "description": "Optimal capacities, links w.r.t. input bus [MW or MWh]",
        "dims": ["year", "region", "carrier"],
        "rename": {"remind_carrier": "carrier"},
    },
    "potentials": {
        "name": "p32_PyPSA_Potential",
        "description": "Maximum installable VRE capacity [MW]",
        "dims": ["year", "region", "carrier"],
        "rename": {"remind_carrier": "carrier"},
    },
    "generation_shares": {
        "name": "p32_PyPSA_shPe2seel",
        "description": "Generation shares of technologies [1]",
        "dims": ["year", "region", "carrier"],
        "rename": {"remind_carrier": "carrier"},
    },
    "markups_supply": {
        "name": "p32_PyPSA_MarkupSupply",
        "description": "Supply-side markups: market value minus average price [$/MWh]",
        "dims": ["year", "region", "carrier"],
        "rename": {"remind_carrier": "carrier"},
    },
    "market_values": {
        "name": "p32_PyPSA_MarketValueSupply",
        "description": "Market values of supply-side generators [$/MWh]",
        "dims": ["year", "region", "carrier"],
        "rename": {"remind_carrier": "carrier"},
    },
    "average_electricity_price": {
        "name": "p32_PyPSA_AverageElectricityPrice",
        "description": "Load-weighted average electricity price [$/MWh]",
        "dims": ["year", "region"],
        "rename": {},
    },
    "sectoral_electricity_prices": {
        "name": "p32_PyPSA_SectoralElectricityPrices",
        "description": "Electricity prices paid by demand-side end-users [$/MWh]",
        "dims": ["year", "region", "carrier"],
        "rename": {"sector": "carrier"},
    },
    "markups_demand": {
        "name": "p32_PyPSA_MarkupDemand",
        "description": "Demand-side markups: sector price minus average price [$/MWh]",
        "dims": ["year", "region", "enduse"],
        "rename": {"sector": "enduse"},
    },
    "peak_residual_load": {
        "name": "p32_PyPSA_PeakResLoadRel",
        "description": "Peak residual load relative to average load [1]",
        "dims": ["year", "region"],
        "rename": {},
    },
    "h2_turbine_generation": {
        "name": "p32_PyPSA_H2TurbRel",
        "description": "H2 fuel cell generation relative to total load [1]",
        "dims": ["year", "region"],
        "rename": {},
    },
    "battery_discharge": {
        "name": "p32_PyPSA_BatteryDischargeRel",
        "description": "Battery discharger generation relative to total load [1]",
        "dims": ["year", "region"],
        "rename": {},
    },
    "grid_losses": {
        "name": "p32_PyPSA_GridLossesRel",
        "description": "AC line losses relative to total load [1]",
        "dims": ["year", "region"],
        "rename": {},
    },
}


# ---------------------------------------------------------------------------
# Mapping and network annotation
# ---------------------------------------------------------------------------


def load_mappings(snakemake):
    """Load carrier→REMIND and country→region mappings; return (map_to_remind, sector_carriers, region_mapping)."""
    with open(snakemake.input["carrier_mapping"]) as f:
        cfg = yaml.safe_load(f)

    map_to_remind = {
        **cfg["generators"],
        **cfg["links"],
        **cfg["stores"],
    }
    sector_carriers = cfg.get("sectors", {})

    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"],
        source="PyPSA-EUR",
        target="REMIND-EU",
        flatten=True,
    )
    return map_to_remind, sector_carriers, region_mapping


def annotate_network(n, map_to_remind, region_mapping):
    """
    Add ``region`` and ``remind_carrier`` columns to all network components.

    ``region`` propagates the REMIND region from the bus the component is
    attached to.  ``remind_carrier`` is the REMIND technology code for the
    component's carrier; components whose carrier is absent from
    ``map_to_remind`` receive NaN and are automatically excluded from any
    ``groupby``-based aggregation - no explicit drop lists needed.
    """
    n.buses["region"] = n.buses["country"].map(region_mapping)

    for comp, bus_col in [
        ("generators", "bus"),
        ("stores", "bus"),
        ("storage_units", "bus"),
        ("loads", "bus"),
        ("links", "bus0"),
        ("lines", "bus0"),
    ]:
        getattr(n, comp)["region"] = getattr(n, comp)[bus_col].map(n.buses["region"])

    # Lines and links span two buses; keep both for grid calculations.
    n.links["region1"] = n.links["bus1"].map(n.buses["region"])
    n.lines["region1"] = n.lines["bus1"].map(n.buses["region"])

    n.generators["remind_carrier"] = n.generators["carrier"].map(map_to_remind)
    n.links["remind_carrier"] = n.links["carrier"].map(map_to_remind)
    n.stores["remind_carrier"] = n.stores["carrier"].map(map_to_remind)

    for comp in ["generators", "links", "stores"]:
        df = getattr(n, comp)
        unmapped = sorted(set(df.loc[df["remind_carrier"].isna(), "carrier"]) - {"load"})
        if unmapped:
            logger.debug("Unmapped %s carriers (excluded from REMIND export): %s", comp, unmapped)


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def get_total_load(n):
    """
    Total electricity consumption for the network in MWh.

    H2 loads are divided by the electrolyser efficiency so they are expressed
    as equivalent electricity demand.
    """
    load_by_carrier = n.statistics.withdrawal(comps="Load", nice_names=False)
    h2_mask = n.links.carrier == "H2 Electrolysis"
    if h2_mask.any() and "H2" in load_by_carrier.index:
        eff = n.links.loc[h2_mask, "efficiency"].values[0]
        load_by_carrier["H2"] /= eff
    return load_by_carrier.sum()


def cutoff_scarcity_prices(n, z_cutoff):
    """
    Return a deep copy of ``n`` with snapshot weightings zeroed for snapshots
    whose mean AC marginal price exceeds ``z_cutoff`` standard deviations.
    """
    n_cut = copy.deepcopy(n)
    ac_buses = n_cut.buses.query("carrier == 'AC'").index
    z = (
        n_cut.buses_t["marginal_price"][ac_buses]
        .apply(zscore)
        .mean(axis="columns")
    )
    z.index = pd.to_datetime(z.index)
    n_cut.snapshot_weightings = n_cut.snapshot_weightings.where(z < float(z_cutoff), 0)

    n_excluded = int(
        n_cut.snapshot_weightings["generators"].shape[0]
        * n_cut.snapshot_weightings["generators"].iloc[0]
        - n_cut.snapshot_weightings["generators"].sum()
    )
    logger.info(
        "Scarcity price cutoff (z=%.1f): excluding %d snapshots.",
        float(z_cutoff),
        n_excluded,
    )
    return n_cut


# ---------------------------------------------------------------------------
# Electricity price calculation
# ---------------------------------------------------------------------------


def calculate_electricity_prices(n, sector_carriers, z_cutoff, hourly=False):
    """
    Load-weighted average electricity price per REMIND region and sector.

    Returns columns [region, sector, value]; sector "total" is the network-wide average.
    ``z_cutoff`` excludes scarcity-price snapshots above that z-score threshold.
    Set ``hourly=True`` for a (T×1) network-wide hourly price series instead.
    """
    n_calc = cutoff_scarcity_prices(n, z_cutoff) if z_cutoff else n

    ac_buses = n_calc.buses.query("carrier == 'AC'").index
    lmp = n_calc.buses_t.marginal_price.reindex(columns=ac_buses).fillna(0.0)
    excl = n_calc.snapshot_weightings.index[
        (n_calc.snapshot_weightings == 0).any(axis=1)
    ]
    lmp.loc[excl] = 0.0

    bus_region = n_calc.buses.loc[ac_buses, "region"]

    def _load_at_buses(sector):
        """Return a (T x AC-buses) DataFrame of electricity consumption for a sector."""
        if sector == "AC":
            # Identify AC loads by the carrier of the bus they're connected to,
            # not by the load's own carrier (which is often an empty string).
            ac_load_idx = n_calc.loads[n_calc.loads.bus.isin(ac_buses)].index
            df = n_calc.loads_t.p_set.reindex(columns=ac_load_idx).fillna(0.0)
            df.columns = n_calc.loads.loc[df.columns, "bus"]
        else:
            carrier = sector_carriers[sector]
            mask = n_calc.links.carrier == carrier
            if not mask.any():
                return pd.DataFrame(0.0, index=n_calc.snapshots, columns=ac_buses)
            df = n_calc.links_t.p0.reindex(columns=n_calc.links.index[mask]).fillna(0.0)
            df.columns = n_calc.links.loc[df.columns, "bus0"]
        # Sum contributions from multiple loads/links at the same bus
        return df.T.groupby(level=0).sum().T.reindex(columns=ac_buses, fill_value=0.0)

    def _wavg_by_region(load_df):
        """Load-weighted average LMP per region."""
        revenue = (load_df * lmp).sum().groupby(bus_region).sum()
        load_total = load_df.sum().groupby(bus_region).sum()
        return (revenue / load_total).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    sector_loads = {s: _load_at_buses(s) for s in ["AC", *sector_carriers]}
    total_load = sum(sector_loads.values())

    if hourly:
        denom = total_load.sum(axis=1).replace(0.0, np.nan)
        return (
            (total_load * lmp).sum(axis=1).div(denom).fillna(0.0)
            .rename("value").reset_index().rename(columns={"index": "snapshot"})
        )

    rows = [
        {"region": region, "sector": sector, "value": float(price)}
        for sector, load_df in {**sector_loads, "total": total_load}.items()
        for region, price in _wavg_by_region(load_df).items()
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coupling calculations (exported to GDX)
# ---------------------------------------------------------------------------


def calculate_capacity_factors(n):
    """Capacity factors grouped by region and REMIND carrier."""
    cf = n.statistics.capacity_factor(
        comps=["Generator", "Link"], groupby=["region", "remind_carrier"]
    )
    return cf.to_frame("value").reset_index().drop(columns=["component"], errors="ignore")


def calculate_optimal_capacities(n, comps, groupby):
    """Optimal installed capacities (p_nom_opt / e_nom_opt)."""
    cap = n.statistics.optimal_capacity(comps=comps, groupby=groupby)
    return cap.to_frame("value").reset_index().drop(columns=["component"], errors="ignore")


def calculate_potentials(n):
    """Maximum installable capacity (p_nom_max) for VRE generators."""
    pot = (
        n.generators.groupby(["region", "remind_carrier"])["p_nom_max"]
        .sum()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return pot.rename("value").reset_index()


def calculate_generation_shares(n):
    """Share of each REMIND technology in total annual generator supply."""
    supply = n.statistics.supply(
        comps=["Generator"], groupby=["region", "remind_carrier"]
    )
    total_by_region = supply.groupby("region").transform("sum")
    shares = (supply / total_by_region).fillna(0.0)
    return shares.rename("value").reset_index().drop(columns=["component"], errors="ignore")


def calculate_markups_supply(n, avg_prices_by_region, z_cutoff):
    """Supply-side markup = market value minus average electricity price per region."""
    n_calc = cutoff_scarcity_prices(n, z_cutoff) if z_cutoff else n
    mv = n_calc.statistics.market_value(
        comps=["Generator"], groupby=["region", "remind_carrier"]
    ).fillna(0.0)
    df = mv.to_frame("value").reset_index().drop(columns=["component"], errors="ignore")
    df["value"] -= df["region"].map(avg_prices_by_region).fillna(0.0)
    return df


def calculate_market_values(n, z_cutoff):
    """Market values (load-weighted electricity revenues) of generators."""
    n_calc = cutoff_scarcity_prices(n, z_cutoff) if z_cutoff else n
    mv = n_calc.statistics.market_value(
        comps=["Generator"], groupby=["region", "remind_carrier"]
    ).fillna(0.0)
    return mv.to_frame("value").reset_index().drop(columns=["component"], errors="ignore")


def calculate_markups_demand(electricity_prices):
    """Demand-side markup = sector electricity price minus average electricity price."""
    avg = (
        electricity_prices.query("sector == 'total'")
        .set_index("region")["value"]
    )
    df = electricity_prices.query("sector != 'total'").copy()
    df["value"] -= df["region"].map(avg).fillna(0.0)
    return df.reset_index(drop=True)


def calculate_peak_residual_loads(n, kind):
    """
    Peak residual load, absolute [MW] or relative to average load [1].

    Dispatchable generators are those without a ``p_max_pu`` time series
    (i.e. not VRE). Load-shedding generators are excluded.
    """
    dispatchable = set(n.generators.index) - set(n.generators_t.p_max_pu.columns)
    n.generators["dispatchable"] = n.generators.index.isin(dispatchable)
    n.generators.loc[
        n.generators.index.str.lower().str.contains("load"), "dispatchable"
    ] = False

    residual_load = (
        n.statistics.energy_balance(
            comps=["Generator"],
            bus_carrier="AC",
            groupby=["region", "dispatchable"],
            aggregate_time=False,
        )
        .groupby(["region", "dispatchable"])
        .sum()
    )

    # Per-region average load for the relative metric
    avg_load = (
        n.statistics.withdrawal(comps="Load", groupby="region", nice_names=False)
        / len(n.snapshots)
    )

    def _prl(group):
        if True not in group.index.get_level_values("dispatchable"):
            return pd.Series({"absolute": 0.0, "relative": 0.0})
        disp = group.xs(True, level="dispatchable")
        peak_snap = disp.idxmax(axis="columns").iloc[0]
        peak_abs = float(disp[peak_snap].iloc[0])
        region = group.index.get_level_values("region")[0]
        avg = float(avg_load.get(region, 1.0))
        return pd.Series({
            "absolute": peak_abs,
            "relative": peak_abs / avg if avg > 0.0 else 0.0,
        })

    prl = residual_load.groupby("region").apply(_prl).reset_index()
    return prl[["region", kind]]


def calculate_link_generation(n, carrier, kind):
    """Absolute [MWh] or relative generation of links with a given carrier (kind: 'absolute'|'relative')."""
    regions = sorted(n.buses["region"].dropna().unique())
    if not (n.links.carrier == carrier).any():
        return pd.DataFrame({"region": regions, kind: 0.0})

    try:
        supply = (
            n.statistics.supply(
                comps=["Link"], groupby=["region", "carrier"], nice_names=False
            )
            .xs("Link", level="component")
            .xs(carrier, level="carrier")
        )
    except KeyError:
        return pd.DataFrame({"region": regions, kind: 0.0})

    total_load = get_total_load(n)
    df = pd.DataFrame({
        "region": supply.index,
        "absolute": supply.values,
        "relative": (supply / total_load).fillna(0.0).values,
    })
    return df[["region", kind]]


def calculate_grid_losses(n, kind):
    """AC line losses, absolute [MWh] or relative to total load [1]."""
    losses = n.statistics.energy_balance(comps="Line", groupby="region").abs()
    if losses.empty:
        regions = sorted(n.buses["region"].dropna().unique())
        losses = pd.Series(0.0, index=pd.Index(regions, name="region"))

    total_load = get_total_load(n)
    df = pd.DataFrame({
        "absolute": losses,
        "relative": losses / total_load,
    }).reset_index()
    return df[["region", kind]]


# ---------------------------------------------------------------------------
# Reporting calculations (CSV only, not exported to GDX)
# ---------------------------------------------------------------------------


def calculate_energy_balance(n, groupby, bus_carrier, comps=None):
    """Energy balance for a given bus carrier."""
    eb = n.statistics.energy_balance(
        comps=comps, groupby=groupby, bus_carrier=bus_carrier, nice_names=False
    )
    return eb.to_frame("value").reset_index()


def calculate_curtailments(n):
    """Curtailment of generators."""
    curt = n.statistics.curtailment(
        comps=["Generator"], groupby=["region", "carrier"], nice_names=False
    )
    return curt.to_frame("value").reset_index()


def calculate_grid_flows(n):
    """Transmission flows for AC lines and DC links."""
    flows = n.statistics.transmission(
        comps=["Link", "Line"], groupby=["region", "carrier"]
    )
    return flows.to_frame("value").reset_index()


def calculate_grid_capacities(n):
    """Transmission capacity in TW*km for AC lines and DC links."""
    n.lines["twkm"] = n.lines["s_nom_opt"] * n.lines["length"] * 1e-6
    ac = n.lines.groupby(["region", "carrier"])["twkm"].sum().rename("value")

    n.links["twkm"] = n.links["p_nom_opt"] * n.links["length"] * 1e-6
    dc = (
        n.links.query("carrier == 'DC'")
        .groupby(["region", "carrier"])["twkm"]
        .sum()
        .rename("value")
    )
    return pd.concat([ac, dc]).rename("value").reset_index()


# ---------------------------------------------------------------------------
# Accumulation and export helpers
# ---------------------------------------------------------------------------


def _acc(store, key, year, df):
    """Prepend a ``year`` column and concat ``df`` into ``store[key]``."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return
    df = df.copy()
    df.insert(0, "year", year)
    if key in store:
        store[key] = pd.concat([store[key], df], ignore_index=True)
    else:
        store[key] = df


def write_gdx(coupling, path, map_to_remind, sector_carriers):
    """Write all coupling parameters to a GDX file."""
    gdx = gt.Container()

    # Derive set records from actual data so the sets are always consistent
    all_years = sorted(
        {int(v) for df in coupling.values() for v in df["year"].unique()}
    )
    all_regions = sorted(
        {r for df in coupling.values() if "region" in df.columns
         for r in df["region"].dropna().unique()}
    )
    all_carriers = sorted(set(map_to_remind.values()) | set(sector_carriers.keys()))
    all_enduses = sorted(sector_carriers.keys())

    gt.Set(gdx, "year", records=all_years,
           description="Years solved by PyPSA-Eur")
    gt.Set(gdx, "region", records=all_regions,
           description="REMIND regions for which PyPSA networks were solved")
    gt.Set(gdx, "carrier", records=all_carriers,
           description="REMIND technology codes and demand sectors")
    gt.Set(gdx, "enduse", records=all_enduses,
           description="Demand-side end-use sectors")

    for key, df in coupling.items():
        meta = GDX_PARAMS[key]
        df_out = df.rename(columns=meta.get("rename", {}))
        gt.Parameter(
            gdx,
            name=meta["name"],
            domain=meta["dims"],
            records=df_out,
            description=meta["description"],
        )

    gdx.write(path)
    logger.info("Written GDX: %s", path)


def write_csvs(store, output_dir):
    """Write every DataFrame in ``store`` as a CSV file into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    for key, df in store.items():
        df.to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "export_to_REMIND",
            configfiles="resources/TEST/i1/config.remind_scenario.yaml",
            iteration="1",
            scenario="TEST",
        )
        snakemake.input["networks"] = [
            f"../results/{snakemake.wildcards['scenario']}"
            f"/i{snakemake.wildcards['iteration']}"
            "/y2030/networks/base_s_4_elec_1H-Ep49.2.nc",
        ]

    configure_logging(snakemake)

    map_to_remind, sector_carriers, region_mapping = load_mappings(snakemake)
    z_cutoff = snakemake.config["remind_coupling"]["export_to_REMIND"]["z_cutoff"]

    networks = pd.DataFrame(snakemake.input["networks"], columns=["filepath"])
    networks["year"] = networks["filepath"].str.extract(r"/y(\d{4})/").astype(int)
    networks = networks.sort_values("year")

    coupling = {}   # exported to GDX + CSV
    reporting = {}  # exported to CSV only

    # Carrier strings for storage generation derived from the YAML mapping
    h2turb = next((c for c, r in map_to_remind.items() if r == "h2turb"), None)
    btout = next((c for c, r in map_to_remind.items() if r == "btout"), None)

    for year, row in networks.set_index("year").iterrows():
        fp = row["filepath"]
        logger.info("Processing year %d: %s", year, fp)

        n = pypsa.Network(fp)
        if not hasattr(n, "objective"):
            logger.warning("Year %d: no objective (solve failed?), skipping.", year)
            continue

        annotate_network(n, map_to_remind, region_mapping)

        # Electricity prices computed once; reused by all price-based metrics
        elec_prices = calculate_electricity_prices(n, sector_carriers, z_cutoff)
        avg_prices = (
            elec_prices.query("sector == 'total'")
            .set_index("region")["value"]
            .to_dict()
        )

        # ----------------------------------------------------------------
        # Coupling parameters (-> GDX + CSV)
        # ----------------------------------------------------------------
        _acc(coupling, "capacity_factors", year,
             calculate_capacity_factors(n))

        _acc(coupling, "optimal_capacities", year,
             calculate_optimal_capacities(n, ["Generator", "Link", "Store"],
                                          ["region", "remind_carrier"]))

        _acc(coupling, "potentials", year,
             calculate_potentials(n))

        _acc(coupling, "generation_shares", year,
             calculate_generation_shares(n))

        _acc(coupling, "markups_supply", year,
             calculate_markups_supply(n, avg_prices, z_cutoff))

        _acc(coupling, "market_values", year,
             calculate_market_values(n, z_cutoff))

        _acc(coupling, "average_electricity_price", year,
             elec_prices.query("sector == 'total'")[["region", "value"]])

        _acc(coupling, "sectoral_electricity_prices", year,
             elec_prices.query("sector != 'total' and value != 0")
             [["region", "sector", "value"]])

        _acc(coupling, "markups_demand", year,
             calculate_markups_demand(elec_prices).query("value != 0"))

        _acc(coupling, "peak_residual_load", year,
             calculate_peak_residual_loads(n, "relative"))

        if h2turb:
            _acc(coupling, "h2_turbine_generation", year,
                 calculate_link_generation(n, h2turb, "relative"))
        if btout:
            _acc(coupling, "battery_discharge", year,
                 calculate_link_generation(n, btout, "relative"))

        _acc(coupling, "grid_losses", year,
             calculate_grid_losses(n, "relative"))

        # ----------------------------------------------------------------
        # Reporting parameters (-> CSV only)
        # ----------------------------------------------------------------
        _acc(reporting, "energy_balance_ac", year,
             calculate_energy_balance(n, ["region", "carrier"], "AC"))

        _acc(reporting, "energy_balance_h2", year,
             calculate_energy_balance(n, ["region", "carrier"], "H2"))

        _acc(reporting, "optimal_capacities_full", year,
             calculate_optimal_capacities(
                 n, ["Generator", "Link", "Store", "StorageUnit", "Line"],
                 ["region", "carrier"],
             ))

        _acc(reporting, "peak_residual_load_absolute", year,
             calculate_peak_residual_loads(n, "absolute"))

        _acc(reporting, "electricity_prices", year,
             calculate_electricity_prices(n, sector_carriers, z_cutoff=False))

        _acc(reporting, "electricity_prices_cutoff", year, elec_prices)

        _acc(reporting, "market_values", year,
             calculate_market_values(n, z_cutoff=False))

        _acc(reporting, "market_values_cutoff", year,
             calculate_market_values(n, z_cutoff))

        _acc(reporting, "curtailments", year,
             calculate_curtailments(n))

        _acc(reporting, "hourly_electricity_prices", year,
             calculate_electricity_prices(
                 n, sector_carriers, z_cutoff=False, hourly=True
             ))

        if h2turb:
            _acc(reporting, "h2_turbine_generation_absolute", year,
                 calculate_link_generation(n, h2turb, "absolute"))
        if btout:
            _acc(reporting, "battery_discharge_absolute", year,
                 calculate_link_generation(n, btout, "absolute"))

        _acc(reporting, "grid_losses_absolute", year,
             calculate_grid_losses(n, "absolute"))

        _acc(reporting, "grid_flows", year,
             calculate_grid_flows(n))

        _acc(reporting, "grid_capacities", year,
             calculate_grid_capacities(n))

    # ----------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------
    logger.info("Writing GDX to %s", snakemake.output["gdx"])
    write_gdx(coupling, snakemake.output["gdx"], map_to_remind, sector_carriers)

    write_csvs(coupling, snakemake.output["coupling_parameters"])
    write_csvs(reporting, snakemake.output["reporting_parameters"])
