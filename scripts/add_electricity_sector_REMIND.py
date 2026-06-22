# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Orchestrator for REMIND-coupled electricity + sector build.

Design principle:
- Reuse shared upstream assembly functions from scripts.add_electricity.
- Add REMIND-specific components via additional functions.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from rpycpl.transforms.mapping import read_region_map as get_region_mapping

from scripts.add_electricity import (
    add_co2_emissions,
    add_missing_carriers,
    attach_conventional_generators,
    attach_existing_batteries,
    attach_load,
    attach_renewable_powerplants,
    attach_storageunits,
    attach_stores,
    attach_wind_and_solar,
    configure_logging,
    estimate_renewable_capacities,
    get_snapshots,
    load_and_aggregate_powerplants,
    normed,
    sanitize_carriers,
    sanitize_locations,
    set_scenario_config,
    set_transmission_costs,
    update_p_nom_max,
)
from scripts.build_transport_demand import transport_degree_factor

logger = logging.getLogger(__name__)

def overwrite_ppl_efficiency_with_costs(ppl: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """Replace each plant's efficiency and marginal cost with the REMIND cost table value for its carrier."""
    ppl = ppl.copy()
    ppl["efficiency"] = ppl["carrier"].map(costs["efficiency"])
    ppl["marginal_cost"] = (
        ppl["carrier"].map(costs["VOM"])
        + ppl["carrier"].map(costs["fuel"]) / ppl["efficiency"]
    )
    return ppl

def fold_sector_into_ac(demand: pd.DataFrame, source_sector: str) -> pd.DataFrame:
    """Re-label a sector as AC and aggregate duplicate rows."""
    demand = demand.copy()
    demand.loc[demand["sector"] == source_sector, "sector"] = "AC"
    return (
        demand.groupby(["year", "region", "sector", "unit"], as_index=False)["value"]
        .sum()
        .sort_values(["year", "region", "sector"])
    )


def prepare_sectoral_load(snakemake: Any, costs: pd.DataFrame, year: int) -> pd.DataFrame:
    """Load and aggregate REMIND sectoral demand according to sector-coupling config."""
    sectoral_load = pd.read_csv(snakemake.input.sectoral_load_country)
    sectoral_load = sectoral_load.query("year == @year").copy()

    sector_coupling = snakemake.params["sector_coupling"]

    if sector_coupling["electrolysis"]["enable"]:
        eta = costs.loc["electrolysis", "efficiency"]
        # Multiply electrolysis load by efficiency to convert from electricity demand to hydrogen demand
        h2_mask = sectoral_load["sector"] == "electrolysis"
        sectoral_load.loc[h2_mask, "value"] *= eta
        sectoral_load.loc[h2_mask, "unit"] = "MWh_H2"
    else:
        sectoral_load = fold_sector_into_ac(sectoral_load, "electrolysis")

    for sector in ("EV_pass", "EV_freight", "heatpump", "resistive"):
        if not sector_coupling[sector]["enable"]:
            sectoral_load = fold_sector_into_ac(sectoral_load, sector)

    return (
        sectoral_load.groupby(["year", "region", "sector", "unit"], as_index=False)
        .sum(numeric_only=True)
        .sort_values(["year", "region", "sector"])
    )


def calculate_load_scaling_factor(
    n: pypsa.Network,
    snakemake: Any,
    sectoral_load: pd.DataFrame,
) -> pd.Series:
    """Return a per-country Series (indexed by iso2) scaling AC load to REMIND demand."""
    remind_ac = (
        sectoral_load.query("sector == 'AC'")
        .set_index("region")["value"]
    )

    # Annual MWh per cluster bus: sum the hourly raw demand over time, then apply
    # the busmap (OSM bus names → cluster buses), then aggregate to country level.
    busmap = pd.read_csv(snakemake.input.busmap, index_col="name").squeeze()
    country_load = (
        xr.open_dataarray(snakemake.input.load)
        .sum("time")
        .to_pandas()
        .groupby(busmap)
        .sum()
        .groupby(n.buses["country"])
        .sum()
    )

    per_country = remind_ac / country_load

    return per_country



def _normalize_weights(s: pd.Series) -> pd.Series:
    """Normalize a non-negative Series to sum to 1; return uniform weights if the total is zero."""
    s = s.astype(float).clip(lower=0.0)
    total = s.sum()
    if total <= 0:
        return pd.Series(1.0 / len(s), index=s.index) if len(s) else s
    return s / total


def _to_scalar_region(value):
    """Return the first non-null member of a list-like value, or the value itself if scalar."""
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if pd.notna(item):
                return item
        return float("nan")
    return value


def _get_country_to_region(region_mapping_fn: str) -> pd.Series:
    """Return a Series mapping PyPSA-EUR country codes to scalar REMIND-EU region labels."""
    mapping = get_region_mapping(region_mapping_fn, source="PyPSA-EUR", target="REMIND-EU")
    return pd.Series(mapping).map(_to_scalar_region)


def _optional_input_path(value: Any) -> str | None:
    """Coerce a snakemake input value (possibly None or a list) to a string path, or None if absent."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    path = str(value)
    return path if path else None


def _fleet_evs_by_node_from_rds(
    n: pypsa.Network,
    fleet_file: str,
    region_mapping_fn: str,
    number_cars: pd.Series,
    year: int,
    kind: str,
) -> pd.Series | None:
    """Read BEV fleet sizes from an EDGE-T RDS file and distribute to nodes by REMIND region; None if unavailable."""
    try:
        import pyreadr
    except ImportError as exc:
        raise ModuleNotFoundError(
            "pyreadr is required to read fleetVehiclesPerTech.RDS but is not available."
        ) from exc

    subsector_map = {
        "pass": "trn_pass_road_LDV",
        "freight": "trn_freight_road_tmp_subsectorL2",
    }

    rds = pyreadr.read_r(fleet_file)
    fleet = next((v for v in rds.values() if isinstance(v, pd.DataFrame)), None)
    if fleet is None or fleet.empty:
        return None

    required = {"period", "region", "technology", "subsectorL2", "totVeh"}
    if not required.issubset(fleet.columns):
        logger.warning("Fleet RDS file is missing required columns %s; using fallback fleet estimate.", required)
        return None

    period = pd.to_numeric(fleet["period"], errors="coerce")
    tot_veh = pd.to_numeric(fleet["totVeh"], errors="coerce")
    rows = fleet[
        period.eq(int(year))
        & fleet["technology"].eq("BEV")
        & fleet["subsectorL2"].eq(subsector_map[kind])
        & tot_veh.notna()
    ].copy()
    if rows.empty:
        return None

    country_to_region = _get_country_to_region(region_mapping_fn)
    node_country = n.buses["country"].reindex(number_cars.index)
    node_region = node_country.map(country_to_region)

    number_evs = pd.Series(0.0, index=number_cars.index)
    found_any_region = False
    for _, row in rows.iterrows():
        region = _to_scalar_region(row["region"])
        if pd.isna(region):
            continue
        region_nodes = number_evs.index[node_region.eq(region)]
        if len(region_nodes) == 0:
            continue
        weights = _normalize_weights(number_cars.reindex(region_nodes).fillna(0.0))
        number_evs.loc[region_nodes] += weights * (float(row["totVeh"]) * 1e6)
        found_any_region = True

    if not found_any_region or number_evs.sum() <= 0:
        return None
    return number_evs


def attach_hydrogen_demand_remind(
    n: pypsa.Network,
    sectoral_load: pd.DataFrame,
) -> None:
    """Add a fixed H2 Load at each H2 bus, weighted by AC load within each country."""
    h2_demand = (
        sectoral_load.query("sector == 'electrolysis'")
        .groupby("region", as_index=True)["value"]
        .sum()
    )
    if h2_demand.empty:
        logger.info("No REMIND electrolysis demand found to attach.")
        return

    h2_buses = n.buses.query("carrier == 'H2'").copy()
    if h2_buses.empty:
        logger.warning("No H2 buses in network; skipping REMIND hydrogen demand attachment.")
        return

    h2_buses["country"] = h2_buses["location"].map(n.buses["country"])

    bus_load = n.loads_t.p_set.sum(axis=0).groupby(n.loads["bus"]).sum()

    snapshot_hours = float(n.snapshot_weightings.generators.sum())
    if snapshot_hours <= 0:
        snapshot_hours = 8760.0

    attached = 0
    for country, demand_mwh in h2_demand.items():
        country_h2 = h2_buses[h2_buses["country"] == country]
        if country_h2.empty:
            continue

        country_ac = country_h2["location"].dropna().astype(str)
        weights = _normalize_weights(bus_load.reindex(country_ac).fillna(0.0))
        if weights.empty:
            continue

        for bus, row in country_h2.iterrows():
            w = float(weights.get(str(row["location"]), 0.0))
            if w <= 0:
                continue
            n.add(
                "Load",
                f"{bus} demand REMIND",
                carrier="H2",
                bus=bus,
                p_set=float(demand_mwh) * w / snapshot_hours,
            )
            attached += 1

    logger.info("Attached REMIND hydrogen demand to %s H2 loads.", attached)


def cycling_shift(df: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
    """Circularly shift all rows of a DataFrame by `steps` positions along the time axis."""
    shifted = df.copy()
    shifted.values[:] = shifted.reindex(index=np.roll(shifted.index, steps)).values
    return shifted


def attach_ev_demand_remind(
    n: pypsa.Network,
    sectoral_load: pd.DataFrame,
    options_ev: dict,
    options_sector: dict,
    transport_demand_fn: str,
    transport_data_fn: str,
    avail_profile_fn: str,
    dsm_profile_fn: str,
    temp_air_total_fn: str,
    fleet_file: str | list,
    region_mapping_fn: str,
    year: int,
    kind: str,
) -> None:
    """
    Add EV demand, charging links, and optional DSM stores for passenger or freight vehicles.

    REMIND annual total is distributed to an hourly profile via transport demand scaled by
    heating degree days. Fleet size uses EDGE-T RDS if available, else config-based estimate.
    """
    if kind not in ("pass", "freight"):
        raise ValueError("kind must be 'pass' or 'freight'")

    sector = f"EV_{kind}"
    demand_mwh = sectoral_load.query("sector == @sector")["value"].sum()
    if demand_mwh <= 0:
        logger.info("No REMIND %s demand found; skipping.", sector)
        return

    spatial_nodes = n.buses.query("carrier == 'AC'").index
    transport = pd.read_csv(transport_demand_fn, index_col=0, parse_dates=True)
    transport = transport.reindex(index=n.snapshots)
    common_cols = [c for c in spatial_nodes if c in transport.columns]
    if not common_cols:
        logger.warning("No overlap between EV transport demand and AC buses; skipping %s.", sector)
        return
    transport = transport[common_cols].fillna(0.0)

    temperature = xr.open_dataarray(temp_air_total_fn).to_pandas()
    temperature = (
        temperature.reindex(index=n.snapshots, columns=common_cols)
        .ffill()
        .bfill()
        .fillna(15.0)
    )
    dd = transport_degree_factor(
        temperature,
        options_sector["transport_heating_deadband_lower"],
        options_sector["transport_heating_deadband_upper"],
        options_sector["EV_lower_degree_factor"],
        options_sector["EV_upper_degree_factor"],
    )
    transport = transport.mul(1 + dd, axis=0)

    demand_by_country = (
        sectoral_load.query("sector == @sector").set_index("region")["value"]
    )
    node_country = n.buses.loc[common_cols, "country"]
    load_p_set = pd.DataFrame(0.0, index=n.snapshots, columns=common_cols)
    for country, demand in demand_by_country.items():
        cnodes = node_country[node_country == country].index.intersection(transport.columns)
        if cnodes.empty:
            logger.warning("No nodes for country %s in EV %s profile; skipping.", country, sector)
            continue
        country_sum = float(transport[cnodes].to_numpy().sum())
        if country_sum <= 0:
            logger.warning("EV %s transport profile sum zero for %s; skipping.", sector, country)
            continue
        load_p_set[cnodes] = transport[cnodes] * (demand / country_sum)

    transport_data = pd.read_csv(transport_data_fn, index_col=0)
    number_cars = transport_data["number cars"].reindex(common_cols).fillna(0.0)
    car_weights = _normalize_weights(number_cars)

    fleet_file_path = _optional_input_path(fleet_file)
    number_evs = None
    if fleet_file_path:
        number_evs = _fleet_evs_by_node_from_rds(
            n,
            fleet_file_path,
            region_mapping_fn,
            number_cars,
            year,
            kind,
        )
        if number_evs is not None:
            number_evs = number_evs.reindex(common_cols).fillna(0.0)
            logger.info("Using EV fleet from %s for %s.", fleet_file_path, sector)

    if number_evs is None:
        number_evs_total = float(demand_mwh) / float(options_ev["annual_consumption"])
        number_evs = car_weights * number_evs_total
        logger.info("Using config-based EV fleet fallback for %s.", sector)

    charge_power = number_evs * float(options_ev["charge_rate"]) * float(options_ev["share_charger"])
    link_p_nom = charge_power * float(options_ev["dsm_availability"])
    store_e_nom = number_evs * float(options_ev["battery_size"]) * float(options_ev["dsm_availability"])

    avail = pd.read_csv(avail_profile_fn, index_col=0, parse_dates=True).reindex(index=n.snapshots)
    avail = avail.reindex(columns=common_cols).fillna(1.0)
    dsm = pd.read_csv(dsm_profile_fn, index_col=0, parse_dates=True).reindex(index=n.snapshots)
    dsm = dsm.reindex(columns=common_cols).fillna(0.0)

    carrier_name = f"EV {kind} battery"
    ev_nodes = pd.Index(common_cols) + f" {carrier_name}"

    n.add("Carrier", [carrier_name])
    n.add(
        "Bus",
        ev_nodes,
        location=common_cols,
        carrier=carrier_name,
        unit="MWh_el",
    )

    if options_ev["dsm"]:
        p_set = load_p_set
    else:
        p_set = (load_p_set + cycling_shift(load_p_set, 1) + cycling_shift(load_p_set, 2)) / 3

    n.add(
        "Load",
        common_cols,
        suffix=f" land transport EV {kind}",
        bus=ev_nodes,
        carrier=f"land transport EV {kind}",
        p_set=p_set,
    )

    n.add(
        "Link",
        common_cols,
        suffix=f" BEV {kind} charger",
        bus0=common_cols,
        bus1=ev_nodes,
        p_nom=link_p_nom.reindex(common_cols).fillna(0.0),
        carrier=f"BEV {kind} charger",
        p_max_pu=avail,
        efficiency=1.0,
    )

    if options_ev["dsm"]:
        n.add(
            "Store",
            common_cols,
            suffix=f" {carrier_name}",
            bus=ev_nodes,
            carrier=carrier_name,
            e_cyclic=True,
            e_nom=store_e_nom.reindex(common_cols).fillna(0.0),
            e_max_pu=1.0,
            e_min_pu=dsm,
        )

    logger.info("Attached REMIND %s demand (%.2f MWh).", sector, float(demand_mwh))


def attach_heat_demand_remind(
    n: pypsa.Network,
    sectoral_load: pd.DataFrame,
    options_heat: dict,
    wh_share_fn: str,
    hourly_heat_demand_fn: str,
    hourly_water_heat_demand_fn: str,
    year: int,
    kind: str,
    cop_profiles_fn: str | None = None,
) -> None:
    """
    Add electric heat demand for heat pumps or resistive heaters, scaled to the REMIND annual total.

    Space/water split uses wh_share; heatpump electricity = thermal / hourly COP (rural air-source).
    Optionally adds a fixed-size thermal storage buffer for DSM with E/P ratio from config.
    """
    if kind not in ("heatpump", "resistive"):
        raise ValueError("kind must be 'heatpump' or 'resistive'")

    demand_by_country = (
        sectoral_load.query("sector == @kind").set_index("region")["value"]
    )
    if demand_by_country.sum() <= 0:
        logger.info("No REMIND %s demand found; skipping.", kind)
        return

    spatial_nodes = n.buses.query("carrier == 'AC'").index
    wh_share = pd.read_csv(wh_share_fn)
    wh_year = min(int(year), 2100)
    wh = wh_share[wh_share["year"] == wh_year]
    item = "heat pumps" if kind == "heatpump" else "resistive heating"
    wh = wh[wh["item"] == item]
    water_share = float(wh["value"].iloc[0]) if not wh.empty else 0.2

    space_ds = xr.open_dataset(hourly_heat_demand_fn)
    space_heat = (space_ds["residential space"] + space_ds["services space"]).to_pandas()
    space_heat = space_heat.reindex(index=n.snapshots, columns=spatial_nodes).fillna(0.0)

    water_ds = xr.open_dataset(hourly_water_heat_demand_fn)
    water_heat = water_ds.to_dataframe()
    if "snapshots" in water_heat.index.names:
        water_heat = water_heat.reset_index().set_index("snapshots")
    water_heat.index = pd.to_datetime(water_heat.index)
    water_heat = water_heat.reindex(index=n.snapshots, columns=spatial_nodes).fillna(0.0)

    if kind == "heatpump":
        if cop_profiles_fn is None:
            raise ValueError("cop_profiles_fn is required for heatpump demand")
        cop = xr.open_dataarray(cop_profiles_fn)
        cop = (
            cop.sel(heat_system="rural", heat_source="air")
            .to_pandas()
            .reindex(index=n.snapshots, columns=spatial_nodes)
            .fillna(1.0)
        )
        space_heat = space_heat / cop
        water_heat = water_heat / cop

    node_country = n.buses.loc[spatial_nodes, "country"]
    total_elec_profile = pd.DataFrame(0.0, index=n.snapshots, columns=spatial_nodes)
    for country, demand in demand_by_country.items():
        cnodes = node_country[node_country == country].index.intersection(space_heat.columns)
        if cnodes.empty:
            logger.warning("No nodes for country %s in %s profile; skipping.", country, kind)
            continue
        space_sum = float(space_heat[cnodes].to_numpy().sum())
        water_sum = float(water_heat[cnodes].to_numpy().sum())
        if space_sum > 0:
            total_elec_profile[cnodes] += space_heat[cnodes] * (demand * (1.0 - water_share) / space_sum)
        if water_sum > 0:
            total_elec_profile[cnodes] += water_heat[cnodes] * (demand * water_share / water_sum)

    carrier = f"{kind} electricity"
    heat_nodes = spatial_nodes + f" {carrier}"

    n.add("Carrier", [carrier])
    n.add(
        "Bus",
        heat_nodes,
        location=spatial_nodes,
        carrier=carrier,
        unit="MWh_el",
    )

    n.add(
        "Load",
        spatial_nodes,
        suffix=f" {carrier}",
        bus=heat_nodes,
        carrier=carrier,
        p_set=total_elec_profile,
    )

    p_nom_max = total_elec_profile.max(axis=0)
    n.add(
        "Link",
        spatial_nodes,
        suffix=f" {kind}",
        carrier=kind,
        bus0=spatial_nodes,
        bus1=heat_nodes,
        p_nom=p_nom_max,
        p_nom_extendable=False,
        efficiency=1.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
    )

    if options_heat["dsm"]:
        ep_ratio = float(options_heat["ep_ratio"])

        if kind == "heatpump":
            total_heat_profile = total_elec_profile * cop
            p_therm_max = total_heat_profile.max(axis=0).fillna(0.0)
            e_therm = p_therm_max * ep_ratio

            cop_safe = cop.clip(lower=1e-6)
            elec_storage = e_therm / cop_safe
            size_store = elec_storage.max(axis=0).fillna(0.0)
            max_pu_store = (elec_storage / size_store.replace(0.0, np.nan)).fillna(0.0)
        else:
            p_therm_max = total_elec_profile.max(axis=0).fillna(0.0)
            size_store = p_therm_max * ep_ratio
            max_pu_store = 1.0

        n.add(
            "Store",
            spatial_nodes,
            suffix=f" {kind} storage",
            bus=heat_nodes,
            carrier=f"{kind} storage",
            e_cyclic=True,
            e_nom=size_store,
            e_nom_extendable=False,
            e_max_pu=max_pu_store,
            e_min_pu=0.0,
        )

        logger.info(
            "Attached %s DSM store with fixed E/P ratio %.2f h (total e_nom=%.2f MWh_el).",
            kind,
            ep_ratio,
            float(pd.Series(size_store).sum()),
        )

    logger.info("Attached REMIND %s demand (%.2f MWh).", kind, float(demand_by_country.sum()))


def attach_hydro_remind(
    n: pypsa.Network,
    costs: pd.DataFrame,
    ppl: pd.DataFrame,
    profile_hydro: str,
    hydro_capacities: str,
    carriers: list,
    hydro_targets_fn: str,
    region_mapping_fn: str,
    year: int,
    **params,
) -> None:
    """
    Attach hydro following the upstream attach_hydro flow, with two REMIND-specific injections:
    scale ror/hydro capacities by REMIND region and scale inflow energy by REMIND region.
    """
    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)

    ror = ppl.query('carrier == "ror"').copy()
    phs = ppl.query('carrier == "PHS"').copy()
    hydro = ppl.query('carrier == "hydro"').copy()

    if ror.empty and hydro.empty:
        if "PHS" in carriers and not phs.empty:
            max_hours = params.get("PHS_max_hours", 6)
            phs = phs.replace({"max_hours": {0: max_hours, np.nan: max_hours}})
            n.add(
                "StorageUnit",
                phs.index,
                carrier="PHS",
                bus=phs["bus"],
                p_nom=phs["p_nom"],
                capital_cost=costs.at["PHS", "capital_cost"],
                max_hours=phs["max_hours"],
                efficiency_store=np.sqrt(costs.at["PHS", "efficiency"]),
                efficiency_dispatch=np.sqrt(costs.at["PHS", "efficiency"]),
                cyclic_state_of_charge=True,
            )
        return

    # Build country/region labels for hydro assets.
    hydro_assets = pd.concat([ror, hydro], axis=0)
    hydro_assets["country"] = hydro_assets["bus"].map(n.buses.country)

    region_mapping = get_region_mapping(region_mapping_fn, source="PyPSA-EUR", target="REMIND-EU")
    country_to_region = pd.Series(region_mapping).map(_to_scalar_region)
    hydro_assets["region"] = hydro_assets["country"].map(country_to_region)

    hydro_targets = pd.read_csv(hydro_targets_fn)
    hydro_targets["year"] = hydro_targets["year"].astype(str)
    hydro_targets_year = hydro_targets[hydro_targets["year"] == str(year)].copy()

    if hydro_targets_year.empty:
        logger.warning(
            "No REMIND hydro targets found for year %s in %s. Using upstream hydro behaviour.",
            year,
            hydro_targets_fn,
        )
        hydro_targets_year = pd.DataFrame(
            columns=["region", "hydro_capacity_mw", "hydro_generation_mwh"]
        )

    target_capacity = (
        hydro_targets_year.groupby("region", observed=False)["hydro_capacity_mw"]
        .sum()
        .astype(float)
    )
    target_generation = (
        hydro_targets_year.groupby("region", observed=False)["hydro_generation_mwh"]
        .sum()
        .astype(float)
    )

    # 1) Capacity scaling by REMIND region.
    current_capacity = hydro_assets.groupby("region", observed=False)["p_nom"].sum()
    capacity_factor = (target_capacity / current_capacity).replace([np.inf, -np.inf], np.nan)
    capacity_factor = capacity_factor.fillna(1.0)
    hydro_assets["cap_factor"] = hydro_assets["region"].map(capacity_factor).fillna(1.0)
    hydro_assets["p_nom"] = hydro_assets["p_nom"] * hydro_assets["cap_factor"]

    # Push adjusted p_nom back to ror/hydro tables.
    ror.loc[hydro_assets.loc[ror.index].index, "p_nom"] = hydro_assets.loc[ror.index, "p_nom"]
    hydro.loc[hydro_assets.loc[hydro.index].index, "p_nom"] = hydro_assets.loc[hydro.index, "p_nom"]

    inflow_idx = ror.index.union(hydro.index)
    country = hydro_assets["country"]

    dist_key = hydro_assets.loc[inflow_idx, "p_nom"].groupby(country.loc[inflow_idx]).transform(normed)

    with xr.open_dataarray(profile_hydro) as inflow:
        inflow_countries = pd.Index(country.loc[inflow_idx])
        missing_c = inflow_countries.unique().difference(inflow.indexes["countries"])
        assert missing_c.empty, (
            f"'{profile_hydro}' is missing "
            f"inflow time-series for at least one country: {', '.join(missing_c)}"
        )

        inflow_t = (
            inflow.sel(countries=inflow_countries)
            .rename({"countries": "name"})
            .assign_coords(name=inflow_idx)
            .transpose("time", "name")
            .to_pandas()
            .multiply(dist_key, axis=1)
        )

    # 2) Inflow energy scaling by REMIND region.
    asset_region = hydro_assets.loc[inflow_idx, "region"]
    current_generation = inflow_t.T.groupby(asset_region).sum().sum(axis=1)
    generation_factor = (target_generation / current_generation).replace([np.inf, -np.inf], np.nan)
    generation_factor = generation_factor.fillna(1.0)
    inflow_factor = asset_region.map(generation_factor).fillna(1.0)
    inflow_t = inflow_t.multiply(inflow_factor, axis=1)

    if "ror" in carriers and not ror.empty:
        n.add(
            "Generator",
            ror.index,
            carrier="ror",
            bus=ror["bus"],
            p_nom=ror["p_nom"],
            efficiency=costs.at["ror", "efficiency"],
            capital_cost=costs.at["ror", "capital_cost"],
            weight=ror["p_nom"],
            p_max_pu=(
                inflow_t[ror.index]
                .divide(ror["p_nom"], axis=1)
                .where(lambda df: df <= 1.0, other=1.0)
            ),
        )

    if "PHS" in carriers and not phs.empty:
        max_hours = params.get("PHS_max_hours", 6)
        phs = phs.replace({"max_hours": {0: max_hours, np.nan: max_hours}})
        n.add(
            "StorageUnit",
            phs.index,
            carrier="PHS",
            bus=phs["bus"],
            p_nom=phs["p_nom"],
            capital_cost=costs.at["PHS", "capital_cost"],
            max_hours=phs["max_hours"],
            efficiency_store=np.sqrt(costs.at["PHS", "efficiency"]),
            efficiency_dispatch=np.sqrt(costs.at["PHS", "efficiency"]),
            cyclic_state_of_charge=True,
        )

    if "hydro" in carriers and not hydro.empty:
        hydro_max_hours = params.get("hydro_max_hours")

        assert hydro_capacities is not None, "No path for hydro capacities given."

        hydro_stats = pd.read_csv(
            hydro_capacities, comment="#", na_values="-", index_col=0
        )
        e_target = hydro_stats["E_store[TWh]"].clip(lower=0.2) * 1e6
        e_installed = hydro.eval("p_nom * max_hours").groupby(hydro.country).sum()
        e_missing = e_target - e_installed
        missing_mh_i = hydro.query("max_hours.isnull() or max_hours == 0").index
        missing_mh_single_i = hydro.index[
            ~hydro.country.duplicated() & hydro.country.isin(e_missing.dropna().index)
        ]
        missing_mh_i = missing_mh_i.union(missing_mh_single_i)

        if hydro_max_hours == "energy_capacity_totals_by_country":
            max_hours_country = (
                e_missing / hydro.loc[missing_mh_i].groupby("country").p_nom.sum()
            )

        elif hydro_max_hours == "estimate_by_large_installations":
            max_hours_country = (
                hydro_stats["E_store[TWh]"] * 1e3 / hydro_stats["p_nom_discharge[GW]"]
            )
        else:
            raise ValueError(f"Unknown hydro_max_hours method: {hydro_max_hours}")

        max_hours_country.clip(0, inplace=True)

        missing_countries = pd.Index(hydro["country"].unique()).difference(
            max_hours_country.dropna().index
        )
        if not missing_countries.empty:
            logger.warning(
                f"Assuming max_hours=6 for hydro reservoirs in the countries: {', '.join(missing_countries)}"
            )
        hydro_max_hours = hydro.max_hours.where(
            (hydro.max_hours > 0) & ~hydro.index.isin(missing_mh_single_i),
            hydro.country.map(max_hours_country),
        ).fillna(6)

        if params.get("flatten_dispatch", False):
            buffer = params.get("flatten_dispatch_buffer", 0.2)
            average_capacity_factor = inflow_t[hydro.index].mean() / hydro["p_nom"]
            p_max_pu = (average_capacity_factor + buffer).clip(upper=1)
        else:
            p_max_pu = 1

        n.add(
            "StorageUnit",
            hydro.index,
            carrier="hydro",
            bus=hydro["bus"],
            p_nom=hydro["p_nom"],
            max_hours=hydro_max_hours,
            capital_cost=costs.at["hydro", "capital_cost"],
            marginal_cost=costs.at["hydro", "marginal_cost"],
            p_max_pu=p_max_pu,
            p_min_pu=0.0,
            efficiency_dispatch=costs.at["hydro", "efficiency"],
            efficiency_store=0.0,
            cyclic_state_of_charge=True,
            inflow=inflow_t.loc[:, hydro.index],
        )


def attach_hydrogen_storage_remind(
    n: pypsa.Network,
    snakemake: Any,
    costs: pd.DataFrame,
) -> None:
    """Attach REMIND hydrogen buses, storage, and conversion links."""
    storage_settings = snakemake.params["h2_settings"]
    electrolysis_settings = snakemake.params["sector_coupling"]["electrolysis"]

    ac_buses = n.buses.query("carrier == 'AC'").index
    if ac_buses.empty:
        logger.info("No AC buses available for REMIND hydrogen infrastructure.")
        return

    n.add("Carrier", "H2")

    h2_buses = pd.Index([f"{bus} H2" for bus in ac_buses])
    n.add(
        "Bus",
        h2_buses,
        location=ac_buses,
        carrier="H2",
        unit="MWh_H2",
        x=n.buses.loc[ac_buses, "x"].values,
        y=n.buses.loc[ac_buses, "y"].values,
    )

    h2_caverns = pd.read_csv(snakemake.input.h2_cavern, index_col=0)
    enabled_types = [
        cavern_type
        for cavern_type in storage_settings["hydrogen_underground_storage_locations"]
        if cavern_type in h2_caverns.columns
    ]
    if enabled_types:
        h2_caverns = h2_caverns[enabled_types].sum(axis=1)
        h2_caverns = h2_caverns[h2_caverns > 2]
        h2_caverns = h2_caverns * 1e6
        h2_caverns.clip(upper=1e9, inplace=True)
    else:
        h2_caverns = pd.Series(dtype=float)

    h2_cavern_buses = pd.Index([])
    if not h2_caverns.empty:
        underground_capital_cost = costs.at["hydrogen storage underground", "capital_cost"]
        underground_lifetime = costs.at["hydrogen storage underground", "lifetime"]
        h2_cavern_buses = h2_caverns.index.intersection(ac_buses)

        n.add(
            "Store",
            h2_cavern_buses.map(lambda bus: f"{bus} H2 Store"),
            bus=h2_cavern_buses.map(lambda bus: f"{bus} H2"),
            carrier="H2 Store",
            e_nom_extendable=True,
            e_nom_max=h2_caverns.reindex(h2_cavern_buses).values,
            e_cyclic=True,
            capital_cost=underground_capital_cost,
            lifetime=underground_lifetime,
        )

    tank_tech = "hydrogen storage tank type 1 including compressor"
    tank_buses = ac_buses.difference(h2_caverns.index)
    if not tank_buses.empty:
        n.add(
            "Store",
            tank_buses.map(lambda bus: f"{bus} H2 Store"),
            bus=tank_buses.map(lambda bus: f"{bus} H2"),
            carrier="H2 Store",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at[tank_tech, "capital_cost"],
            lifetime=costs.at[tank_tech, "lifetime"],
        )

    if electrolysis_settings.get("enable", True):
        electrolysis_names = ac_buses.map(lambda bus: f"{bus} H2 Electrolysis")
        ramp_limit = electrolysis_settings.get("ramp_limit", np.nan)
        n.add(
            "Link",
            electrolysis_names,
            bus0=ac_buses,
            bus1=h2_buses,
            carrier="H2 Electrolysis",
            p_nom_extendable=True,
            p_min_pu=electrolysis_settings.get("p_min_pu", 0.0),
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            marginal_cost=costs.at["electrolysis", "marginal_cost"],
            lifetime=costs.at["electrolysis", "lifetime"],
            ramp_limit_up=None if pd.isna(ramp_limit) else ramp_limit,
            ramp_limit_down=None if pd.isna(ramp_limit) else ramp_limit,
        )

    fuel_cell_names = ac_buses.map(lambda bus: f"{bus} H2 Fuel Cell")
    n.add(
        "Link",
        fuel_cell_names,
        bus0=h2_buses,
        bus1=ac_buses,
        carrier="H2 Fuel Cell",
        p_nom_extendable=True,
        efficiency=costs.at["fuel cell", "efficiency"],
        capital_cost=costs.at["fuel cell", "capital_cost"] * costs.at["fuel cell", "efficiency"],
        marginal_cost=costs.at["fuel cell", "marginal_cost"],
        lifetime=costs.at["fuel cell", "lifetime"],
    )

    logger.info(
        "Attached REMIND hydrogen infrastructure to %s AC buses (%s cavern sites, %s tank sites).",
        len(ac_buses),
        len(h2_cavern_buses),
        len(tank_buses),
    )


def apply_regional_costs(
    n: pypsa.Network,
    costs_regional: pd.DataFrame,
    co2_price_by_region: dict,
    country_to_region: pd.Series,
) -> None:
    """
    Override capital_cost and marginal_cost per component using REMIND regional values.

    Also adds the regional CO₂ cost to generator/storage marginal costs, replacing the
    global add_emission_prices() call that prepare_network.py would otherwise make.
    """
    def _apply_costs(df: pd.DataFrame, bus_col: str) -> None:
        if df.empty:
            return
        countries = n.buses.loc[df[bus_col], "country"].values
        regions = pd.Series(countries, index=df.index).map(country_to_region)
        # Build (region, carrier) lookup keys for each component row
        keys = pd.MultiIndex.from_arrays([regions, df["carrier"]])
        valid_mask = keys.isin(costs_regional.index)
        if not valid_mask.any():
            return
        valid_idx = df.index[valid_mask]
        valid_keys = keys[valid_mask]
        for col in ["capital_cost", "marginal_cost"]:
            if col not in costs_regional.columns:
                continue
            new_vals = costs_regional.loc[valid_keys, col].values
            not_nan = ~pd.isna(new_vals)
            df.loc[valid_idx[not_nan], col] = new_vals[not_nan]

    _apply_costs(n.generators, "bus")
    _apply_costs(n.links, "bus0")
    _apply_costs(n.storage_units, "bus")
    _apply_costs(n.stores, "bus")

    # Add regional CO₂ cost contribution to generator marginal costs.
    # This replaces the global add_emission_prices() that prepare_network.py would
    # otherwise call; emission_prices.enable is set to False in _remind_emission_prices.
    if not n.generators.empty and co2_price_by_region:
        countries = n.buses.loc[n.generators["bus"], "country"].values
        regions = pd.Series(countries, index=n.generators.index).map(country_to_region)
        co2_price_series = regions.map(co2_price_by_region).fillna(0.0)
        co2_intensity = n.generators["carrier"].map(n.carriers["co2_emissions"]).fillna(0.0)
        efficiency = n.generators["efficiency"].replace(0.0, 1.0)
        n.generators["marginal_cost"] += co2_intensity * co2_price_series / efficiency

    if not n.storage_units.empty and co2_price_by_region:
        countries = n.buses.loc[n.storage_units["bus"], "country"].values
        regions = pd.Series(countries, index=n.storage_units.index).map(country_to_region)
        co2_price_series = regions.map(co2_price_by_region).fillna(0.0)
        co2_intensity = n.storage_units["carrier"].map(n.carriers["co2_emissions"]).fillna(0.0)
        efficiency = n.storage_units["efficiency_dispatch"].replace(0.0, 1.0)
        n.storage_units["marginal_cost"] += co2_intensity * co2_price_series / efficiency

    logger.info("Applied regional REMIND costs to all network components.")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_electricity_sector_REMIND",
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            year_REMIND="2030",
            clusters=4,
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)

    params = snakemake.params
    max_hours = params.electricity["max_hours"]
    year = int(snakemake.wildcards.year_REMIND)
    landfall_lengths = {
        tech: settings["landfall_length"]
        for tech, settings in params.renewable.items()
        if "landfall_length" in settings.keys()
    }

    n = pypsa.Network(snakemake.input.base_network)

    time = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)
    n.set_snapshots(time)

    # Load costs with (region, technology) MultiIndex; derive region-averaged fallback
    # for upstream attach_* functions that don't know about regional variation.
    costs_regional = pd.read_csv(snakemake.input.costs, index_col=[0, 1])
    costs = costs_regional.groupby(level="technology").mean()

    # Per-region CO₂ prices (applied in apply_regional_costs below)
    co2_price_df = pd.read_csv(snakemake.input.co2_price)
    co2_price_by_region = (
        co2_price_df.query("year == @year")
        .set_index("region")["co2_price"]
        .to_dict()
    )

    country_to_region = _get_country_to_region(snakemake.input.region_mapping)

    ppl = load_and_aggregate_powerplants(
        snakemake.input.powerplants,
        costs,
        params.consider_efficiency_classes,
        params.aggregation_strategies,
        params.exclude_carriers,
    )

    # load_and_aggregate_powerplants's carrier_dict maps "ccgt"→"CCGT" and "ocgt"→"OCGT"
    # (uppercase). Fueltype values for all other REMIND carriers are pre-set in
    # adjust_powerplants_REMIND.py and pass through to_pypsa_names() as-is (lowercase).
    ppl["carrier"] = ppl["carrier"].replace({"CCGT": "ccgt", "OCGT": "ocgt"})

    # Overwrite plant-specific efficiencies with REMIND efficiencies
    ppl = overwrite_ppl_efficiency_with_costs(ppl, costs)

    # REMIND specific
    sectoral_load = prepare_sectoral_load(snakemake, costs, year=year)
    load_scaling_per_country = calculate_load_scaling_factor(n, snakemake, sectoral_load)
    load_scaling_per_bus = n.buses["country"].map(load_scaling_per_country)

    # Attach AC load, other loads are attached below
    attach_load(
        n,
        snakemake.input.load,
        snakemake.input.busmap,
        load_scaling_per_bus,
    )

    set_transmission_costs(
        n,
        costs,
        params.line_length_factor,
        params.link_length_factor,
    )

    renewable_carriers = set(params.electricity["renewable_carriers"])
    extendable_carriers = params.electricity["extendable_carriers"]
    conventional_carriers = params.electricity["conventional_carriers"]
    conventional_inputs = {
        k: v for k, v in snakemake.input.items() if k.startswith("conventional_")
    }

    if params.conventional["unit_commitment"]:
        unit_commitment = pd.read_csv(snakemake.input.unit_commitment, index_col=0)
    else:
        unit_commitment = None

    if params.conventional["dynamic_fuel_price"]:
        fuel_price = pd.read_csv(
            snakemake.input.fuel_price, index_col=0, parse_dates=True
        )
        fuel_price = fuel_price.reindex(n.snapshots).ffill()
    else:
        fuel_price = None

    # add_electricity.py's add_co2_emissions() splits carrier names on "-" to get the
    # suptech prefix (e.g. "coal-PC" → "coal") and looks that up in the costs index.
    # REMIND cost entries use full carrier names, so we add alias rows for each
    # suptech prefix that is missing from the index.
    costs_for_attach = costs.copy()
    for carrier in list(costs.index):
        suptech = carrier.split("-")[0]
        if suptech not in costs_for_attach.index:
            costs_for_attach.loc[suptech] = costs_for_attach.loc[carrier]

    attach_conventional_generators(
        n,
        costs_for_attach,
        ppl,
        conventional_carriers,
        extendable_carriers,
        renewable_carriers,
        conventional_params=params.conventional,
        conventional_inputs=conventional_inputs,
        unit_commitment=unit_commitment,
        fuel_price=fuel_price,
    )

    attach_wind_and_solar(
        n,
        costs,
        snakemake.input,
        renewable_carriers,
        extendable_carriers,
        params.line_length_factor,
        landfall_lengths,
    )

    if "hydro" in renewable_carriers:
        p = params.renewable["hydro"].copy()
        carriers = p.pop("carriers", [])
        attach_hydro_remind(
            n,
            costs,
            ppl,
            snakemake.input.profile_hydro,
            snakemake.input.hydro_capacities,
            carriers,
            snakemake.input.hydro_targets,
            snakemake.input.region_mapping,
            year,
            **p,
        )

    estimate_renewable_caps = params.electricity["estimate_renewable_capacities"]
    if estimate_renewable_caps["enable"]:
        if params.foresight != "overnight":
            logger.info(
                "Skipping renewable capacity estimation because they are added later "
                "in rule `add_existing_baseyear` with foresight mode 'myopic'."
            )
        else:
            tech_map = estimate_renewable_caps["technology_mapping"]
            expansion_limit = estimate_renewable_caps["expansion_limit"]
            year = estimate_renewable_caps["year"]

            if estimate_renewable_caps["from_powerplantmatching"]:
                attach_renewable_powerplants(n, tech_map, snakemake.input)

            if estimate_renewable_caps["from_irenastat"]:
                estimate_renewable_capacities(
                    n, year, tech_map, expansion_limit, params.countries
                )

    update_p_nom_max(n)

    attach_storageunits(
        n, costs, n.buses.index, extendable_carriers["StorageUnit"], max_hours
    )

    # Exclude H2 from generic store attachment to use REMIND-specific hydrogen setup
    store_carriers = [c for c in extendable_carriers["Store"] if c != "H2"]
    attach_stores(n, costs, n.buses.index, store_carriers)

    # Apply REMIND-specific hydrogen storage and links BEFORE sector coupling
    # (sector coupling needs H2 buses to attach hydrogen demand)
    if "H2" in extendable_carriers["Store"]:
        attach_hydrogen_storage_remind(n, snakemake, costs)

    # Apply battery e_min_pu constraint from REMIND configuration
    battery_settings = snakemake.params.get("battery_settings", {})
    if "e_min_pu" in battery_settings and "battery" in extendable_carriers["Store"]:
        e_min_pu = battery_settings["e_min_pu"]
        battery_mask = n.stores["carrier"] == "battery"
        n.stores.loc[battery_mask, "e_min_pu"] = e_min_pu
        logger.info(f"Applied REMIND battery e_min_pu constraint: {e_min_pu}")

    if params.electricity.get("estimate_battery_capacities", False):
        attach_existing_batteries(n, costs, ppl)

    # Add sectors if configured
    sector_coupling = snakemake.params["sector_coupling"]

    if sector_coupling["electrolysis"]["enable"]:
        attach_hydrogen_demand_remind(
            n,
            sectoral_load,
        )

    if sector_coupling["EV_pass"]["enable"]:
        attach_ev_demand_remind(
            n,
            sectoral_load,
            sector_coupling["EV_pass"],
            snakemake.params["sector"],
            snakemake.input.transport_demand,
            snakemake.input.transport_data,
            snakemake.input.avail_profile,
            snakemake.input.dsm_profile,
            snakemake.input.temp_air_total,
            snakemake.input.fleet_file,
            snakemake.input.region_mapping,
            year,
            kind="pass",
        )

    if sector_coupling["EV_freight"]["enable"]:
        attach_ev_demand_remind(
            n,
            sectoral_load,
            sector_coupling["EV_freight"],
            snakemake.params["sector"],
            snakemake.input.transport_demand,
            snakemake.input.transport_data,
            snakemake.input.avail_profile,
            snakemake.input.dsm_profile,
            snakemake.input.temp_air_total,
            snakemake.input.fleet_file,
            snakemake.input.region_mapping,
            year,
            kind="freight",
        )

    if sector_coupling["heatpump"]["enable"]:
        attach_heat_demand_remind(
            n,
            sectoral_load,
            sector_coupling["heatpump"],
            snakemake.input.wh_share,
            snakemake.input.hourly_heat_demand_total,
            snakemake.input.hourly_water_heat_demand_total,
            year,
            kind="heatpump",
            cop_profiles_fn=snakemake.input.cop_profiles,
        )

    if sector_coupling["resistive"]["enable"]:
        attach_heat_demand_remind(
            n,
            sectoral_load,
            sector_coupling["resistive"],
            snakemake.input.wh_share,
            snakemake.input.hourly_heat_demand_total,
            snakemake.input.hourly_water_heat_demand_total,
            year,
            kind="resistive",
        )

    sanitize_carriers(n, snakemake.config)
    if "location" in n.buses:
        sanitize_locations(n)

    # Lowercase the carrier/class suffix of generator names (keep bus prefix as-is)
    # and sort alphabetically. Example: "DE0 0 CCGT" → "DE0 0 ccgt".
    buses = set(n.buses.index)
    rename_map = {}
    for name in n.generators.index:
        for bus in buses:
            if name.startswith(bus + " "):
                rename_map[name] = bus + " " + name[len(bus) + 1:].lower()
                break
        else:
            rename_map[name] = name
    n.generators = n.generators.rename(index=rename_map).sort_index()
    for attr, df in n.generators_t.items():
        n.generators_t[attr] = df.rename(columns=rename_map).reindex(
            sorted(df.rename(columns=rename_map).columns), axis=1
        )

    # add_co2_emissions() in add_electricity.py uses only the suptech prefix (first
    # part before "-") to look up CO2 intensity, so all biomass carriers would
    # inherit biomass-chp's value of 0.  Override with the full-name lookup so
    # that biomass-igcc-ccs (negative CO2 intensity from BECCS) is set correctly.
    for carrier in costs.index:
        if carrier in n.carriers.index and not pd.isna(costs.at[carrier, "CO2 intensity"]):
            n.carriers.at[carrier, "co2_emissions"] = costs.at[carrier, "CO2 intensity"]

    # Override component costs with per-region REMIND values and add regional CO₂ costs.
    # Must run after carrier CO₂ intensities are set above (needed for CO₂ cost calc).
    apply_regional_costs(n, costs_regional, co2_price_by_region, country_to_region)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
