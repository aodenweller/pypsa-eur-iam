"""REMIND → PyPSA-Eur coupling adapter.

Subclasses ``rpycpl.CouplingAdapter``. Inherits the generic Stage-1 builders (co2 prices,
country loads, capacity targets) and supplies the EUR-specific pieces:

- ``extract_cost_parameters`` — the per-parameter REMIND cost reads + unit factors (ported
  from the old ``import_REMIND_costs.extract_remind_parameter_data``), driven by the central
  symbol config so no GDX names are hardcoded here.
- ``prepare_capacities`` — REMIND-tech-specific prep (VRE-variant merge, battery scaling),
  ported from ``import_REMIND_capacities``.
- ``build_config_overrides`` — the EUR config-key structure.

Depends only on ``rpycpl`` + pandas (no PyPSA-Eur ``_helpers``), so it is unit-testable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from rpycpl.adapters.base import CouplingAdapter
from rpycpl.symbols import load_frame

# Link-like techs whose REMIND output-capacity is converted to input-capacity (÷ efficiency).
LINK_TECHS = {"elh2", "h2turb", "btin", "elh2VRE", "h2turbVRE"}
_VRE_TO_PRIMARY = {"elh2VRE": "elh2", "h2turbVRE": "h2turb"}
_BATTERY_SCALING = {"storspv": 4.0, "storwindon": 1.2, "storwindoff": 1.2}


class RemindEurAdapter(CouplingAdapter):
    """Expose REMIND-derived inputs to the PyPSA-Eur workflow."""

    def build_config_overrides(self) -> dict[str, Any]:
        """Build PyPSA-Eur config overrides (planning horizons + CO2 price pathway)."""
        # Bind each piece to a named variable (no inline calls in the dict) so the
        # config-override inputs can be inspected when debugging.
        planning_horizons = list(self.config.get("planning_horizons", []))
        co2_prices = self.build_co2_prices().to_dict(orient="records")
        return {
            "scenario": {"planning_horizons": planning_horizons},
            "co2_prices": co2_prices,
        }

    def extract_cost_parameters(self, year: int) -> pd.DataFrame:
        """Extract REMIND cost params as long ``[region, reference, parameter, value, unit]``."""
        y = str(year)
        load = lambda name: load_frame(self.loader, self.symbols[name])  # noqa: E731

        costs = load("cost_investment").query("year == @y").copy()
        costs["value"] *= 1e6
        costs["parameter"] = "investment"
        costs["unit"] = "USD/MW"
        costs.loc[costs["technology"].isin(["h2stor", "btstor"]), "unit"] = "USD/MWh"

        pm = load("tech_data")
        lifetime = pm.query("char == 'lifetime'").assign(parameter="lifetime", unit="years")
        fom = pm.query("char == 'omf'").copy()
        fom["value"] *= 100
        fom = fom.assign(parameter="FOM", unit="%/year")
        vom = pm.query("char == 'omv'").copy()
        vom["value"] *= 1e6 / 8760
        vom = vom.assign(parameter="VOM", unit="USD/MWh")

        co2i = load("emission_factor").query(
            "to_carrier == 'seel' & emission_type == 'co2' & year == @y"
        ).copy()
        co2i["value"] *= 1e9 * ((2 * 16 + 12) / 12) / 8760 / 1e6
        co2i = co2i.assign(parameter="CO2 intensity", unit="t_CO2/MWh_th")

        eta = load("efficiency_conv").query("year == @y")
        dataeta = load("efficiency_data").query("year == @y")
        keys = set(zip(eta["region"], eta["technology"]))
        fallback = dataeta[
            ~pd.MultiIndex.from_arrays([dataeta["region"], dataeta["technology"]]).isin(keys)
        ]
        eff = pd.concat([eta, fallback]).assign(parameter="efficiency", unit="p.u.")
        eff.loc[eff["technology"].isin(["fnrs", "tnrs"]), "value"] *= 8760 / 1e6
        eff.loc[eff["technology"].isin(["fnrs", "tnrs"]), "unit"] = "MWh/g_U"
        eff.loc[eff["technology"] == "btin", "value"] **= 2

        fuel = load("fuel_price").query("year == @y").copy()
        fuel["parameter"] = "fuel"
        fuel.loc[fuel["technology"] != "peur", "value"] *= 1e6 / 8760
        fuel["unit"] = "USD/MWh_th"
        fuel.loc[fuel["technology"] == "peur", "unit"] = "USD/g_U"

        df = pd.concat([costs, lifetime, fom, vom, co2i, eff, fuel])[
            ["region", "technology", "parameter", "value", "unit"]
        ].rename(columns={"technology": "reference"})
        return df[df["region"].isin(self.remind_regions)]

    def prepare_capacities(self, caps: pd.DataFrame) -> pd.DataFrame:
        """Merge VRE-coupled variants and scale battery techs before carrier mapping."""
        caps = caps.copy()
        tech = caps["technology"].astype(str)
        caps["technology"] = tech.map(lambda t: _VRE_TO_PRIMARY.get(t, t))

        tech = caps["technology"].astype(str)
        btin_present = ((tech == "btin") & (caps["value"] > 0)).any()
        is_stor = tech.isin(_BATTERY_SCALING)
        if btin_present:
            return caps[~is_stor].copy()
        scale = tech.map(_BATTERY_SCALING)
        caps.loc[scale.notna(), "value"] *= scale[scale.notna()]
        caps["technology"] = tech.map(lambda t: "btin" if t in _BATTERY_SCALING else t)
        return caps
