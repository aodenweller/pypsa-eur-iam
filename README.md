<!--
SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
SPDX-FileCopyrightText: Potsdam Institute for Climate Impact Research (PIK)
SPDX-License-Identifier: CC-BY-4.0
-->

[![Upstream](https://img.shields.io/badge/upstream-pypsa--eur_v2026.02.0-blue)](https://github.com/PyPSA/pypsa-eur/releases/tag/v2026.02.0)
[![Snakemake](https://img.shields.io/badge/snakemake-≥9-brightgreen.svg?style=flat)](https://snakemake.readthedocs.io)


# PyPSA-Eur-IAM: Coupling PyPSA-Eur with Integrated Assessment Models

> **This is a fork of [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) maintained by the 
> [Potsdam Institute for Climate Impact Research (PIK)](https://www.pik-potsdam.de).**
> It extends PyPSA-Eur with modifications required to couple it with Integrated Assessment
> Models (IAMs) such as [REMIND](https://github.com/remindmodel/remind).

## What's different from upstream PyPSA-Eur

This fork tracks upstream PyPSA-Eur releases.

- **IAM coupling interface:** scripts and rules to exchange data with IAMs such as REMIND, using the [IAM-PyPSA-coupling](https://github.com/pik-piam/iam-pypsa-coupling) package
- **Simplified sector coupling:** The coupling currently uses a simplified structure of buses and links to represent sector coupling based on electricity demand profiles for different sectors. Details will be added.

Changes are kept as minimal and non-invasive as possible to simplify syncing with future upstream releases.

## Syncing with upstream

This fork is periodically synced with upstream PyPSA-Eur releases, currently `v2026.02.0`. Also see the tag `upstream-v2026.02.0`.

## Getting started

See the [upstream PyPSA-Eur documentation](https://pypsa-eur.readthedocs.io) for general usage. IAM-specific functionality will be documented in the [IAM-PyPSA-coupling](https://github.com/pik-piam/iam-pypsa-coupling) package in the future.

For now, see the following key files:

- `Snakefile_REMIND`: Main snakemake file for the coupling with IAMs, currently configured for REMIND.
  - Includes new wildcards for `iter_REMIND` (only used for bidirectional coupling) and `year_REMIND` for REMIND timesteps 
- `REMIND_coupling.smk`: Contains all new rules required for the coupling, in particular:
  - `import_REMIND_demand`: Importing electricity demand from REMIND
  - `downscale_REMIND_demand`: Downscaling electricity demand from REMIND regions to country level
  - `import_REMIND_capacities`: Importing power plant capacities from REMIND (optionally enforced per region in `installed_capacity_constraints_REMIND.py`)
  - `import_REMIND_co2price`: Importing CO2 price pathway from REMIND per region
  - `import_REMIND_costs`: Importing all required techno-economic parameters from REMIND. Note that costs are different across REMIND regions (PyPSA-Eur default is uniform costs across Europe)
  - `import_REMIND_hydro`: Special case for importing hydropower from REMIND. Current implementation adjusts PyPSA-Eur's inflow time series to match REMIND's capacity factor.
  - `adjust_powerplants_REMIND`: Adjusting PyPSA-Eur's power plant matching database to be consistent with REMIND's capacities. See file for further information.
  - `add_electricity_sector_REMIND`: Main file that builds the full network. Based on `add_electricity`, but includes additional sectoral demand profiles.
  - `prepare_network_REMIND`: Same as `prepare_network` with additional wildcards.
  - `solve_network_REMIND`: Same as `solve_network` with additional wildcards and using an optional SSH tunnel for Gurobi license verification if run on PIK HPC.
  - `export_to_REMIND`: Currently not in use!
- `config/config.remind.yaml`: Config file for REMIND coupling
- `config/technology_mapping_REMIND.yaml`: Per-technology parameter mapping file (IAM/PyPSA/fixed) for costs and capacities from canonical technology output of IAM-PyPSA-coupling package.
- `config/regionmapping_21_EU11.csv`: Region mapping file from REMIND to ISO.
- `scripts/remind`: All scripts for the new rules.

# Licence

PyPSA-Eur-IAM inherits the license of the upstream PyPSA-Eur project. Additional code contributed by PIK is also released under the MIT License.

The code in PyPSA-Eur is released as free software under the
[MIT License](https://opensource.org/licenses/MIT), see [`doc/licenses.rst`](doc/licenses.rst).
However, different licenses and terms of use may apply to the various
input data, see [`doc/data_sources.rst`](doc/data_sources.rst).
