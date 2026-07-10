<!--
SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
SPDX-FileCopyrightText: Potsdam Institute for Climate Impact Research (PIK)
SPDX-License-Identifier: CC-BY-4.0
-->

![Upstream](https://img.shields.io/badge/upstream-pypsa--eur_v2026.02.0-blue)
[![Zenodo PyPSA-Eur](https://zenodo.org/badge/DOI/10.5281/zenodo.3520874.svg)](https://doi.org/10.5281/zenodo.3520874)
[![Zenodo PyPSA-Eur-Sec](https://zenodo.org/badge/DOI/10.5281/zenodo.3938042.svg)](https://doi.org/10.5281/zenodo.3938042)
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
- `REMIND_coupling.smk`: Contains all new rules.
- `config/config.remind.yaml`: Config file for REMIND coupling
- `config/technology_cost_mapping.csv`: File to map all costs from REMIND output, with a few PyPSA fallbacks.
- `scripts/remind`: All scripts for the new rules.

# Licence

PyPSA-Eur-IAM inherits the license of the upstream PyPSA-Eur project. Additional code contributed by PIK is also released under the MIT License.

The code in PyPSA-Eur is released as free software under the
[MIT License](https://opensource.org/licenses/MIT), see [`doc/licenses.rst`](doc/licenses.rst).
However, different licenses and terms of use may apply to the various
input data, see [`doc/data_sources.rst`](doc/data_sources.rst).
