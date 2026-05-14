#!/usr/bin/env python3
"""
CLI script to generate a scenario-specific config for REMIND-PyPSA-Eur coupling.
Must be run before invoking Snakemake.

Usage:
    python scripts/import_REMIND_config.py \
        --gdx resources/{scen}/i{iter}/REMIND2PyPSAEUR.gdx \
        --config-remind config/config.remind.yaml
        --config-changes-file config/config.remind_changes.yaml \
        --config-changes-overrides "remind_coupling.battery_storage_e_min_pu=0.2; remind_coupling.sector_coupling.enable_ev=true" \
        --output resources/{scen}/i{iter}/config.remind_scenario.yaml \
"""
import argparse
import logging
import sys
from functools import reduce
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import read_remind_data

logger = logging.getLogger(__name__)


def set_nested_value(config, key_path, value):
    keys = key_path.split(".")
    last_key = keys.pop()
    nested_dict = reduce(lambda d, k: d.setdefault(k, {}), keys, config)
    nested_dict[last_key] = value


def has_nested_key(config: dict, key_path: str) -> bool:
    """Return True if key_path (dot-separated) exists in config."""
    d = config
    for k in key_path.split("."):
        if not isinstance(d, dict) or k not in d:
            return False
        d = d[k]
    return True


def apply_overrides(config, override_string, reference_config=None):
    """Apply semicolon-separated key=value overrides to config dict."""
    for part in override_string.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        if reference_config is not None and not has_nested_key(reference_config, key):
            logger.warning(
                "Override key '%s' not found in config.remind.yaml — possible typo?", key
            )
        set_nested_value(config, key, yaml.safe_load(val.strip()))
        logger.info("Applied override: %s = %s", key, val.strip())


def read_years(gdx_path):
    """Read coupled years from tPy32 set in REMIND config GDX."""
    return sorted(
        read_remind_data(
            gdx_path,
            "tPy32",
            rename_columns={"ttot": "year"},
        )
        .year.unique()
        .astype(int)
        .tolist()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate scenario-specific PyPSA config from REMIND GDX data."
    )
    parser.add_argument(
        "--gdx", required=True, help="Path to REMIND2PyPSAEUR.gdx"
    )
    parser.add_argument(
        "--config-remind",
        default="config/config.remind.yaml",
        help="Path to config.remind.yaml used to validate override keys (default: config/config.remind.yaml)",
    )
    parser.add_argument(
        "--config-changes-file",
        required=True,
        help="Path to the YAML changes file (e.g. config/config.remind_changes.yaml)",
    )
    parser.add_argument(
        "--config-changes-overrides",
        default="",
        help='Semicolon-separated key=value overrides, e.g. "remind_coupling.battery_storage_e_min_pu=0.2; remind_coupling.sector_coupling.enable_ev=true"',
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for config.remind_scenario.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    reference_config = {}
    remind_cfg_path = Path(args.config_remind)
    if remind_cfg_path.exists():
        with open(remind_cfg_path) as f:
            reference_config = yaml.safe_load(f) or {}
    else:
        logger.warning(
            "Reference config '%s' not found; skipping override key validation.", args.config_remind
        )

    with open(args.config_changes_file) as f:
        config = yaml.safe_load(f) or {}

    if args.config_changes_overrides:
        apply_overrides(config, args.config_changes_overrides, reference_config=reference_config)

    years = read_years(args.gdx)
    set_nested_value(config, "remind_coupling.years", years)
    logger.info("REMIND coupled years: %s", years)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Written to %s", args.output)


if __name__ == "__main__":
    main()
