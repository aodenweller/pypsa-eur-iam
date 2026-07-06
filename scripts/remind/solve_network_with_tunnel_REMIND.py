# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
REMIND-specific solve entry point: sets up the Gurobi SSH tunnel when configured,
then executes the upstream solve_network.py in the same process.

Using exec() rather than a separate tunnel rule means:
- env vars set here (https_proxy, GRB_LICENSE_FILE, GRB_CAFILE) are visible to
  gurobipy at initialisation time, with no cross-process propagation problems.
- There is no stale marker file that can cause the tunnel to be skipped on reruns
  after a failed solve.
"""

import logging
from pathlib import Path
from typing import Any

from _helpers import (
    check_gurobi_license,
    configure_logging,
    is_tunnel_alive,
    setup_gurobi_tunnel_and_env,
)

logger = logging.getLogger(__name__)

# Provided by Snakemake at runtime when used via `script:`.
snakemake: Any

if __name__ == "__main__":
    configure_logging(snakemake)

    solving_cfg = snakemake.config.get("solving", {})
    solver_name = solving_cfg.get("solver", {}).get("name")
    tunnel_cfg = solving_cfg.get("gurobi_hpc_tunnel", None)

    should_setup_tunnel = (
        solver_name == "gurobi"
        and tunnel_cfg is not None
        and tunnel_cfg.get("use_tunnel", False)
    )

    if should_setup_tunnel:
        logger.info("Setting up Gurobi tunnel before solve.")
        setup_gurobi_tunnel_and_env(tunnel_cfg, logger=logger)

        if not is_tunnel_alive(tunnel_cfg):
            logger.warning("Tunnel not alive after first attempt; retrying.")
            setup_gurobi_tunnel_and_env(tunnel_cfg, logger=logger)

        if not is_tunnel_alive(tunnel_cfg):
            raise RuntimeError("Failed to establish Gurobi tunnel.")

        timeout = int(tunnel_cfg.get("timeout_s", 10))
        if not check_gurobi_license(attempts=1, timeout=timeout):
            raise RuntimeError("Gurobi license check failed before solve.")
    else:
        logger.info("Skipping tunnel setup (not configured or solver is not gurobi).")

    # Execute the upstream solve script in this process so that env vars set above
    # are inherited by gurobipy when it initialises.
    _script = Path(__file__).parent / "solve_network.py"
    exec(compile(_script.read_text(), str(_script), "exec"))
