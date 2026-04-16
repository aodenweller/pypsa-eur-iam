# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


rule solve_network_REMIND:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=ITERATION_RESOURCES
        + "y{year}/networks/base_s_{clusters}_elec_{opts}.nc",
        RCL_p_nom_limits=ITERATION_RESOURCES + "y{year}/RCL_p_nom_limits_updated_s_{clusters}.csv",
        region_mapping="config/regionmapping_21_EU11.csv",
        technology_cost_mapping="config/technology_cost_mapping.csv",
    output:
        network=ITERATION_RESULTS
        + "y{year}/networks/base_s_{clusters}_elec_{opts}.nc",
        config=ITERATION_RESULTS
        + "y{year}/configs/config.base_s_{clusters}_elec_{opts}.yaml",
    log:
        solver=normpath(
            ITERATION_LOGS
            + "y{year}/solve_network/base_s_{clusters}_elec_{opts}_solver.log"
        ),
        memory=ITERATION_LOGS
        + "y{year}/solve_network/base_s_{clusters}_elec_{opts}_memory.log",
        python=ITERATION_LOGS
        + "y{year}/solve_network/base_s_{clusters}_elec_{opts}_python.log",
    benchmark:
        (
            ITERATION_BENCHMARKS
            + "y{year}/solve_network/base_s_{clusters}_elec_{opts}"
        )
    threads: solver_threads
    group:
        "iy"
    resources:
        mem_mb=memory,
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network_REMIND.py"

