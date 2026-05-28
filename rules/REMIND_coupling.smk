# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# Additional rules required for REMIND coupling
# Author: Adrian Odenweller (adrian.odenweller@pik-potsdam.de)

rule retrieve_ssp_data:
    params:
        ssp_scenario=config_provider("remind_coupling", "demand_downscaling", "ssp_scenario"),
        ssp_population_model=config_provider("remind_coupling", "demand_downscaling", "ssp_population_model"),
        ssp_gdp_model=config_provider("remind_coupling", "demand_downscaling", "ssp_gdp_model"),
    output:
        population="data/ssp/population.csv",
        gdp="data/ssp/gdp.csv",
    log:
        "logs/retrieve_ssp_data.log",
    benchmark:
        "benchmarks/retrieve_ssp_data"
    script:
        scripts("retrieve_ssp_data.py")


# Download and prepare all files, which are independent of REMIND inputs and
# requre internet connection on login nodes. Configure Gurobi before running.
#   export GRB_LICENSE_FILE=/p/projects/rd3mod/gurobi.lic
#   snakemake -s Snakefile_REMIND --cores 4 download_and_prepare_REMIND --omit-from add_electricity
# The --omit from flag makes sure that add_electricity isn't called
rule download_and_prepare_REMIND:
    input:
        expand(
            rules.add_electricity.output[0],
            clusters=config["scenario"]["clusters"],
        ),
        rules.retrieve_ssp_data.output,

# Before calling PyPSA-Eur the config file is created by import_REMIND_config.py
#
#   python scripts/import_REMIND_config.py \
#       --gdx resources/{scen}/i{iter}/REMIND2PyPSAEUR.gdx \
#       --config-changes-file config/config.remind_changes.yaml \
#       --config-changes-overrides "key=value; ..." \
#       --output resources/{scen}/i{iter}/config.remind_scenario.yaml
#
# Then start Snakemake with:
#   snakemake -s Snakefile_REMIND --configfile resources/{scen}/i{iter}/config.remind_scenario.yaml ...


# Input 1: Read demand data from REMIND and create a csv with the demand for each technology and region.
rule import_REMIND_demand:
    input:
        remind_data=ITERATION_RESOURCES + "REMIND2PyPSAEUR.gdx",
        region_mapping="config/regionmapping_21_EU11.csv",
    output:
        sectoral_load=ITERATION_RESOURCES + "sectoral_load.csv",
    log:
        ITERATION_LOGS + "import_REMIND_demand.log",
    benchmark:
        ITERATION_BENCHMARKS + "import_REMIND_demand"
    script:
        scripts("import_REMIND_demand.py")


# Input 1b: Disaggregate REMIND regional demand to country level using SSP population and GDP weights.
rule downscale_REMIND_demand:
    params:
        sector_weights=config_provider("remind_coupling", "demand_downscaling", "sector_weights"),
    input:
        sectoral_load=ITERATION_RESOURCES + "sectoral_load.csv",
        population="data/ssp/population.csv",
        gdp="data/ssp/gdp.csv",
        region_mapping="config/regionmapping_21_EU11.csv",
    output:
        sectoral_load_country=ITERATION_RESOURCES + "sectoral_load_country.csv",
    log:
        ITERATION_LOGS + "downscale_REMIND_demand.log",
    benchmark:
        ITERATION_BENCHMARKS + "downscale_REMIND_demand"
    script:
        scripts("downscale_REMIND_demand.py")


# Input 2: Read capacity data from REMIND and create a csv with the installed capacities for each technology and region.
rule import_REMIND_capacities:
    input:
        remind_data=ITERATION_RESOURCES + "REMIND2PyPSAEUR.gdx",
        region_mapping="config/regionmapping_21_EU11.csv",
        technology_cost_mapping="config/technology_cost_mapping.csv",
    output:
        capacities=ITERATION_RESOURCES + "installed_capacities.csv",
    log:
        ITERATION_LOGS + "import_REMIND_capacities.log",
    benchmark:
        ITERATION_BENCHMARKS + "import_REMIND_capacities"
    script:
        scripts("import_REMIND_capacities.py")

# Input 3: Read CO2 price pathway from REMIND and create a csv with year and CO2 price.
rule import_REMIND_co2price:
    input:
        remind_data=ITERATION_RESOURCES + "REMIND2PyPSAEUR.gdx",
        region_mapping="config/regionmapping_21_EU11.csv",
    output:
        co2_price=ITERATION_RESOURCES + "co2_price.csv",
    log:
        ITERATION_LOGS + "import_REMIND_co2price.log",
    benchmark:
        ITERATION_BENCHMARKS + "import_REMIND_co2price"
    script:
        scripts("import_REMIND_co2price.py")

# Input 4: Build cost data compatible with the original PyPSA-EUR cost data structure
rule import_REMIND_costs:
    params:
        costs=config_provider("costs"),
        max_hours=config_provider("electricity", "max_hours"),
    input:
        # no TD data for < 2020 and > 2050
        original_costs=lambda w: f"resources/costs_{max(2020, min(int(w['year_REMIND']), 2050))}.csv",
        network=resources("networks/base_s.nc"),
        custom_costs=config_provider("costs", "custom_cost_fn"),
        region_mapping="config/regionmapping_21_EU11.csv",
        technology_cost_mapping="config/technology_cost_mapping.csv",
        remind_data=ITERATION_RESOURCES + "REMIND2PyPSAEUR.gdx",
    output:
        costs_processed=ITERATION_RESOURCES + "y{year_REMIND}/costs_processed.csv",
        costs_raw_overwritten=ITERATION_RESOURCES + "y{year_REMIND}/costs_raw_overwritten.csv",
    log:
        ITERATION_LOGS + "y{year_REMIND}/import_REMIND_costs.log",
    benchmark:
        ITERATION_BENCHMARKS + "y{year_REMIND}/import_REMIND_costs"
    group:
        "iy"
    script:
        scripts("import_REMIND_costs.py")

# Input 5: Special case for hydro, read capacity and generation from REMIND.
# In add_electricity_sector_REMIND, hydro infeed and capacity is adjusted to follow REMIND's capacity factor.
rule import_REMIND_hydro:
    input:
        remind_data=ITERATION_RESOURCES + "REMIND2PyPSAEUR.gdx",
        region_mapping="config/regionmapping_21_EU11.csv",
    output:
        hydro_targets=ITERATION_RESOURCES + "hydro_targets.csv",
    log:
        ITERATION_LOGS + "import_REMIND_hydro.log",
    benchmark:
        ITERATION_BENCHMARKS + "import_REMIND_hydro"
    script:
        scripts("import_REMIND_hydro.py")

# Adjust powerplants database based on REMIND capacities
rule adjust_powerplants_REMIND:
    params:
        countries=config_provider("countries"),
    input:
        powerplants=resources("powerplants_s_{clusters}.csv"),
        capacities=ITERATION_RESOURCES + "installed_capacities.csv",
        region_mapping="config/regionmapping_21_EU11.csv",
        technology_mapping="config/technology_cost_mapping.csv",
    output:
        powerplants_adjusted=SCENARIO_RESOURCES + "i{iter_REMIND}/y{year_REMIND}/powerplants_adjusted_s_{clusters}.csv",
    log:
        LOGS + "{scen_REMIND}/i{iter_REMIND}/y{year_REMIND}/adjust_powerplants_s_{clusters}.log",
    benchmark:
        BENCHMARKS + "{scen_REMIND}/i{iter_REMIND}/y{year_REMIND}/adjust_powerplants_s_{clusters}"
    group:
        "iy"
    script:
        scripts("adjust_powerplants_REMIND.py")

# New rule to create hourly water heat demand based on BDEW space heating profile
# This looks more plausible than the flat water heat profile in the BDEW data
# This rule does not create a DSM profile for heating (introduced in v2026.02.0)
rule build_hourly_water_heat_demand_REMIND:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
    input:
        heat_profile="data/heat_load_profile_BDEW.csv",
        clustered_pop_layout=resources("pop_layout_base_s_{clusters}.csv"),
    output:
        water_heat_demand=resources("hourly_water_heat_demand_total_base_s_{clusters}.nc"),
    resources:
        mem_mb=2000,
    threads: 8
    log:
        logs("build_hourly_water_heat_demand_total_s_{clusters}.loc"),
    benchmark:
        benchmarks("build_hourly_water_heat_demand/total_s_{clusters}")
    script:
        "../scripts/build_hourly_water_heat_demand_REMIND.py"

# Helper function for optional EDGE-T fleet file
def _get_fleet_input(wildcards):
    fleet_path = f"resources/{wildcards.scen_REMIND}/i{wildcards.iter_REMIND}/fleetVehiclesPerTech.RDS"
    exists = os.path.exists(fleet_path)
    if exists:
        return fleet_path
    else:
        return []

# This is the main rule that brings all REMIND inputs together and creates the electricity network for the given scenario, iteration and year.
# TODO: Revert to default add_electricity and use this rule to overwrite afterwards?
# TODO: Make this call carrier_to_remind.yaml to name network components
rule add_electricity_sector_REMIND:
    message:
        "Adding electricity and sector coupling to REMIND-coupled network"
    params:
        # Standard parameters
        line_length_factor=config_provider("lines", "length_factor"),
        link_length_factor=config_provider("links", "length_factor"),
        scaling_factor=config_provider("load", "scaling_factor"),
        countries=config_provider("countries"),
        snapshots=config_provider("snapshots"),
        renewable=config_provider("renewable"),
        electricity=config_provider("electricity"),
        conventional=config_provider("conventional"),
        foresight=config_provider("foresight"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        consider_efficiency_classes=config_provider(
            "clustering", "consider_efficiency_classes"
        ),
        aggregation_strategies=config_provider("clustering", "aggregation_strategies"),
        exclude_carriers=config_provider("clustering", "exclude_carriers"),
        sector=config_provider("sector"),
        # REMIND parameters
        preinstalled_capacities=config_provider("remind_coupling","preinstalled_capacities"),
        h2_settings=config_provider("remind_coupling", "hydrogen_storage"),
        battery_settings=config_provider("remind_coupling", "battery_storage"),
        sector_coupling=config_provider("remind_coupling","sector_coupling"),
    input:
        # Standard input files
        unpack(input_profile_tech),
        unpack(input_class_regions),
        unpack(input_conventional),
        base_network=resources("networks/base_s_{clusters}.nc"),
        costs=ITERATION_RESOURCES + "y{year_REMIND}/costs_processed.csv",  # REMIND specific
        regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        powerplants=ITERATION_RESOURCES + "y{year_REMIND}/powerplants_adjusted_s_{clusters}.csv",  # REMIND specific
        hydro_capacities=ancient("data/hydro_capacities.csv"),
        unit_commitment="data/unit_commitment.csv",
        fuel_price=lambda w: (
            resources("monthly_fuel_price.csv")
            if config_provider("conventional", "dynamic_fuel_price")(w)
            else []
        ),
        load=resources("electricity_demand_base_s.nc"),
        busmap=resources("busmap_base_s_{clusters}.csv"),
        # Required sector coupling input files
        # Hydrogen salt cavern potential
        h2_cavern=resources("salt_cavern_potentials_s_{clusters}.csv"),
        # EV input files
        transport_demand=resources("transport_demand_s_{clusters}.csv"),
        transport_data=resources("transport_data_s_{clusters}.csv"),
        avail_profile=resources("avail_profile_s_{clusters}.csv"),
        dsm_profile=resources("dsm_profile_s_{clusters}.csv"),
        temp_air_total=resources("temp_air_total_base_s_{clusters}.nc"),
        # Heating input files
        hourly_heat_demand_total=resources("hourly_heat_demand_total_base_s_{clusters}.nc"),
        cop_profiles=resources("cop_profiles_base_s_{clusters}_2030.nc"),  # Use arbitrary planning_horizons wildcard
        hourly_water_heat_demand_total=resources("hourly_water_heat_demand_total_base_s_{clusters}.nc"),  # From new rule above
        # REMIND input files
        region_mapping="config/regionmapping_21_EU11.csv",
        technology_cost_mapping="config/technology_cost_mapping.csv",
        sectoral_load_country=ITERATION_RESOURCES + "sectoral_load_country.csv",
        hydro_targets=ITERATION_RESOURCES + "hydro_targets.csv",
        capacities=ITERATION_RESOURCES + "installed_capacities.csv",
        wh_share="data/REMIND_SSP2_wh_share.csv",  # REMIND share of water heating
        fleet_file=_get_fleet_input,  # EDGE-T fleet
    output:
        ITERATION_RESOURCES + "y{year_REMIND}/networks/base_s_{clusters}_elec.nc",
    log:
        ITERATION_LOGS + "y{year_REMIND}/add_electricity_{clusters}.log",
    benchmark:
        ITERATION_BENCHMARKS + "y{year_REMIND}/add_electricity_{clusters}"
    threads: 1
    group:
        "iy"
    resources:
        mem_mb=10000,
    script:
        scripts("add_electricity_sector_REMIND.py")


# Helper function to read CO2 prices from CSV to feed into prepare_network_REMIND parameter without changing the script
def _remind_emission_prices(wildcards, input):
    co2_prices = pd.read_csv(str(input.remind_co2_price)).set_index("year")["co2_price"]
    return {
        "enable": True,
        "dynamic": False,
        "co2": float(co2_prices.loc[int(wildcards.year_REMIND)]),
    }


# Prepare network, specifically adding the CO2 price
rule prepare_network_REMIND:
    message:
        "Preparing network for model with {wildcards.clusters} clusters and options {wildcards.opts}"
    params:
        time_resolution=config_provider("clustering", "temporal", "resolution_elec"),
        links=config_provider("links"),
        lines=config_provider("lines"),
        co2base=config_provider("electricity", "co2base"),
        co2limit_enable=config_provider("electricity", "co2limit_enable", default=False),
        co2limit=config_provider("electricity", "co2limit"),
        gaslimit_enable=config_provider("electricity", "gaslimit_enable", default=False),
        gaslimit=config_provider("electricity", "gaslimit"),
        emission_prices=_remind_emission_prices,  # REMIND specific
        adjustments=config_provider("adjustments", "electricity"),
        autarky=config_provider("electricity", "autarky", default={}),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        transmission_limit=config_provider("electricity", "transmission_limit"),
    input:
        ITERATION_RESOURCES + "y{year_REMIND}/networks/base_s_{clusters}_elec.nc",
        costs=ITERATION_RESOURCES + "y{year_REMIND}/costs_processed.csv",  # REMIND specific
        remind_co2_price=ITERATION_RESOURCES + "co2_price.csv",  # REMIND specific; triggers import_REMIND_co2price
        co2_price=lambda w: (
            resources("co2_price.csv")
            if config_provider("costs", "emission_prices", "dynamic")(w)
            else []
        ),
    output:
        ITERATION_RESOURCES + "y{year_REMIND}/networks/base_s_{clusters}_elec_{opts}.nc",
    log:
        ITERATION_LOGS + "y{year_REMIND}/prepare_network_base_s_{clusters}_elec_{opts}.log",
    benchmark:
        ITERATION_BENCHMARKS + "y{year_REMIND}/prepare_network_base_s_{clusters}_elec_{opts}",
    threads: 1
    group:
        "iy"
    resources:
        mem_mb=4000,
    script:
        scripts("prepare_network.py")


# Setup ssh tunnel to login node if configured to run on PIK HPC
USE_GUROBI_TUNNEL_REMIND = config.get("solving", {}).get("gurobi_hpc_tunnel", {}).get(
    "use_tunnel", False
)
GUROBI_TUNNEL_READY_MARKER_REMIND = (
    ITERATION_RESULTS
    + "y{year_REMIND}/logs/solve_network/base_s_{clusters}_elec_{opts}_gurobi_tunnel_ready.txt"
)

if USE_GUROBI_TUNNEL_REMIND:
    rule setup_gurobi_tunnel_REMIND:
        message:
            "Setting up Gurobi tunnel before solving {wildcards.clusters} clusters and {wildcards.opts} options"
        output:
            ready_marker=GUROBI_TUNNEL_READY_MARKER_REMIND,
        log:
            python=ITERATION_RESULTS
            + "y{year_REMIND}/logs/solve_network/base_s_{clusters}_elec_{opts}_gurobi_tunnel_python.log",
        benchmark:
            ITERATION_RESULTS
            + "y{year_REMIND}/benchmarks/solve_network/base_s_{clusters}_elec_{opts}_gurobi_tunnel",
        threads: 1
        group:
            "iy"
        resources:
            mem_mb=500,
            runtime="00:05:00",
        script:
            scripts("setup_gurobi_tunnel_REMIND.py")


# Solve network for given scenario, iteration and year.
# Additional constraint on capacities per region is passed via custom_extra_functionality,
# which requires further input files from REMIND
rule solve_network_REMIND:
    message:
        "Solving electricity network optimization for {wildcards.clusters} clusters and {wildcards.opts} electric options"
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,  # REMIND-specific path provided in config
    input:
        network=ITERATION_RESOURCES + "y{year_REMIND}/networks/base_s_{clusters}_elec_{opts}.nc",
        # REMIND input files for capacity constraint
        capacities=ITERATION_RESOURCES + "installed_capacities.csv",
        region_mapping="config/regionmapping_21_EU11.csv",
        technology_cost_mapping="config/technology_cost_mapping.csv",
        # Optional marker to wait for Gurobi tunnel to be ready if configured
        tunnel_ready=(GUROBI_TUNNEL_READY_MARKER_REMIND if USE_GUROBI_TUNNEL_REMIND else []),
    output:
        network=ITERATION_RESULTS + "y{year_REMIND}/networks/base_s_{clusters}_elec_{opts}.nc",
        config=ITERATION_RESULTS + "y{year_REMIND}/configs/config.base_s_{clusters}_elec_{opts}.yaml",
        model=(
            ITERATION_RESULTS + "y{year_REMIND}/models/base_s_{clusters}_elec_{opts}.nc"
            if config["solving"]["options"]["store_model"]
            else []
        ),
    log:
        solver=normpath(
            ITERATION_RESULTS + "y{year_REMIND}/logs/solve_network/base_s_{clusters}_elec_{opts}_solver.log"
        ),
        memory=ITERATION_RESULTS + "y{year_REMIND}/logs/solve_network/base_s_{clusters}_elec_{opts}_memory.log",
        python=ITERATION_RESULTS + "y{year_REMIND}/logs/solve_network/base_s_{clusters}_elec_{opts}_python.log",
    benchmark:
        (ITERATION_RESULTS + "y{year_REMIND}/benchmarks/solve_network/base_s_{clusters}_elec_{opts}")
    threads: solver_threads
    group:
        "iy"
    resources:
        mem_mb=memory,
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    script:
        scripts("solve_network.py")


# Expand rule to solve all networks across years.
rule solve_all_networks_REMIND:
    input:
        networks=lambda wildcards: expand(
            ITERATION_RESULTS + "y{year_REMIND}/networks/base_s_{clusters}_elec_{opts}.nc",
            scen_REMIND=[wildcards.scen_REMIND],  # Given by output path in export_to_REMIND
            iter_REMIND=[wildcards.iter_REMIND],  # Given by output path in export_to_REMIND
            clusters=config["scenario"]["clusters"],  # Given by config
            opts=[""],  # Empty because CO2 price is inserted in prepare_network_REMIND
            year_REMIND=config["remind_coupling"]["years"],  # set by import_REMIND_config.py via --configfile
        ),
    output:
        # Marker file to call rule and populate wildcards if needed
        # snakemake -s Snakefile_REMIND results/{scen_REMIND}/i{iter_REMIND}/solve_all_networks_REMIND.done
        touch(ITERATION_RESULTS + "solve_all_networks_REMIND.done")


# Export PyPSA-Eur results to REMIND as GDX and create additional reporting CSVs
rule export_to_REMIND:
    params:
        remind_settings=config_provider("remind_coupling"),
    input:
        networks=rules.solve_all_networks_REMIND.input["networks"],
        region_mapping="config/regionmapping_21_EU11.csv",
        technology_cost_mapping="config/technology_cost_mapping.csv",
        carrier_mapping="config/carrier_to_remind.yaml",
    output:
        # Main output file that is read by REMIND
        gdx=ITERATION_RESULTS + "PyPSAEUR2REMIND.gdx",
        # coupling_parameters/ contains the same data as the GDX as CSVs
        coupling_parameters=directory(ITERATION_RESULTS + "coupling_parameters"),
        # reporting_parameters/ contains additional diagnostics (not used by REMIND)
        reporting_parameters=directory(ITERATION_RESULTS + "reporting_parameters"),
    log:
        ITERATION_LOGS + "export_to_REMIND.log",
    benchmark:
        ITERATION_BENCHMARKS + "export_to_REMIND"
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 30000,
        walltime="00:10:00",
    script:
        scripts("export_to_REMIND.py")