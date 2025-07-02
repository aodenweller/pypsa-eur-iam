# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
# This file contains those rules of build_sector.smk that are required for the REMIND coupling
# This currently includes input data for modelling EVs and heating


rule build_population_layouts:
    input:
        nuts3_shapes=resources("nuts3_shapes.geojson"),
        urban_percent="data/worldbank/API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2.csv",
        cutout=lambda w: input_cutout(w),
    output:
        pop_layout_total=resources("pop_layout_total.nc"),
        pop_layout_urban=resources("pop_layout_urban.nc"),
        pop_layout_rural=resources("pop_layout_rural.nc"),
    log:
        logs("build_population_layouts.log"),
    resources:
        mem_mb=20000,
    benchmark:
        benchmarks("build_population_layouts")
    threads: 8
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_population_layouts.py"


rule build_clustered_population_layouts:
    input:
        pop_layout_total=resources("pop_layout_total.nc"),
        pop_layout_urban=resources("pop_layout_urban.nc"),
        pop_layout_rural=resources("pop_layout_rural.nc"),
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
        cutout=lambda w: input_cutout(w),
    output:
        clustered_pop_layout=resources("pop_layout_base_s_{clusters}.csv"),
    log:
        logs("build_clustered_population_layouts_s_{clusters}.log"),
    resources:
        mem_mb=10000,
    benchmark:
        benchmarks("build_clustered_population_layouts/s_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_clustered_population_layouts.py"


rule build_simplified_population_layouts:
    input:
        pop_layout_total=resources("pop_layout_total.nc"),
        pop_layout_urban=resources("pop_layout_urban.nc"),
        pop_layout_rural=resources("pop_layout_rural.nc"),
        regions_onshore=resources("regions_onshore_base_s.geojson"),
        cutout=lambda w: input_cutout(w),
    output:
        clustered_pop_layout=resources("pop_layout_base_s.csv"),
    resources:
        mem_mb=10000,
    log:
        logs("build_simplified_population_layouts_s"),
    benchmark:
        benchmarks("build_simplified_population_layouts/s")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_clustered_population_layouts.py"


rule build_temperature_profiles:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
    input:
        pop_layout=resources("pop_layout_total.nc"),
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
        cutout=lambda w: input_cutout(
            w, config_provider("sector", "heat_demand_cutout")(w)
        ),
    output:
        temp_soil=resources("temp_soil_total_base_s_{clusters}.nc"),
        temp_air=resources("temp_air_total_base_s_{clusters}.nc"),
    resources:
        mem_mb=20000,
    threads: 8
    log:
        logs("build_temperature_profiles_total_s_{clusters}.log"),
    benchmark:
        benchmarks("build_temperature_profiles/total_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_temperature_profiles.py"


rule build_energy_totals:
    params:
        countries=config_provider("countries"),
        energy=config_provider("energy"),
    input:
        nuts3_shapes=resources("nuts3_shapes.geojson"),
        co2="data/bundle/eea/UNFCCC_v23.csv",
        swiss="data/switzerland-new_format-all_years.csv",
        swiss_transport="data/gr-e-11.03.02.01.01-cc.csv",
        idees="data/jrc-idees-2021",
        district_heat_share="data/district_heat_share.csv",
        eurostat="data/eurostat/Balances-April2023",
        eurostat_households="data/eurostat/eurostat-household_energy_balances-february_2024.csv",
    output:
        transformation_output_coke=resources("transformation_output_coke.csv"),
        energy_name=resources("energy_totals.csv"),
        co2_name=resources("co2_totals.csv"),
        transport_name=resources("transport_data.csv"),
        district_heat_share=resources("district_heat_share.csv"),
        heating_efficiencies=resources("heating_efficiencies.csv"),
    threads: 16
    resources:
        mem_mb=10000,
    log:
        logs("build_energy_totals.log"),
    benchmark:
        benchmarks("build_energy_totals")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_energy_totals.py"


rule build_salt_cavern_potentials:
    input:
        salt_caverns="data/bundle/h2_salt_caverns_GWh_per_sqkm.geojson",
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
        regions_offshore=resources("regions_offshore_base_s_{clusters}.geojson"),
    output:
        h2_cavern_potential=resources("salt_cavern_potentials_s_{clusters}.csv"),
    threads: 1
    resources:
        mem_mb=2000,
    log:
        logs("build_salt_cavern_potentials_s_{clusters}.log"),
    benchmark:
        benchmarks("build_salt_cavern_potentials_s_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_salt_cavern_potentials.py"


rule build_population_weighted_energy_totals:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
    input:
        energy_totals=resources("{kind}_totals.csv"),
        clustered_pop_layout=resources("pop_layout_base_s_{clusters}.csv"),
    output:
        resources("pop_weighted_{kind}_totals_s_{clusters}.csv"),
    threads: 1
    resources:
        mem_mb=2000,
    log:
        logs("build_population_weighted_{kind}_totals_{clusters}.log"),
    benchmark:
        benchmarks("build_population_weighted_{kind}_totals_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_population_weighted_energy_totals.py"


rule build_transport_demand:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        sector=config_provider("sector"),
        energy_totals_year=config_provider("energy", "energy_totals_year"),
    input:
        clustered_pop_layout=resources("pop_layout_base_s_{clusters}.csv"),
        pop_weighted_energy_totals=resources(
            "pop_weighted_energy_totals_s_{clusters}.csv"
        ),
        transport_data=resources("transport_data.csv"),
        traffic_data_KFZ="data/bundle/emobility/KFZ__count",
        traffic_data_Pkw="data/bundle/emobility/Pkw__count",
        temp_air_total=resources("temp_air_total_base_s_{clusters}.nc"),
    output:
        transport_demand=resources("transport_demand_s_{clusters}.csv"),
        transport_data=resources("transport_data_s_{clusters}.csv"),
        avail_profile=resources("avail_profile_s_{clusters}.csv"),
        dsm_profile=resources("dsm_profile_s_{clusters}.csv"),
    threads: 1
    resources:
        mem_mb=2000,
    log:
        logs("build_transport_demand_s_{clusters}.log"),
    benchmark:
        benchmarks("build_transport_demand/s_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_transport_demand.py"


rule build_daily_heat_demand:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
    input:
        pop_layout=resources("pop_layout_total.nc"),
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
        cutout=lambda w: input_cutout(
            w, config_provider("sector", "heat_demand_cutout")(w)
        ),
    output:
        heat_demand=resources("daily_heat_demand_total_base_s_{clusters}.nc"),
    resources:
        mem_mb=20000,
    threads: 8
    log:
        logs("build_daily_heat_demand_total_s_{clusters}.loc"),
    benchmark:
        benchmarks("build_daily_heat_demand/total_s_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_daily_heat_demand.py"


rule build_hourly_heat_demand:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
    input:
        heat_profile="data/heat_load_profile_BDEW.csv",
        heat_demand=resources("daily_heat_demand_total_base_s_{clusters}.nc"),
    output:
        heat_demand=resources("hourly_heat_demand_total_base_s_{clusters}.nc"),
    resources:
        mem_mb=2000,
    threads: 8
    log:
        logs("build_hourly_heat_demand_total_s_{clusters}.loc"),
    benchmark:
        benchmarks("build_hourly_heat_demand/total_s_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_hourly_heat_demand.py"


# Removed planning_horizons wildcard (in both rule and code)
rule build_central_heating_temperature_profiles_REMIND:
    params:
        max_forward_temperature_central_heating_baseyear=config_provider(
            "sector",
            "district_heating",
            "supply_temperature_approximation",
            "max_forward_temperature_baseyear",
        ),
        min_forward_temperature_central_heating_baseyear=config_provider(
            "sector",
            "district_heating",
            "supply_temperature_approximation",
            "min_forward_temperature_baseyear",
        ),
        return_temperature_central_heating_baseyear=config_provider(
            "sector",
            "district_heating",
            "supply_temperature_approximation",
            "return_temperature_baseyear",
        ),
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        lower_threshold_ambient_temperature=config_provider(
            "sector",
            "district_heating",
            "supply_temperature_approximation",
            "lower_threshold_ambient_temperature",
        ),
        upper_threshold_ambient_temperature=config_provider(
            "sector",
            "district_heating",
            "supply_temperature_approximation",
            "upper_threshold_ambient_temperature",
        ),
        rolling_window_ambient_temperature=config_provider(
            "sector",
            "district_heating",
            "supply_temperature_approximation",
            "rolling_window_ambient_temperature",
        ),
        relative_annual_temperature_reduction=config_provider(
            "sector",
            "district_heating",
            "supply_temperature_approximation",
            "relative_annual_temperature_reduction",
        ),
        energy_totals_year=config_provider("energy", "energy_totals_year"),
    input:
        temp_air_total=resources("temp_air_total_base_s_{clusters}.nc"),
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
    output:
        central_heating_forward_temperature_profiles=resources(
            "central_heating_forward_temperature_profiles_base_s_{clusters}.nc"
        ),
        central_heating_return_temperature_profiles=resources(
            "central_heating_return_temperature_profiles_base_s_{clusters}.nc"
        ),
    resources:
        mem_mb=20000,
    log:
        logs(
            "build_central_heating_temperature_profiles_s_{clusters}.log"
        ),
    benchmark:
        benchmarks(
            "build_central_heating_temperature_profiles/s_{clusters}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_central_heating_temperature_profiles/run_REMIND.py"


# Removed planning_horizons wildcard from rule (no change in code necessary)
rule build_cop_profiles_REMIND:
    params:
        heat_pump_sink_T_decentral_heating=config_provider(
            "sector", "heat_pump_sink_T_individual_heating"
        ),
        heat_source_cooling_central_heating=config_provider(
            "sector", "district_heating", "heat_source_cooling"
        ),
        heat_pump_cop_approximation_central_heating=config_provider(
            "sector", "district_heating", "heat_pump_cop_approximation"
        ),
        heat_pump_sources=config_provider("sector", "heat_pump_sources"),
        limited_heat_sources=config_provider(
            "sector", "district_heating", "limited_heat_sources"
        ),
        snapshots=config_provider("snapshots"),
    input:
        central_heating_forward_temperature_profiles=resources(
            "central_heating_forward_temperature_profiles_base_s_{clusters}.nc"
        ),
        central_heating_return_temperature_profiles=resources(
            "central_heating_return_temperature_profiles_base_s_{clusters}.nc"
        ),
        temp_soil_total=resources("temp_soil_total_base_s_{clusters}.nc"),
        temp_air_total=resources("temp_air_total_base_s_{clusters}.nc"),
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
    output:
        cop_profiles=resources("cop_profiles_base_s_{clusters}.nc"),
    resources:
        mem_mb=20000,
    log:
        logs("build_cop_profiles_s_{clusters}.log"),
    benchmark:
        benchmarks("build_cop_profiles/s_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_cop_profiles/run.py"