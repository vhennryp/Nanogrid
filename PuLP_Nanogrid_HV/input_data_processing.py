""""
========================== Input data for the optimal operation =========================
"""
import json

# Opening JSON file
with open('input_data_NANOGRID_15min.json') as json_file:
    data = json.load(json_file)

T = data["set_of_time"]
S = data["set_of_scenarios"]
O = data["set_of_outage"]
Prob = data["probability_of_the_scenarios"]
Smax = data["maximum_power_PCC"]
delta_t = data["variation_of_time"]
cEDS = {}
for index in range(len(data["cost_of_the_energy"])):
	cEDS[T[index]] = data["cost_of_the_energy"][index]
PD = {}
for index in range(len(data["active_load"])):
	PD[T[index]] = data["active_load"][index]
PV = data["photovoltaic_generation"]
Pi_min = data["minimum_charging_limit_ess"]
Pi_max = data["maximum_charging_limit_ess"]
Pe_min = data["minimum_discharging_limit_ess"]
Pe_max = data["maximum_discharging_limit_ess"]
Ebi = data["initial_energy_of_the_ess"]
Eb_min = data["minimum_energy_capacity_ess"]
Eb_max = data["maximum_energy_capacity_ess"]
n_char_b = data["ess_charging_efficiency"]
n_dis_b = data["ess_discharging_efficiency"]
db_ch = data["limits_number_operations_charge_mode"]
db_dis = data["limits_number_operations_discharge_mode"]
t_arrival = data["arrival_time"]
t_departure = data["departure_time"]
n_char_ev = data["ev_charging_efficiency"]
Evi = data["initial_energy_of_the_ev"]
Eve_min = data["minimum_energy_capacity_ev"]
Eve_max = data["maximum_energy_capacity_ev"]
Pve_max = data["maximum_charging_limit_ev"]