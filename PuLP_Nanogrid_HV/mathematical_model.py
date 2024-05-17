""""
========================== Mathematical Modelling ==============================
"""
import sys
from pulp import *
import math
import numpy as np
import matplotlib.pyplot as plt
import input_data_processing as data
from pulp import GUROBI

List_ST = []
for s in data.S:
	for t in data.T:
		List_ST += [[s, t]]

List_STO = []
for s in data.S:
	for t in data.T:
		for c in data.O:
			List_STO += [[s, t, c]]

# Type of problem
prob = LpProblem("EMS_for_Nanogrid", LpMinimize)

# Declare variables
PS = LpVariable.dicts("PS", (data.S, data.T))
PS_p = LpVariable.dicts("PS_p", (data.S, data.T), 0)
PS_n = LpVariable.dicts("PS_n", (data.S, data.T), 0)

PS_out = LpVariable.dicts("PS_out", (data.S, data.T, data.O))
PS_p_out = LpVariable.dicts("PS_p_out", (data.S, data.T, data.O), 0)
PS_n_out = LpVariable.dicts("PS_n_out", (data.S, data.T, data.O), 0)
cc = LpVariable.dicts("cc",(data.S, data.T, data.O),cat='Binary')
fv = LpVariable.dicts("fv",(data.S, data.T, data.O),cat='Binary')

# Energy Storage System - Battery
b_ch = LpVariable.dicts("b_ch", (data.T), cat='Binary')
b_dis = LpVariable.dicts("b_dis", (data.T), cat='Binary')
Pch_bat = LpVariable.dicts("Pch_bat", (data.T))
Pdis_bat = LpVariable.dicts("Pdis_bat", (data.T))
Eb = LpVariable.dicts("Eb", (data.T))
b_ch_aux_1 = LpVariable.dicts("b_ch_aux_1", (data.T), 0, 1)
b_ch_aux_2 = LpVariable.dicts("b_ch_aux_2", (data.T), 0, 1)
b_dis_aux_1 = LpVariable.dicts("b_dis_aux_1", (data.T), 0, 1)
b_dis_aux_2 = LpVariable.dicts("b_dis_aux_2", (data.T), 0, 1)

# Electric Vehicles
Ev = LpVariable.dicts("Ev", (data.S, data.T))
Pch_ve = LpVariable.dicts("Pch_ve", (data.S, data.T),0)

Ev_out = LpVariable.dicts("Ev_out", (data.S, data.T, data.O))
Pch_ve_out = LpVariable.dicts("Pch_ve_out", (data.S, data.T, data.O),0)


# NOVA DEFINIÇAO DE FUNÇAO OBJETIVO

# Objective function
lpSum([])
prob += \
	lpSum([data.delta_t * data.Prob[(s)] * 0.01 * (PS_p_out[s][t][c] * data.cEDS[(t)]   +  data.PD[(t)] * 10 * cc[s][t][c]) for (s,t,c) in List_STO]) + \
 	lpSum([data.delta_t * data.Prob[(s)] * 0.99 *  PS_p[s][t]        * data.cEDS[(t)]    for (s, t) in List_ST]), "Objective_Function" #10 Euilibrar unidad de demanda

# lpSum([])
# prob += \
# 	lpSum([data.Prob[(s)] * ((0.01/len(data.O) * (lpSum([data.cEDS[(t)] * data.delta_t * PS_p_out[s][t][c] for (s,t,c) in List_STO]) + \
#  	lpSum([data.delta_t * 10 * data.PD[(t)] * cc[s][t][c] for (s,t,c) in List_STO]))) + \
#  	(0.99 * lpSum([data.cEDS[(t)] * data.delta_t * PS_p[s][t] for (s, t) in List_ST]))) for s in data.S]), "Objective_Function" #10 Euilibrar unidad de demanda

# --------------------- Constraints --------------------------------------
# --------------------- Without outage------------------------------------
for (s,t) in List_ST:
	prob += PS[s][t] + data.PV[s][t] + Pdis_bat[t] - Pch_bat[t] - Pch_ve[s][t] - data.PD[(t)] == 0, "Active_Power_Balance_Equation_%s" % str((s,t))

for (s, t) in List_ST:
	prob += PS[s][t] <= data.Smax, "Maximum_powew_PCC_%s" % str((s,t))

for (s, t) in List_ST:
	prob += PS[s][t] - PS_p[s][t] + PS_n[s][t] == 0, "PCC_power_%s" % str((s,t))

# ----------------- BESS ---------------------------
for t in data.T:
	prob += data.Pe_min * b_ch[t] <= Pch_bat[t], "ESS_charge_lim_1_%s" % str((t))

for t in data.T:
	prob += Pch_bat[t] <= data.Pe_max * b_ch[t], "ESS_charge_lim_2_%s" % str((t))

for t in data.T:
	prob += data.Pi_min * b_dis[t] <= Pdis_bat[t], "ESS_discharge_lim_1_%s" % str((t))

for t in data.T:
	prob += Pdis_bat[t] <= data.Pi_max * b_dis[t], "ESS_discharge_lim_2_%s" % str((t))

prob += lpSum([b_ch_aux_1[t] + b_ch_aux_2[t] for t in data.T]) <= data.db_ch, "ESS_variation_limit_ch"

for t in data.T:
	if int(t) == 1:
		prob += b_ch[t] - b_ch_aux_1[t] + b_ch_aux_2[t] == 0, "ESS_state_variation_ch_1_%s" % str((t))
	else:
		prob += b_ch[t] - b_ch[str(int(t)-1)] - b_ch_aux_1[t] + b_ch_aux_2[t] == 0, "ESS_state_variation_ch_2_%s" % str((t))

prob += lpSum([b_dis_aux_1[t] + b_dis_aux_2[t] for t in data.T]) <= data.db_dis, "ESS_variation_limit_dis_%s"

for t in data.T:
	if int(t) == 1:
		prob += b_dis[t] - b_dis_aux_1[t] + b_dis_aux_2[t] == 0, "ESS_state_variation_dis_1_%s" % str((t))
	else:
		prob += b_dis[t] - b_dis[str(int(t)-1)] - b_dis_aux_1[t] + b_dis_aux_2[t] == 0, "ESS_state_variation_dis_2_%s" % str((t))

for t in data.T:
	prob += 0 <= b_ch[t] + b_dis[t], "ESS_Operation_Mode_1_%s" % str((t))

for t in data.T:
	prob += b_ch[t] + b_dis[t] <= 1, "ESS_Operation_Mode_2_%s" % str((t))

for t in data.T:
	if int(t) == 1:
		prob += Eb['1'] - (Pch_bat[t] * data.n_char_b * data.delta_t) + ((Pdis_bat[t] * data.delta_t)/data.n_dis_b) == data.Ebi, "State_of_Charge_1_%s" % str((t))
	else:
		prob += Eb[t] - Eb[str(int(t)-1)] - (Pch_bat[t] * data.n_char_b * data.delta_t) + ((Pdis_bat[t] * data.delta_t)/data.n_dis_b) == 0, "State_of_Charge_2_%s" % str((t))

for t in data.T:
	prob += data.Eb_min <= Eb[t], "ESS_limits_1_%s" % str((t))

for t in data.T:
	prob += Eb[t] <= data.Eb_max, "ESS_limits_2_%s" % str((t))

# ----------------- Electric Vehicles ---------------------------
for (s,t) in List_ST:
	if t == str(data.t_arrival[s]):
		prob += Ev[s][str(data.t_arrival[s])] - (Pch_ve[s][str(data.t_arrival[s])] * data.n_char_ev * data.delta_t) == data.Evi, "State_of_Charge_VE_1_%s" % str((s,t))

for (s,t) in List_ST:
	if int(t) > data.t_arrival[s] and int(t) <= data.t_departure[s]:
		prob += Ev[s][t] - Ev[s][str(int(t)-1)] - (Pch_ve[s][t] * data.n_char_ev * data.delta_t) == 0, "State_of_Charge_VE_2_%s" % str((s,t))

for (s,t) in List_ST:
	if int(t) >= data.t_arrival[s] and int(t) <= data.t_departure[s]:
		prob += data.Eve_min <= Ev[s][t], "VE_limits_1_%s" % str((s,t))

for (s,t) in List_ST:
	if int(t) >= data.t_arrival[s] and int(t) <= data.t_departure[s]:
		prob += Ev[s][t] <= data.Eve_max, "VE_limits_2_%s" % str((s,t))

for (s, t) in List_ST:
	if int(t) >= data.t_arrival[s] and int(t) <= data.t_departure[s]:
		prob += Pch_ve[s][t] <= data.Pve_max, "VE_charge_lim_%s" % str((s,t))

# t == t_departure
for (s,t) in List_ST:
	if t == str(data.t_departure[s]):
		prob += Ev[s][str(data.t_departure[s])] == data.Eve_max, "Require_Energy_VE_%s" % str((s,t))

for (s,t) in List_ST:
	if int(t) < data.t_arrival[s] or int(t) > data.t_departure[s]:
		prob += Pch_ve[s][t] == 0, "Out_off_window_VE_%s" % str((s,t))

# ---------------------- Island Operation ---------------------------------------------
for (s,t,c) in List_STO:
	if int(t) >= int(c) and int(t) < int(c) + 4:
		prob += PS_out[s][t][c] == 0, "Contingency_constraint1_%s" %str((s,t,c))


#------------- Operation with outage -------------------------------------------
for (s,t,c) in List_STO:
	prob += PS_out[s][t][c] + (data.PV[s][t] * (1 - fv[s][t][c])) + Pdis_bat[t] - Pch_bat[t] - Pch_ve_out[s][t][c] - (data.PD[(t)] * (1 - cc[s][t][c])) == 0, "Active_Power_Balance_Equation_with_Outage_%s" % str((s,t,c))

for (s,t,c) in List_STO:
	prob += PS_out[s][t][c] - PS_p_out[s][t][c] + PS_n_out[s][t][c] == 0, "PCC_power_with_Outage_%s" % str((s,t,c))

# ----------------- Electric Vehicles with Outage ---------------------------
for (s,t,c) in List_STO:
	if t == str(data.t_arrival[s]):
		prob += Ev_out[s][str(data.t_arrival[s])][c] - (Pch_ve_out[s][str(data.t_arrival[s])][c] * data.n_char_ev * data.delta_t) == data.Evi, "State_of_Charge_VE_1_with_outage_%s" % str((s,t,c))

for (s,t,c) in List_STO:
	if int(t) > data.t_arrival[s] and int(t) <= data.t_departure[s]:
		prob += Ev_out[s][t] - Ev_out[s][str(int(t)-1)][c] - (Pch_ve_out[s][t][c] * data.n_char_ev * data.delta_t) == 0, "State_of_Charge_VE_2_with_Outage_%s" % str((s,t,c))

for (s,t,c) in List_STO:
	if int(t) >= data.t_arrival[s] and int(t) <= data.t_departure[s]:
		prob += data.Eve_min <= Ev_out[s][t][c], "VE_limits_1_with_Outage_%s" % str((s,t,c))

for (s,t,c) in List_STO:
	if int(t) >= data.t_arrival[s] and int(t) <= data.t_departure[s]:
		prob += Ev_out[s][t][c] <= data.Eve_max, "VE_limits_2_with_Outage_%s" % str((s,t,c))

for (s,t,c) in List_STO:
	if int(t) >= data.t_arrival[s] and int(t) <= data.t_departure[s]:
		prob += Pch_ve_out[s][t][c] <= data.Pve_max, "VE_charge_lim_with_Outage_%s" % str((s,t,c))

# t == t_departure
for (s,t,c) in List_STO:
	if t == str(data.t_departure[s]):
		prob += Ev_out[s][str(data.t_departure[s])][c] == data.Eve_max, "Require_Energy_VE_with_Outage_%s" % str((s,t,c))

for (s,t,c) in List_STO:
	if int(t) < data.t_arrival[s] or int(t) > data.t_departure[s]:
		prob += Pch_ve_out[s][t][c] == 0, "Out_off_window_VE_with_Outage_%s" % str((s,t,c))

# --------------------------------------------------------------
# The problem data is written to an .lp file
prob.writeLP("EMS_for_Nanogrid.lp")

print("\n************* START: Solving mathematical model  *************\n", flush=True)

# Solve the model
status = prob.solve()
#prob.solve(GUROBI(msg=True))

print("\n************* END: Solving mathematical model  *************\n", flush=True)

######################################################################
# ========================= SUMMARY RESULTS  ==========================
######################################################################

# --------------------- Plotting Results ---------------------------------
# Creating the directory for the folders with the figures
script_dir = os.path.dirname(os.path.abspath(__file__))
print("\n************* START: Plotting final results  *************\n", flush=True)
print("Creating Figures...", flush=True)
# Creating a list with the number of hours only for graphics purposes!
hours = list(map(int, data.T))
folder = "Figures_Scenarios" + "/"
figures_dir = os.path.join('Figures/', folder)
if not os.path.isdir(figures_dir):
	os.makedirs(figures_dir)
# Creating figures per contingency and per scenario.
for s in data.S:
	aux_list_1 = [(PS[s][t].value(), data.PV[s][t], data.PD[(t)], ((-Pdis_bat[t]).value()+(Pch_bat[t]).value()), Pch_ve[s][t].value()) for t in data.T]
	aux_list_1 = np.array(aux_list_1)
	plt.figure(s)
	plt.style.use('ggplot')
	plt.plot(range(len(data.T)),aux_list_1[:,0],color='red', linewidth=1.5)
	plt.fill_between(range(len(data.T)), aux_list_1[:,0], color='red', alpha=0.4)
	plt.plot(range(len(data.T)),aux_list_1[:,1],color="darkorange", linewidth=1.5)
	plt.fill_between(range(len(data.T)), aux_list_1[:,1], color='darkorange', alpha=0.4)
	plt.plot(range(len(data.T)),aux_list_1[:,2],color="black", linewidth=1.5)
	plt.fill_between(range(len(data.T)), aux_list_1[:,2], color='black', alpha=0.4)
	plt.plot(range(len(data.T)),aux_list_1[:,3],color="darkgreen", linewidth=1.5)
	plt.fill_between(range(len(data.T)), aux_list_1[:,3], color='darkgreen', alpha=0.4)
	plt.plot(range(len(data.T)),aux_list_1[:,4],color="blue", linewidth=1.0)
	plt.fill_between(range(len(data.T)), aux_list_1[:,4], color='blue', alpha=0.4)
	plt.xlabel('Tempo [h]', color= 'black', fontsize=14)
	plt.ylabel('Potência ativa [kW]', color= 'black', fontsize=14)
	plt.tick_params(axis='y', colors='black')
	plt.tick_params(axis='x', colors='black')
	plt.ylim(-60, 60) # definir limite do eixo
	#plt.xticks(np.arange(min(hours)-1, max(hours), 4.0),range(0,100,2)) #96 periodos de tempo
	plt.xticks([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88], ['00', '02', '04', '06', '08', '10', '12', '14', '16', '18', '20', '22'],fontsize=14)
	plt.title("Cenário "+str(s) + " sem contingência")
	plt.yticks(fontsize=14)
	#plt.yticks([0, 500, 1000, 1500, 2000, 2500])
	#plt.xticks(np.arange(min(hours)-1, max(hours), 2),range(0,48,2))
	#plt.show()
	# Saving figures in a specific folder in the current path
	plt.savefig(figures_dir +"Fig_S" + str(s) + ".pdf")
	# Cleaning aux_list to move to the next scenario
	aux_list_1 = []
	
	for c in data.O:
		aux_list_1 = [(PS_out[s][t][c].value(), (data.PV[s][t] * (1 - cc[s][t][c].value())), data.PD[(t)] * (1 - cc[s][t][c].value()), ((-Pdis_bat[t]).value()+(Pch_bat[t]).value()), Pch_ve_out[s][t][c].value()) for t in data.T]
		aux_list_1 = np.array(aux_list_1)
		plt.figure(c+s)
		plt.style.use('ggplot')
		#plt.bar(range(len(data.T)),aux_list_1[:,0],color='blue',label="PAC",bottom=aux_list_1[:,2])
		plt.plot(range(len(data.T)),aux_list_1[:,0],color='red', linewidth=1.5)
		plt.fill_between(range(len(data.T)), aux_list_1[:,0], color='red', alpha=0.4)
		#plt.bar(range(len(data.T)),aux_list_1[:,1],color="darkorange", label="FV")
		plt.plot(range(len(data.T)),aux_list_1[:,1],color="darkorange", linewidth=1.5)
		plt.fill_between(range(len(data.T)), aux_list_1[:,1], color='darkorange', alpha=0.4)
		plt.plot(range(len(data.T)),aux_list_1[:,2],color="black", linewidth=1.5)
		plt.fill_between(range(len(data.T)), aux_list_1[:,2], color='black', alpha=0.4)
		#plt.bar(range(len(data.T)),aux_list_1[:,3],color="darkgreen", label="BESS")
		plt.plot(range(len(data.T)),aux_list_1[:,3],color="darkgreen", linewidth=1.5)
		plt.fill_between(range(len(data.T)), aux_list_1[:,3], color='darkgreen', alpha=0.4)
		#plt.bar(range(len(data.T)),aux_list_1[:,4],color="red", label="VE")
		plt.plot(range(len(data.T)),aux_list_1[:,4],color="blue", linewidth=1.0)
		plt.fill_between(range(len(data.T)), aux_list_1[:,4], color='blue', alpha=0.4)
		plt.xlabel('Tempo [h]', color= 'black', fontsize=14)
		plt.ylabel('Potência ativa [kW]', color= 'black', fontsize=14)
		plt.tick_params(axis='y', colors='black')
		plt.tick_params(axis='x', colors='black')
		#plt.xticks(np.arange(min(hours)-1, max(hours), 4.0),range(0,100,2)) #96 periodos de tempo
		plt.xticks([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88], ['00', '02', '04', '06', '08', '10', '12', '14', '16', '18', '20', '22'], fontsize=14)
		plt.ylim(-60, 60) # definir limite do eixo
		plt.yticks(fontsize=14)
		#plt.xticks([0, 12, 24, 36, 48, 60, 72, 84], ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'])
		#plt.yticks([0, 500, 1000, 1500, 2000, 2500])
		#plt.xticks(np.arange(min(hours)-1, max(hours), 2),range(0,48,2))
		#plt.show()
		plt.title("Cenário "+str(s)+ " com contingência")
		plt.savefig(figures_dir +"Fig_S" + str(s)+"_C" + str(int(c)) + ".pdf")
		# Cleaning aux_list to move to the next scenario
		aux_list_1 = []
		plt.close(plt.figure(s))
print("\n************* END: Plotting final results  *************\n", flush=True)

# Checking the status of the solution
if LpStatus[prob.status] == 'Optimal':
    # If the solution is optimal, print the value of the objective function
    print("The optimal value of the objective function is:", value(prob.objective))
else:
    print("An optimal solution was not found.")


sys.exit()
