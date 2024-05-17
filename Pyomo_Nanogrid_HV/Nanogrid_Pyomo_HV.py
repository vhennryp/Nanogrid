from pyomo.environ import *
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product


#with open('Test_004.json', 'r') as file:
with open('input_data_NANOGRID_15min.json', 'r') as file:
    data = json.load(file)

m = ConcreteModel()

##################################################################################################
# =========== C O N J U N T O S   D E   D A T O S 
##################################################################################################

m.S                 = Set(initialize=data['set_of_scenarios'])  # Escenarios
m.T                 = Set(initialize=data['set_of_time'])  # Tiempos
m.C                  = Set(initialize=data['set_of_outage'])  # Tiempos


print("here")
##################################################################################################
# =========== C O N J U N T O S   D E   P A R Á M E T R O S  
##################################################################################################

# ---- Time 
Δt                  = data['variation_of_time']

# ---- PV 
PV_P                = data['photovoltaic_generation']
S_p                 = data['probability_of_the_scenarios']
cOS_cc              = data['cost_of_the_energy_cc']

# ---- EV       
EV_t_c              = data['arrival_time']
EV_t_p              = data['departure_time']

EV_n_c              = data['ev_charging_efficiency']
EV_E_i              = data['initial_energy_of_the_ev']
EV_E_min            = data["minimum_energy_capacity_ev"]
EV_E_max            = data["maximum_energy_capacity_ev"]
EV_P_max            = data["maximum_charging_limit_ev"]

EV_no_c             = data["limits_number_operations_charge_mode"]
EV_no_d             = data["limits_number_operations_discharge_mode"]

# ---- RED      

T                   = data['set_of_time']                               # Inicialización para D_cOS  y  D_P
D_cOS               = data['cost_of_the_energy'] 
D_cOS               = {str(time): cost for time, cost in zip(T, D_cOS)}
D_P                 = data['active_load']
D_P                 = {str(time): demand for time, demand in zip(T, D_P)}
PAC_P_max           = data['maximum_power_PCC']

# ---- RED OUTAGE

D_cOS_out           = data['cost_of_the_energy_cc']   

# --- BESS

BESS_E_i            = data['initial_energy_of_the_ess']
BESS_P_c_min        = data['minimum_charging_limit_ess']
BESS_P_c_max        = data['maximum_charging_limit_ess']
BESS_P_d_min        = data['minimum_discharging_limit_ess']
BESS_P_d_max        = data['maximum_discharging_limit_ess']
BESS_n_c            = data['ess_charging_efficiency']  
BESS_n_d            = data['ess_discharging_efficiency']  
BESS_E_min          = data['minimum_energy_capacity_ess'] 
BESS_E_max          = data['maximum_energy_capacity_ess'] 

Δ_BESS_num_c        = data['limits_number_operations_charge_mode'] 
Δ_BESS_num_d        = data['limits_number_operations_discharge_mode'] 


##################################################################################################
# =========== C O N J U N T O S   D E    V A R I A B L E S
##################################################################################################

#RED
m.PAC_P             = Var(m.S, m.T, within=Reals)
m.PAC_P_p           = Var(m.S, m.T, within=NonNegativeReals)        # potência demandada da rede
m.PAC_P_n           = Var(m.S, m.T, within=NonNegativeReals)        # potência injetada da rede

# RED con CONTINGENCIA

m.PAC_P_out             = Var(m.S, m.T, m.C,  within=Reals)
m.PAC_P_p_out           = Var(m.S, m.T, m.C, within=NonNegativeReals)        # potência demandada da rede
m.PAC_P_n_out           = Var(m.S, m.T, m.C, within=NonNegativeReals)        # potência injetada da rede

# Corte de Carga
m.bin_cc                = Var(m.S, m.T, m.C,  within=Binary)
m.bin_fv                = Var(m.S, m.T, m.C,  within=Binary)

# m.bin_cc                = Var(m.S, m.T, m.C,  within=NonNegativeReals, bounds=(0, 1))
# m.bin_fv                = Var(m.S, m.T, m.C,  within=NonNegativeReals,bounds=(0, 1))



#BESS
m.BESS_P_c          = Var(m.T, within=NonNegativeReals)             # potência ativa de injeção do armazenador
m.BESS_P_d          = Var(m.T, within=NonNegativeReals)             # potência ativa de extração do armazenadorm.
m.BESS_P            = Var(m.T, within=Reals)             # potência ativa de compra
m.BESS_E            = Var(m.T, within=NonNegativeReals)             # potência ativa de compra
m.BESS_β_c          = Var(m.T, domain=Binary)                       # estado de operação do armazenador   
m.BESS_β_d          = Var(m.T, domain=Binary)                       # horário de operação da carga gerenciável (1 operando, 0 cc)

m.BESS_c_pos        = Var(m.T, within=NonNegativeReals)             # potência ativa de injeção do armazenador
m.BESS_c_neg        = Var(m.T, within=NonNegativeReals)             # potência ativa de injeção do armazenador
m.BESS_d_pos        = Var(m.T, within=NonNegativeReals)             # potência ativa de injeção do armazenador
m.BESS_d_neg        = Var(m.T, within=NonNegativeReals)             # potência ativa de injeção do armazenador

#EV
m.PEV               = Var(m.S, m.T, within=NonNegativeReals)        # potência ativa de extração do veículo elétrico
m.EEV               = Var(m.S, m.T, within=NonNegativeReals)        # energia armazenada no veículo elétrico

#EV
m.EV_P               = Var(m.S, m.T, within=NonNegativeReals)        # potência ativa de extração do veículo elétrico
m.EV_E               = Var(m.S, m.T, within=NonNegativeReals)        # energia armazenada no veículo elétrico

# # EV con CONTINGENCIA
m.EV_P_out           = Var(m.S, m.T, m.C, within=NonNegativeReals)        # potência ativa de extração do veículo elétrico
m.EV_E_out           = Var(m.S, m.T, m.C, within=NonNegativeReals)        # energia armazenada no veículo elétrico

##################################################################################################
# =========== F U N C I O N    O B J E T I V O
##################################################################################################

# Definición de la función objetivo
def objective_rule(m):
    return  sum( Δt * S_p[str(s)] * 0.01 * ((m.PAC_P_p_out[s, t, c] * D_cOS[str(t)]) +  (D_P[str(t)] * 10 *  m.bin_cc [s, t, c]) ) for s in m.S for t in m.T  for c in m.C) + \
            sum( Δt * S_p[str(s)] * 0.99 * (m.PAC_P_p[s, t] * D_cOS[str(t)]) for s in m.S for t in m.T )

# def objective_rule(m):
#     return  sum( S_p[str(s)] * (sum( Δt * 0.01 * ((m.PAC_P_p_out[s, t, c] * D_cOS[str(t)]) +  (D_P[str(t)] * 10 *  m.bin_cc [s, t, c]) ) for t in m.T  for c in m.C) + \
#             sum( Δt * 0.99 * (m.PAC_P_p[s, t] * D_cOS[str(t)]) for t in m.T) ) for s in m.S)



m.objective = Objective(rule=objective_rule, sense=minimize)
m.objective.pprint()

##################################################################################################
# =========== R E S T R I C C I O N E S
##################################################################################################

#-------------------------------------------------------------------------------------------------
# ----------- W I T H O U T     O U T A G E
#-------------------------------------------------------------------------------------------------

# ----- RED
def ec_2_REDE(m, s, t):
    return m.PAC_P[s, t] + PV_P[str(s)][str(t)] +  m.BESS_P_d[t] - m.BESS_P_c[t] - m.EV_P[s, t] == D_P[str(t)]
m.ec_2_REDE = Constraint(m.S, m.T, rule=ec_2_REDE)

# ----- PCCA
def ec_3_PCCA(m, s, t):
    return m.PAC_P[s, t] <= PAC_P_max 
m.ec_3_PCCA = Constraint(m.S, m.T, rule=ec_3_PCCA)

def ec_4_PCCA(m, s, t):
    return m.PAC_P[s, t] == m.PAC_P_p[s, t] - m.PAC_P_n[s, t]
m.ec_4_PCCA = Constraint(m.S, m.T, rule=ec_4_PCCA)

# # # ----- EV

def ec_6_EV(m, s, t):
    if t == EV_t_c[str(s)]:
        return  EV_E_i +  Δt * m.EV_P[s, t] * EV_n_c  == m.EV_E[s, t]  
    else:
        return Constraint.Skip
m.ec_6_EV = Constraint(m.S, m.T, rule=ec_6_EV)

def ec_7_EV(m, s, t):
    if t > EV_t_c[str(s)] and t <= EV_t_p[str(s)]:
        return m.EV_E[s, t] == m.EV_E[s, t-1] +  Δt * m.EV_P[s, t] * EV_n_c
    else:
        return Constraint.Skip
m.ec_7_EV = Constraint(m.S, m.T, rule=ec_7_EV)

def ec_8_EV(m, s, t):
    if t >= EV_t_c[str(s)] and t <= EV_t_p[str(s)]:
        return inequality(EV_E_min, m.EV_E[s, t], EV_E_max)
    else:
        return Constraint.Skip
m.ec_8_EV = Constraint(m.S, m.T, rule=ec_8_EV)

def ec_9_EV(m, s, t):
    if t >= EV_t_c[str(s)] and t <= EV_t_p[str(s)]:
        return inequality(0, m.EV_P[s, t], EV_P_max)
    else:
        return Constraint.Skip
m.ec_9_EV = Constraint(m.S, m.T, rule=ec_9_EV)

def ec_10_EV(m, s, t):
    if t == EV_t_p[str(s)]:
        return m.EV_E[s, t] == EV_E_max 
    else:
        return Constraint.Skip
m.ec_10_EV = Constraint(m.S, m.T, rule=ec_10_EV)

def ec_11_EV(m, s, t):
    if t < EV_t_c[str(s)] or t > EV_t_p[str(s)]:
        return m.EV_P[s, t] == 0
    else:
        return Constraint.Skip
m.ec_11_EV = Constraint(m.S, m.T, rule=ec_11_EV)

# ----- BESS

def ec_12_BESS(m, s, t):
    if t == 1:
        return BESS_E_i  + m.BESS_P[t] * Δt  == m.BESS_E[t] 
    else:
        return Constraint.Skip
m.ec_12_BESS = Constraint(m.S, m.T, rule=ec_12_BESS)

def ec_13_BESS(m, s, t):
    if t > 1:   
        return m.BESS_E[t] == m.BESS_E[t-1] + m.BESS_P[t] * Δt
    else:
        return Constraint.Skip
m.ec_13_BESS = Constraint(m.S, m.T, rule=ec_13_BESS)

def ec_14_BESS(m, s, t):
    return  m.BESS_P[t] == (m.BESS_P_c[t] * BESS_n_c  -  m.BESS_P_d[t]/ BESS_n_d) 
m.ec_14_BESS = Constraint(m.S, m.T, rule=ec_14_BESS)

def ec_15_BESS_lower(m, s, t):
    return BESS_E_min <= m.BESS_E[t]
m.ec_15_BESS_lower = Constraint(m.S, m.T, rule=ec_15_BESS_lower)

def ec_15_BESS_upper(m, s, t):
    return m.BESS_E[t] <= BESS_E_max
m.ec_15_BESS_upper = Constraint(m.S, m.T, rule=ec_15_BESS_upper)

def ec_16_BESS_lower(m, s, t):
    return BESS_P_c_min * m.BESS_β_c[t] <= m.BESS_P_c[t]
m.ec_16_BESS_lower = Constraint(m.S, m.T, rule=ec_16_BESS_lower)

def ec_16_BESS_upper(m, s, t):
    return m.BESS_P_c[t] <= BESS_P_c_max * m.BESS_β_c[t]
m.ec_16_BESS_upper = Constraint(m.S, m.T, rule=ec_16_BESS_upper)

def ec_17_BESS_lower(m, s, t):
    return BESS_P_d_min * m.BESS_β_d[t] <= m.BESS_P_d[t]
m.ec_17_BESS_lower = Constraint(m.S, m.T, rule=ec_17_BESS_lower)

def ec_17_BESS_upper(m, s, t):
    return m.BESS_P_d[t] <= BESS_P_d_max* m.BESS_β_d[t]
m.ec_17_BESS_upper = Constraint(m.S, m.T, rule=ec_17_BESS_upper)

# def ec_18_BESS(m, s, t):
#     return sum(abs(m.BESS_β_c[t] - m.BESS_β_c[t-1]) for t in m.T if t > 1) <= Δ_BESS_num_c
# m.ec_18_BESS = Constraint(m.S, m.T, rule=ec_18_BESS)

# def ec_19_BESS(m, s, t):
#     return sum(abs(m.BESS_β_d[t] - m.BESS_β_d[t-1]) for t in m.T if t > 1) <= Δ_BESS_num_d
# m.ec_19_BESS = Constraint(m.S, m.T, rule=ec_19_BESS)

def ec_20_BESS(m, s, t):
    return inequality(0, m.BESS_β_c[t] + m.BESS_β_d[t], 1)       #m.BESS_β_c[t] + m.BESS_β_d[t] <= 1
m.ec_20_BESS = Constraint(m.S, m.T, rule=ec_20_BESS)

def ec_21_BESS(m, s, t):
    if t > 1: 
        return m.BESS_β_c[t] - m.BESS_β_c[t-1] + m.BESS_c_neg[t] == m.BESS_c_pos[t] 
    else:
        return Constraint.Skip
m.ec_21_BESS = Constraint(m.S, m.T, rule=ec_21_BESS)

def ec_21_2_BESS(m, s, t):
    if t == 1: 
        return m.BESS_β_c[t] + m.BESS_c_neg[t] == m.BESS_c_pos[t] 
    else:
        return Constraint.Skip
m.ec_21_2_BESS = Constraint(m.S, m.T, rule=ec_21_2_BESS)

def ec_22_BESS_pos_lower(m, t):
    return m.BESS_c_pos[t] >= 0
m.ec_22_BESS_pos_lower = Constraint(m.T, rule=ec_22_BESS_pos_lower)

def ec_22_BESS_pos_upper(m, t):
    return m.BESS_c_pos[t] <= 1
m.ec_22_BESS_pos_upper = Constraint(m.T, rule=ec_22_BESS_pos_upper)

def ec_22_BESS_neg_lower(m, t):
    return m.BESS_c_neg[t] >= 0
m.ec_22_BESS_neg_lower = Constraint(m.T, rule=ec_22_BESS_neg_lower)

def ec_22_BESS_neg_upper(m, t):
    return m.BESS_c_neg[t] <= 1
m.ec_22_BESS_neg_upper = Constraint(m.T, rule=ec_22_BESS_neg_upper)

def ec_23_BESS(m, s, t):
    return sum((m.BESS_c_pos[t] + m.BESS_c_neg[t]) for t in m.T if t > 1) <= 2 #Δ_BESS_num_c
m.ec_23_BESS = Constraint(m.S, m.T, rule=ec_23_BESS)

def ec_24_BESS(m, s, t):
    if t > 1: 
        return m.BESS_β_d[t] - m.BESS_β_d[t-1] == m.BESS_d_pos[t] - m.BESS_d_neg[t]
    else:
        return Constraint.Skip
m.ec_24_BESS = Constraint(m.S, m.T, rule=ec_24_BESS)

def ec_24_2_BESS(m, s, t):
    if t == 1: 
        return m.BESS_β_d[t] == m.BESS_d_pos[t] - m.BESS_d_neg[t]
    else:
        return Constraint.Skip
m.ec_24_2_BESS = Constraint(m.S, m.T, rule=ec_24_2_BESS)

def ec_25_BESS_pos_lower(m, t):
    return m.BESS_d_pos[t] >= 0
m.ec_25_BESS_pos_lower = Constraint(m.T, rule=ec_25_BESS_pos_lower)

def ec_25_BESS_pos_upper(m, t):
    return m.BESS_d_pos[t] <= 1
m.ec_25_BESS_pos_upper = Constraint(m.T, rule=ec_25_BESS_pos_upper)

def ec_25_BESS_neg_lower(m, t):
    return m.BESS_d_neg[t] >= 0
m.ec_25_BESS_neg_lower = Constraint(m.T, rule=ec_25_BESS_neg_lower)

def ec_25_BESS_neg_upper(m, t):
    return m.BESS_d_neg[t] <= 1
m.ec_25_BESS_neg_upper = Constraint(m.T, rule=ec_25_BESS_neg_upper)

def ec_26_BESS(m, s, t):
    return sum((m.BESS_d_pos[t] + m.BESS_d_neg[t]) for t in m.T if t > 1) <= 2 #Δ_BESS_num_d
m.ec_26_BESS = Constraint(m.S, m.T, rule=ec_26_BESS)

#-------------------------------------------------------------------------------------------------
# ----------- W I T H     O U T A G E
#-------------------------------------------------------------------------------------------------

def ec_5_PAC_out(m, s, t, c):
    if t >= c and t < c + 4:
        return m.PAC_P_out[s, t, c] == 0
    else:
        return Constraint.Skip
m.ec_5_PAC_out = Constraint(m.S, m.T, m.C, rule=ec_5_PAC_out)

def ec_5_1_PAC_out(m, s, t, c):
    if t < c or t >= c + 4:
        return m.bin_fv[s, t, c] == 0
    else:
        return Constraint.Skip
m.ec_5_1_PAC_out = Constraint(m.S, m.T, m.C, rule=ec_5_1_PAC_out)

def ec_5_2_PAC_out(m, s, t, c):
    if t < c or t >= c + 4:
        return m.bin_cc[s, t, c] == 0
    else:
        return Constraint.Skip
m.ec_5_2_PAC_out = Constraint(m.S, m.T, m.C, rule=ec_5_2_PAC_out)

#------------- REDE

# ----- RED
def ec_2_REDE_out(m, s, t, c):
    return m.PAC_P_out[s, t, c]  + PV_P[str(s)][str(t)] * (1 - m.bin_fv[s, t, c]) +  m.BESS_P_d[t] - m.BESS_P_c[t] - m.EV_P_out[s, t, c] == D_P[str(t)] * (1 - m.bin_cc[s, t, c]) 
m.ec_2_REDE_out = Constraint(m.S, m.T, m.C, rule=ec_2_REDE_out)

# ----- PCCA

def ec_3_PCCA_out(m, s, t, c):
    return m.PAC_P_out[s, t, c]  <= PAC_P_max 
m.ec_3_PCCA_out = Constraint(m.S, m.T, m.C, rule=ec_3_PCCA_out)

def ec_4_PCCA_out(m, s, t, c):
    return m.PAC_P_out[s, t, c]  == m.PAC_P_p_out[s, t, c]  - m.PAC_P_n_out[s, t, c] 
m.ec_4_PCCA_out = Constraint(m.S, m.T, m.C, rule=ec_4_PCCA_out)

# # # ----- EV

def ec_6_EV_out(m, s, t, c):
    if t == EV_t_c[str(s)]:
        return  m.EV_E_out[s, t, c] == EV_E_i +  Δt * m.EV_P_out[s, t, c] * EV_n_c  
    else:
        return Constraint.Skip
m.ec_6_EV_out = Constraint(m.S, m.T, m.C, rule=ec_6_EV_out)

def ec_7_EV_out(m, s, t, c):
    if t > EV_t_c[str(s)] and t <= EV_t_p[str(s)]:
        return m.EV_E_out[s, t, c] == m.EV_E_out[s, t-1, c] +  Δt * m.EV_P_out[s, t, c] * EV_n_c
    else:
        return Constraint.Skip
m.ec_7_EV_out = Constraint(m.S, m.T, m.C, rule=ec_7_EV_out)

def ec_8_EV_out(m, s, t, c):
    if t >= EV_t_c[str(s)] and t <= EV_t_p[str(s)]:
        return inequality(EV_E_min, m.EV_E_out[s, t, c], EV_E_max)
    else:
        return Constraint.Skip
m.ec_8_EV_out = Constraint(m.S, m.T, m.C, rule=ec_8_EV_out)

def ec_9_EV_out(m, s, t, c):
    if t >= EV_t_c[str(s)] and t <= EV_t_p[str(s)]:
        return inequality(0, m.EV_P_out[s, t, c], EV_P_max)
    else:
        return Constraint.Skip
m.ec_9_EV_out = Constraint(m.S, m.T, m.C, rule=ec_9_EV_out)

def ec_10_EV_out(m, s, t, c):
    if t == EV_t_p[str(s)]:
        return m.EV_E_out[s, t, c] == EV_E_max 
    else:
        return Constraint.Skip
m.ec_10_EV_out = Constraint(m.S, m.T, m.C, rule=ec_10_EV_out)

def ec_11_EV_out(m, s, t, c):
    if t < EV_t_c[str(s)] or t > EV_t_p[str(s)]:
        return m.EV_P_out[s, t, c] == 0
    else:
        return Constraint.Skip
m.ec_11_EV_out = Constraint(m.S, m.T, m.C, rule=ec_11_EV_out)

##################################################################################################
# =========== S O L V E R
##################################################################################################

# Resolver el modelo
solver = SolverFactory('gurobi')
results = solver.solve(m, tee=True)
# m.display()

##################################################################################################
# =========== I M P R E S I O N     D E    R E S U L T A D O S
##################################################################################################

index = pd.MultiIndex.from_product([m.S, m.T], names=['Scenario', 'Time'])
results_df = pd.DataFrame(index=index)

# Llenar el DataFrame con los datos
for s in m.S:
    for t in m.T:
        
        results_df.at[(s, t), 'PAC_P_p']        = m.PAC_P_p[s, t].value
        results_df.at[(s, t), 'PAC_P_n']        = m.PAC_P_n[s, t].value
        results_df.at[(s, t), 'PAC_P']          = m.PAC_P[s, t].value
        results_df.at[(s, t), 'PV']             = PV_P[str(s)][str(t)] 
        results_df.at[(s, t), 'BESS_E[t]']      = m.BESS_E[t] .value
        results_df.at[(s, t), 'BESS_β_d']       = m.BESS_β_d[t].value
        results_df.at[(s, t), 'BESS_d_pos']     = m.BESS_d_pos[t].value
        results_df.at[(s, t), 'BESS_d_neg']     = m.BESS_d_neg[t].value
        results_df.at[(s, t), 'BESS_P_d']       = m.BESS_P_d[t].value
        results_df.at[(s, t), 'BESS_β_c']       = m.BESS_β_c[t].value
        results_df.at[(s, t), 'BESS_c_pos']     = m.BESS_c_pos[t].value
        results_df.at[(s, t), 'BESS_c_neg']     = m.BESS_c_neg[t].value
        results_df.at[(s, t), 'BESS_P_c']       = m.BESS_P_c[t].value
        
        
        results_df.at[(s, t), 'BESS_P']         = m.BESS_P[t].value
        results_df.at[(s, t), 'EV_P']           = m.EV_P[s, t].value
        results_df.at[(s, t), 'EV_E']           = m.EV_E[s, t].value
        results_df.at[(s, t), 'PD'] = D_P[str(t)]
     
        # Calcular y almacenar el costo contribuido por cada término
        cost_contrib = D_cOS[str(t)] * Δt* m.PAC_P_p[s, t].value * S_p[str(s)] 
        results_df.at[(s, t), 'Cost Contribution'] = cost_contrib

# Configurar pandas para mostrar todas las filas y columnas si lo deseas
pd.set_option('display.max_rows', None)  # None significa sin límite en la cantidad de filas
pd.set_option('display.max_columns', None)  # None significa sin límite en la cantidad de columnas
pd.set_option('display.width', None)  # None ajusta automáticamente el ancho para evitar el recorte de la visualización
pd.set_option('display.max_colwidth', None)  # Elimina la limitación en la longitud de la visualización del contenido de cada celda

# Ahora, al imprimir el DataFrame, debería mostrarse completamente
#print(results_df.head(96))
#print(results_df)

# Inicializamos el DataFrame con un MultiIndex que incluye el nuevo conjunto 'C'
index = pd.MultiIndex.from_product([m.S, m.T, m.C], names=['Scenario', 'Time', 'C'])
results_1_df = pd.DataFrame(index=index)

# Llenamos el DataFrame con los datos
for s in m.S:
    for t in m.T:
        for c in m.C:  # Aunque solo tiene un elemento, todavía debemos iterar sobre él
            
            results_1_df.at[(s, t, c), 'PAC_P_p_out']    = m.PAC_P_p_out[s, t, c].value
            results_1_df.at[(s, t, c), 'PAC_P_n_out']    = m.PAC_P_n_out[s, t, c].value
            results_1_df.at[(s, t, c), 'PAC_P_out']      = m.PAC_P_out[s, t, c].value
            results_1_df.at[(s, t, c), 'PV']             = PV_P[str(s)][str(t)]
            results_1_df.at[(s, t, c), 'bin_fv']         = m.bin_fv[s, t, c].value 
            results_1_df.at[(s, t, c), 'BESS_P_d']       = m.BESS_P_d[t].value
            results_1_df.at[(s, t, c), 'BESS_P_d']       = m.BESS_P_d[t].value
            results_1_df.at[(s, t, c), 'BESS_P']         = m.BESS_P[t].value
            results_1_df.at[(s, t, c), 'EV_P_out']       = m.EV_P_out[s, t, c].value
            results_1_df.at[(s, t, c), 'EV_E_out']       = m.EV_E_out[s, t, c].value
            results_1_df.at[(s, t, c), 'PD']             = D_P[str(t)]   
            results_1_df.at[(s, t, c), 'bin_cc']         = m.bin_fv[s, t, c].value 

######################################################################
# ========================= SUMMARY RESULTS  ==========================
######################################################################

print("here")

# --------------------- Plotting Results ---------------------------------
# Creating the directory for the folders with the figures
script_dir = os.path.dirname(os.path.abspath(__file__))
print("\n************* START: Plotting final results  *************\n", flush=True)
print("Creating Figures...", flush=True)
# Creating a list with the number of hours only for graphics purposes!
hours = list(map(int, m.T))
folder = "Figures_Scenarios" + "/"
figures_dir = os.path.join('Figures/', folder)
if not os.path.isdir(figures_dir):
	os.makedirs(figures_dir)

for s in m.S:     
    aux_list_1 = [(m.PAC_P[s, t].value, PV_P[str(s)][str(t)], D_P[str(t)], (-(m.BESS_P_d[t]).value + (m.BESS_P_c[t]).value ), m.EV_P[s, t].value) for t in m.T]
    aux_list_1 = np.array(aux_list_1)
    plt.figure(s)
    plt.style.use('ggplot')
    plt.plot(range(len(m.T)),aux_list_1[:,0],color='red', linewidth=1.5)
    plt.fill_between(range(len(m.T)), aux_list_1[:,0], color='red', alpha=0.4)
    plt.plot(range(len(m.T)),aux_list_1[:,1],color="darkorange", linewidth=1.5)
    plt.fill_between(range(len(m.T)), aux_list_1[:,1], color='darkorange', alpha=0.4)
    plt.plot(range(len(m.T)),aux_list_1[:,2],color="black", linewidth=1.5)
    plt.fill_between(range(len(m.T)), aux_list_1[:,2], color='black', alpha=0.4)
    plt.plot(range(len(m.T)),aux_list_1[:,3],color="darkgreen", linewidth=1.5)
    plt.fill_between(range(len(m.T)), aux_list_1[:,3], color='darkgreen', alpha=0.4)
    plt.plot(range(len(m.T)),aux_list_1[:,4],color="blue", linewidth=1.0)
    plt.fill_between(range(len(m.T)), aux_list_1[:,4], color='blue', alpha=0.4)
    plt.xlabel('Tempo [h]', color= 'black', fontsize=14)
    plt.ylabel('Potência ativa [kW]', color= 'black', fontsize=14)
    plt.tick_params(axis='y', colors='black')
    plt.tick_params(axis='x', colors='black')
    plt.ylim(-60, 60) # definir limite do eixo
    plt.xticks([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88], ['00', '02', '04', '06', '08', '10', '12', '14', '16', '18', '20', '22'],fontsize=14)
    plt.title("Cenário "+str(s) + " sem contingência")
    plt.yticks(fontsize=14)
    plt.savefig(figures_dir +"Fig_S" + str(s) + ".pdf")
    aux_list_1 = []

    for c in m.C:
        aux_list_1 = [(m.PAC_P_out[s, t, c].value, PV_P[str(s)][str(t)] * (1 - m.bin_fv[s, t, c].value), D_P[str(t)] * (1 - m.bin_cc[s, t, c].value), (-(m.BESS_P_d[t]).value + (m.BESS_P_c[t]).value ), m.EV_P_out[s, t, c].value) for t in m.T]
        aux_list_1 = np.array(aux_list_1)
        plt.figure(c+s)
        plt.style.use('ggplot')
        plt.plot(range(len(m.T)),aux_list_1[:,0],color='red', linewidth=1.5)

        plt.plot(range(len(m.T)),aux_list_1[:,0],color='red', linewidth=1.5)
        plt.fill_between(range(len(m.T)), aux_list_1[:,0], color='red', alpha=0.4)
        #plt.bar(range(len(data.T)),aux_list_1[:,1],color="darkorange", label="FV")
        plt.plot(range(len(m.T)),aux_list_1[:,1],color="darkorange", linewidth=1.5)
        plt.fill_between(range(len(m.T)), aux_list_1[:,1], color='darkorange', alpha=0.4)
        plt.plot(range(len(m.T)),aux_list_1[:,2],color="black", linewidth=1.5)
        plt.fill_between(range(len(m.T)), aux_list_1[:,2], color='black', alpha=0.4)
        #plt.bar(range(len(data.T)),aux_list_1[:,3],color="darkgreen", label="BESS")
        plt.plot(range(len(m.T)),aux_list_1[:,3],color="darkgreen", linewidth=1.5)
        plt.fill_between(range(len(m.T)), aux_list_1[:,3], color='darkgreen', alpha=0.4)
        #plt.bar(range(len(data.T)),aux_list_1[:,4],color="red", label="VE")
        plt.plot(range(len(m.T)),aux_list_1[:,4],color="blue", linewidth=1.0)
        plt.fill_between(range(len(m.T)), aux_list_1[:,4], color='blue', alpha=0.4)
        plt.xlabel('Tempo [h]', color= 'black', fontsize=14)
        plt.ylabel('Potência ativa [kW]', color= 'black', fontsize=14)
        plt.tick_params(axis='y', colors='black')
        plt.tick_params(axis='x', colors='black')
        #plt.xticks(np.arange(min(hours)-1, max(hours), 4.0),range(0,100,2)) #96 periodos de tempo
        plt.xticks([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88], ['00', '02', '04', '06', '08', '10', '12', '14', '16', '18', '20', '22'],fontsize=14)
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



if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
    print("Optimal solution found")
    print(value(m.objective))

