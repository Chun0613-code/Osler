% Human-owned, grounded Prolog rules for the Osler DKA embodied loop.
% JEPA may consume these rules as supervision but may not edit this file.

blocked(insulin_iv, critical_hypokalemia) :- requested(insulin_iv), critical_hypokalemia.
blocked(insulin_rapid_sc, critical_hypokalemia) :- requested(insulin_rapid_sc), critical_hypokalemia.
blocked(insulin_intermediate_sc, critical_hypokalemia) :- requested(insulin_intermediate_sc), critical_hypokalemia.
blocked(insulin_basal_sc, critical_hypokalemia) :- requested(insulin_basal_sc), critical_hypokalemia.
blocked(kcl, hyperkalemia) :- requested(kcl), hyperkalemia.
blocked(kcl, impaired_renal_clearance) :- requested(kcl), renal_dysfunction, potassium_not_low.
blocked(kcl, oliguria) :- requested(kcl), oliguria, potassium_not_low.

blocked_action(insulin_iv) :- blocked(insulin_iv, critical_hypokalemia).
blocked_action(insulin_rapid_sc) :- blocked(insulin_rapid_sc, critical_hypokalemia).
blocked_action(insulin_intermediate_sc) :- blocked(insulin_intermediate_sc, critical_hypokalemia).
blocked_action(insulin_basal_sc) :- blocked(insulin_basal_sc, critical_hypokalemia).
blocked_action(kcl) :- blocked(kcl, hyperkalemia).
blocked_action(kcl) :- blocked(kcl, impaired_renal_clearance).
blocked_action(kcl) :- blocked(kcl, oliguria).

allowed(insulin_iv) :- requested(insulin_iv), not(blocked_action(insulin_iv)).
allowed(insulin_rapid_sc) :- requested(insulin_rapid_sc), not(blocked_action(insulin_rapid_sc)).
allowed(insulin_intermediate_sc) :- requested(insulin_intermediate_sc), not(blocked_action(insulin_intermediate_sc)).
allowed(insulin_basal_sc) :- requested(insulin_basal_sc), not(blocked_action(insulin_basal_sc)).
allowed(fluids) :- requested(fluids).
allowed(kcl) :- requested(kcl), not(blocked_action(kcl)).
allowed(bicarbonate) :- requested(bicarbonate).
allowed(dextrose) :- requested(dextrose).

required(kcl, low_potassium_with_insulin) :- insulin_requested, low_potassium, not(critical_hypokalemia), not(requested(kcl)).
required(kcl, depleted_total_body_store) :- insulin_requested, total_body_potassium_depletion, not(requested(kcl)), not(renal_dysfunction), not(oliguria).
required(dextrose, unresolved_ketoacidosis_below_glucose_target) :- insulin_requested, ketoacidosis_unresolved, not(hyperglycemia), not(requested(dextrose)).
required(fluids, severe_hypotension) :- severe_hypotension, not(requested(fluids)).
required(fluids, hyperosmolar_injury) :- high_hyperosmolar_injury_burden, not(requested(fluids)).

expected(insulin_iv, g, decrease, iv_insulin_lowers_glucose) :- allowed(insulin_iv), not(requested(dextrose)).
expected(insulin_rapid_sc, g, decrease, rapid_sc_insulin_lowers_glucose) :- allowed(insulin_rapid_sc), not(requested(dextrose)).
expected(insulin_iv, ke, decrease, iv_insulin_lowers_potassium_without_kcl) :- allowed(insulin_iv), not(requested(kcl)).
expected(fluids, map, increase, fluids_raise_map) :- allowed(fluids).
expected(fluids, v, increase, fluids_raise_volume) :- allowed(fluids).
expected(kcl, ke, increase, kcl_raises_potassium_without_iv_insulin) :- allowed(kcl), not(insulin_requested).
expected(kcl, k_store, increase, kcl_raises_total_body_store) :- allowed(kcl).
expected(bicarbonate, hco3, increase, bicarbonate_raises_hco3) :- allowed(bicarbonate).
expected(bicarbonate, ph, increase, bicarbonate_raises_ph) :- allowed(bicarbonate).
expected(dextrose, g, increase, dextrose_raises_glucose_without_iv_insulin) :- allowed(dextrose), not(insulin_requested).

expected(insulin_intermediate_sc, g, decrease, intermediate_sc_insulin_lowers_glucose) :- allowed(insulin_intermediate_sc), not(requested(dextrose)).
expected(insulin_basal_sc, g, decrease, basal_sc_insulin_lowers_glucose) :- allowed(insulin_basal_sc), not(requested(dextrose)).

% Differentiable constraints are compiled from these declarations. Thresholds
% are fixed-point values scaled by 1,000,000.
training_constraint(iv_insulin_lowers_glucose, 50000, 1000).
training_constraint(rapid_sc_insulin_lowers_glucose, 50000, 100).
training_constraint(iv_insulin_lowers_potassium_without_kcl, 50000, 500).
training_constraint(fluids_raise_map, 50000, 500).
training_constraint(fluids_raise_volume, 50000, 500).
training_constraint(kcl_raises_potassium_without_iv_insulin, 50000, 500).
training_constraint(kcl_raises_total_body_store, 50000, 500).
training_constraint(bicarbonate_raises_hco3, 50000, 500).
training_constraint(bicarbonate_raises_ph, 50000, 100).
training_constraint(dextrose_raises_glucose_without_iv_insulin, 50000, 500).
