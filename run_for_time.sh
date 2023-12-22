fname="withoutBP"
activation="sigmoid"
cpu=8
a=35
living_time=200
in_param="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow  Main_Flm_Int"

#time mpirun -n $cpu --oversubscribe python3 src/colony_cants.py --data_dir ./ --data_files sample_data.csv --input_names $in_param --output_names Main_Flm_Int --living_time $living_time --out_dir OUT_single_cants --log_dir LOG_single_cants --term_log_level INFO --log_file_name "cants_trial_ants"$a"_"$fname"_"$i --file_log_level INFO --col_log_level INFO --num_ants $a --use_cants --loss_fun mse --act_fun $activation


time mpirun -n $cpu --oversubscribe python3 src/colony_cants.py --data_dir ./ --data_files sample_data.csv --input_names $in_param --output_names Main_Flm_Int --log_dir LOG_single_cantsbp --out_dir OUT_single_cantsbp --living_time $living_time  --term_log_level INFO --log_file_name cants_trial_$f_name_$i --file_log_level INFO --col_log_level INFO --num_ants $a --use_bp --bp_epochs 20 --use_cants --loss_fun mse --act_fun $activation
