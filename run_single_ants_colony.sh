f_name="withBP"
activation="sigmoid"
for i in {1..10}; do
    for a in 5 10 15 25 35 50; do
        echo "Starting ANTS Experiment: " $i
    time python src/colony_cants.py --data_dir ./ --data_files sample_data.csv --input_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow  Main_Flm_Int --output_names Main_Flm_Int --log_dir LOG_single_ants --out_dir OUT_single_ants --living_time 1000  --term_log_level INFO --log_dir ./log --log_file_name ants_trial_$f_name_$i --file_log_level INFO --col_log_level INFO --num_ants $a --use_bp --bp_epochs 20 --num_threads 30 --act_fun mse --act_fun $activation
    done
done
