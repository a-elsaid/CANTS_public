for i in {1..10}; do
    echo "Starting CANTS Experiment: " $i
time python src/colony_cants.py --data_dir ./data/2018_coal/ --data_files burner_{0..2}.csv --input_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow --output_names Main_Flm_Int --living_time 2000  --term_log_level INFO --log_dir ./log --log_file_name ants_trial_$i --file_log_level INFO --col_log_level INFO --num_ants 200 --use_bp --bp_epochs 50 --num_threads 60
done

