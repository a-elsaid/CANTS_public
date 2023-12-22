cpu=8
f_name="withoutBP"
activation="sigmoid"
for i in {1..10}; do
    for a in 5 10 15 25 35 50; do
        echo "Starting Colonies CANTS Experiment without BP: " $i
        mpirun -n $cpu --oversubscribe python3 src/colonies_cants.py --data_dir ./ --data_files sample_data.csv --input_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_names Main_Flm_Int --living_time 1000  --out_dir OUT_colonies_cants --log_dir LOG_colonies_cants --term_log_level INFO --log_file_name colonies_cants_trail_$f_name_$i --file_log_level INFO --col_log_level INFO --num_ants $a --use_cants --loss_fun mse --act_fun $activation --comm_interval 50
    done
done
