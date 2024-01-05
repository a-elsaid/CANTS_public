cpu=5
f_name="withBP"
activation="sigmoid"
ulimit -v unlimited

#for i in {1..10}; do
for i in {1..1}; do
        echo "Starting Colonies CANTS Experiment with BP: " $i
        mpirun -n $cpu python3 src/colonies_cants.py --data_dir ./ --data_files _sample_data.csv --input_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_names Main_Flm_Int --living_time 1000  --out_dir OUT_cantsbp --log_dir LOG_cantsbp --term_log_level INFO --log_file_name colonies_cants_trail_$f_name_$i --file_log_level INFO --col_log_level INFO --use_bp --bp_epochs 2 --act_fun sigmoid --loss_fun mse --act_fun $activation --comm_interval 2 --use_cants  --num_col 2
    done

exit

        mpirun -n $cpu python3 src/colonies_cants.py --data_dir ./ --data_files sample_data.csv --input_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_names Main_Flm_Int --living_time 1000  --out_dir OUT_cantsbp --log_dir LOG_cantsbp --term_log_level INFO --log_file_name colonies_cants_trail_$f_name_"0" --file_log_level INFO --col_log_level INFO --use_bp --bp_epochs 20 --act_fun sigmoid --loss_fun mse --act_fun $activation --comm_interval 50 --use_cants --num_col 10
echo "FINIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIISH"
exit

