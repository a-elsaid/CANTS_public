"""
to run the colonies in parallel and evolve them
using PSO
"""
import sys
import pickle
import threading as th
import numpy as np
from loguru import logger
from colony_cants import Colony
from timeseries import Timeseries
from helper import Args_Parser


args = Args_Parser(sys.argv)
NUM_COLONIES = 20
LIVING_TIME = 1000

fitness_global = -1

logger.remove()
logger.add(sys.stderr, level=args.term_log_level)
logger.add(f"{args.log_dir}/cants.log", level=args.file_log_level)

colonies = []
threads = []
for c in range(args.num_colonies):
    data_files = "burner_0.csv"
    data_dir = "2018_coal"
    input_params = "Conditioner_Inlet_Temp,Conditioner_Outlet_Temp".replace(",", " ")
    output_params = "Main_Flm_Int"

    data_files = args.data_files
    data_dir = args.data_dir
    input_params = args.input_names
    output_params = args.output_names

    data = Timeseries(
        data_files=data_files,
        input_params=input_params,
        output_params=output_params,
        data_dir=data_dir,
    )
    data.train_input = data.train_input[:20]
    data.test_input = data.test_input[:20]
    data.train_output = data.train_output[:20]
    data.test_output = data.test_output[:20]
    num_ants = np.random.randint(low=5, high=50)
    population_size = np.random.randint(low=10, high=100)
    evaporation_rate = np.random.uniform(low=0.7, high=0.9)
    colony = Colony(
        data=data,
        num_ants=num_ants,
        use_bp=args.use_bp,
        max_pheromone=args.max_pheromone,
        min_pheromone=args.min_pheromone,
        default_pheromone=args.default_pheromone,
        num_epochs=args.bp_epochs,
        population_size=population_size,
        space_lags=args.time_lags,
        dbscan_dist=args.dbscan_dist,
        evaporation_rate=evaporation_rate,
        log_dir=args.log_dir,
        logger=logger,
        col_log_level=args.col_log_level,
    )
    colonies.append(colony)


def living_colony(colony: Colony, forages: int):
    """
    used by threads to get the colonies to live in parallel
    """
    colony.live(forages)


intervals = args.communication_intervals
if intervals > args.living_time + 1:
    logger.error(
        f"Colonies evolution intervals ({intervals}) less than the total number of iterations ({args.living_time+1})"
    )
    sys.exit()
for tim in range(intervals, args.living_time + 1, intervals):
    for col in colonies:
        thread = th.Thread(
            target=living_colony,
            args=(
                col,
                intervals,
            ),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    for coln in colonies:
        if (
            np.average(np.array(coln.best_rnns)[:, 0]) < fitness_global
            or fitness_global == -1
        ):
            best_position_global = coln.pso_best_position
            fitness_global = np.average(np.array(coln.best_rnns)[:, 0])
    for coln in colonies:
        coln.update_velocity(best_position_global)
        coln.update_position()
    logger.info(f"Finished {tim}/{args.living_time}")

    np.save(
        f"colony1_space_sample_{tim}",
        np.array(
            [
                [p.pos_x, p.pos_y, p.pos_l, p.pos_w]
                for p in colonies[0].space.all_points.values()
            ]
        ),
    )


best_rnn_colony = colonies[0]
for coln in colonies[1:]:
    if coln.best_rnns[0][0] < best_rnn_colony.best_rnns[0][0]:
        best_rnn_colony = coln


best_rnn_colony.use_bp = True
best_rnn_colony.evaluate_rnn(best_rnn_colony.best_rnns[0][1])


with open("best_rnn.nn", "bw") as file_obj:
    pickle.dump(best_rnn_colony.best_rnns[0][1], file_obj)
