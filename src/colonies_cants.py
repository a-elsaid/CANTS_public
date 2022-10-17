"""
to run the colonies in parallel and evolve them
using PSO
"""
import sys
sys.path.insert(1, "/home/aaevse/loguru")
import pickle
import threading as th
import numpy as np
from loguru import logger
from colony_cants import Colony
from timeseries import Timeseries
from helper import Args_Parser
from search_space_cants import RNNSearchSpaceCANTS


args = Args_Parser(sys.argv)

NUM_COLONIES = 20
LIVING_TIME = 1000

fitness_global = -1

logger.remove()
logger.add(sys.stdout, level=args.term_log_level)
logger.add(f"{args.log_dir}/{args.log_file_name}_cants.log", level=args.file_log_level)

colonies = []
threads = []
for c in range(args.num_colonies):

    data = Timeseries(
        data_files=args.data_files,
        input_params=args.input_names,
        output_params=args.output_names,
        data_dir=args.data_dir,
    )

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
        out_dir=args.out_dir,
        logger=logger,
        col_log_level=args.col_log_level,
        log_file_name=args.log_file_name,
        num_threads=args.num_threads,
        ants_mortality=0.1,
        use_cants=args.use_cants,
        loss_fun=args.loss_fun,
        act_fun=args.act_fun,
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
        avg_col_fit = sum(np.array(coln.best_rnns)[:, 0]) / len(
            np.array(coln.best_rnns)[:, 0]
        )
        if avg_col_fit < fitness_global or fitness_global == -1:
            best_position_global = coln.pso_best_position
            fitness_global = avg_col_fit
    for coln in colonies:
        coln.update_velocity(best_position_global)
        coln.update_position()
    logger.info(f"Finished {tim}/{args.living_time}")

    if isinstance(colonies[0].space, RNNSearchSpaceCANTS):
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


logger.info(f"** Evaluating Best RNN in Best Colony({best_rnn_colony.id}) **")
best_rnn_colony.use_bp = True
best_rnn_colony.evaluate_rnn(best_rnn_colony.best_rnns[0][1])


with open(
    "{}/{}_best_rnn.nn".format(args.out_dir, args.log_file_name), "bw"
) as file_obj:
    pickle.dump(best_rnn_colony.best_rnns[0][1], file_obj)
