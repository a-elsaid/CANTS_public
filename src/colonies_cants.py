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
from search_space_cants import RNNSearchSpaceCANTS
#import concurrent.futures as conc_futures
from mpi4py import MPI

comm_mpi = MPI.COMM_WORLD
comm_size = comm_mpi.Get_size()
rank = comm_mpi.Get_rank()

sys.path.insert(1, "/home/aaevse/loguru")

args = Args_Parser(sys.argv)

NUM_COLONIES = 20
LIVING_TIME = 1000

def logger_setup():
    logger.remove()
    logger.add(sys.stdout, level=args.term_log_level)
    logger.add(f"{args.log_dir}/{args.log_file_name}_cants.log", level=args.file_log_level)

def create_colony():
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

    return colony

intervals = args.communication_intervals
if intervals > args.living_time + 1:
    logger.error(
        f"Colonies evolution intervals ({intervals}) less" +
        f"than the total number of iterations ({args.living_time+1})"
    )
    sys.exit()


def living_colony():
    """
    used by threads to get the colonies to live in parallel
    """
    logger_setup()
    logger.info(f"Starting Colony Worker For Colony {rank}")
    logger.info(f"Worker {rank} reporting for duty")
    colony = create_colony()
    col_id, workers, best_position_global, fitness_global = comm_mpi.recv(source=0)
    colony.id = col_id+1
    logger.info(f"Worker {rank} Received Main's Kickoff Msg")


    for tim in range(intervals, args.living_time + 1, intervals):
        colony.life_mpi(intervals, comm_mpi, workers, rank)
        if rank==workers[0]:
            best_rnns = np.array(colony.best_rnns)
            logger.trace(f"Worker({rank}:: Collecting Fitnees from Colony({colony.id})")
            best_fits = best_rnns[:, 0]
            avg_col_fit = sum(best_fits) / len(best_fits)
            if avg_col_fit < fitness_global or fitness_global == -1:
                best_position_global = colony.pso_best_position
                fitness_global = avg_col_fit

            logger.info(f"Lead Worker {rank} reporting its OverallFitness: {fitness_global} for Colony {colony.id} No. Ants ({colony.num_ants}) ER ({colony.evaporation_rate}) MR ({colony.mortality_rate})  ({tim}/{intervals})")
            comm_mpi.send((tim, best_position_global, fitness_global), dest=0)
            best_position_global, fitness_global = comm_mpi.recv(source=0)
            colony.update_velocity(best_position_global)
            colony.update_position()

    if rank==workers[0]:
        comm_mpi.send(None, dest=0)

worker_group = np.arange(1,comm_size).reshape(args.num_colonies,-1)
def main():
    logger_setup()
    best_position_global = None
    fitness_global = -1
    BEST_POS_GOL = [0] * args.num_colonies
    FIT_GOL = np.zeros(args.num_colonies)
    logger.info(f"Main reporting for duty")
    for c in range(args.num_colonies):
        for w in worker_group[c]:
            logger.info(f"Main sending Worker {w} its' kickoff msg") 
            comm_mpi.send((c, worker_group[c], best_position_global, fitness_global), dest=w)
            logger.info(f"Main finished sending Worker {w} its' kickoff msg") 

    done_workers = 0
    while True:
        for c in range(args.num_colonies):
            msg = comm_mpi.recv(source=worker_group[c][0])
            if msg:
                tim, best_position_global, fitness_global = msg
                BEST_POS_GOL[c] = best_position_global
                FIT_GOL[c] = fitness_global
            else:
                done_workers+=1
        if done_workers==args.num_colonies:
            break
        elif 0<done_workers<args.num_colonies:
            print("SOMETHING IS WRONG")
            sys.exit()
        fitness_global = np.min(FIT_GOL)
        best_position_global = BEST_POS_GOL[np.argmin(FIT_GOL)]
        logger.info(f"*** Finished {tim}/{args.living_time} Living Time ** Best Global Fitness: {fitness_global} ***")
        for c in range(args.num_colonies):
            comm_mpi.send((best_position_global, fitness_global), dest=worker_group[c][0])
        
    '''
        **** TODO ****
        add code to save the best performing RNN in each round of intervals    
        this can be done by sending a signal to the lead-worker to save its
        best RNN if its group did the best job
    '''
    """
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
    """

if rank==0:
    main()
else:
    living_colony()
