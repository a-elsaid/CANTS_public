import sys
from typing import List
import os
import numpy as np
import torch
from loguru import logger


def mse(results: List[float], groundtruth: List[float]) -> np.ndarray:
    mse = []
    for res, gth in zip(results, groundtruth):
        mse.append(0.5 * (gth - res) ** 2)
        logger.debug(f"mse: {mse[-1]}")
    return mse


def mae(results: List[float], groundtruth: List[float]) -> np.ndarray:
    return np.abs(torch.sub(groundtruth - results))


LOSS = {"mae": mae, "mse": mse}


def sigmoid(x):
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    return 1 / (1 + np.exp(-x))


def tanh(x):
    if isinstance(x, torch.Tensor):
        return torch.tanh(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    expo = np.exp(x)
    expo_sum = np.sum(np.exp(x))
    return expo / expo_sum


def leaky_relu(x):
    y1 = (x > 0) * x
    y2 = (x <= 0) * x * 0.01
    return y1 + y2


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "softmax": softmax,
    "leaky_relu": leaky_relu,
}


class Args_Parser:
    def __init__(self, args):
        def add_params(param, count):
            while count < len(args) and "--" not in args[count]:
                param.append(args[count])
                count += 1
            return count - 1

        self.data_files = []
        self.input_names = ""
        self.output_names = ""
        self.file_log_level = "INFO"
        self.term_log_level = "INFO"
        self.col_log_level = "INFO"
        self.data_dir = "."
        self.out_dir = "."
        self.log_dir = "logs"
        self.use_bp = False
        self.bp_epochs = 0
        self.num_ants = 10
        self.max_pheromone = 10
        self.min_pheromone = 0.5
        self.ant_population_size = 10
        self.colony_population_size = 10
        self.time_lags = 5
        self.evaporation_rate = 0.9
        self.default_pheromone = 1.0
        self.dbscan_dist = 0.1
        self.num_colonies = 20
        self.communication_intervals = 50
        self.living_time = 1000
        self.dbscan_min_sample = 2
        count = 0
        while count < len(args):
            if args[count] in ["--data_files", "-f"]:
                count += 1
                count = add_params(self.data_files, count)
                self.data_files = " ".join(self.data_files)
            elif args[count] in ["--input_names", "-inms"]:
                count += 1
                self.input_names = []
                count = add_params(self.input_names, count)
                self.input_names = " ".join(self.input_names)
            elif args[count] in ["--output_names", "-onms"]:
                count += 1
                self.output_names = []
                count = add_params(self.output_names, count)
                self.output_names = " ".join(self.output_names)
            elif args[count] in ["--file_log_level", "-l"]:
                count += 1
                self.file_log_level = args[count].upper()
            elif args[count] in ["--term_log_level", "-l"]:
                count += 1
                self.term_log_level = args[count].upper()
            elif args[count] in ["--col_log_level", "-cl"]:
                count += 1
                self.col_log_level = args[count].upper()
            elif args[count] in ["--data_dir", "-d"]:
                count += 1
                self.data_dir = args[count]
            elif args[count] in ["--out_dir", "-o"]:
                count += 1
                self.out_dir = args[count].upper()
            elif args[count] in ["--log_dir", "-x"]:
                count += 1
                self.log_dir = args[count].upper()
            elif args[count] in ["--use_bp", "-b"]:
                self.use_bp = True
            elif args[count] in ["--bp_epochs", "-e"] and self.use_bp:
                count += 1
                self.bp_epochs = int(args[count])
            elif args[count] in ["--num_ants", "-a"]:
                count += 1
                self.num_ants = int(args[count])
            elif args[count] in ["--max_pheromone", "-m"]:
                count += 1
                self.max_pheromone = float(args[count])
            elif args[count] in ["--min_pheromone", "-n"]:
                count += 1
                self.min_pheromone = float(args[count])
            elif args[count] in ["--ant_population", "-s"]:
                count += 1
                self.ant_population_size = int(args[count])
            elif args[count] in ["--colony_population", "-c"]:
                count += 1
                self.colony_population_size = int(args[count])
            elif args[count] in ["--lags", "-t"]:
                count += 1
                self.time_lags = int(args[count])
            elif args[count] in ["--default_pheromone", "-dph"]:
                count += 1
                self.default_pheromone = float(args[count])
            elif args[count] in ["--evaporation_rate", "-evp"]:
                count += 1
                self.evaporation_rate = float(args[count])
            elif args[count] in ["--max_dbscan_dist", "-dbdst"]:
                count += 1
                self.dbscan_dist = float(args[count])
                if self.dbscan_dist < 0.012 or self.dbscan_dist > 0.098:
                    logger.error(
                        f"Max DBSCAN distance ({self.dbscan_dist}) is not in [0.012, 0.098]"
                    )
                    sys.exit()
            elif args[count] in ["--max_dbscan_smpl", "-dbsmpl"]:
                count += 1
                self.dbscan_min_sample = int(args[count])

            elif args[count] in ["--num_col", "-nc"]:
                count += 1
                self.num_colonies = int(args[count])
            elif args[count] in ["--comm_interval", "-comi"]:
                count += 1
                self.communication_intervals = int(args[count])
            elif args[count] in ["--living_time", "-livt"]:
                count += 1
                self.living_time = int(args[count])
            count += 1
        if self.data_files == "":
            logger.error("No Data Files Provided")
            sys.exit()
        if self.input_names == "":
            logger.error("No Input Name(s) Provided")
            sys.exit()
        if self.output_names == "":
            logger.error("No Output Name(s) Provided")
            sys.exit()
        logger.info(f"Data Files: {self.data_files}")
        logger.info(f"Input Parameters Names: {self.input_names}")
        logger.info(f"Output Parameters Names: {self.output_names}")
        logger.info(f"Terminal Log Level: {self.term_log_level}")
        logger.info(f"Terminal Log Level: {self.file_log_level}")
        logger.info(f"Colony Log Level: {self.col_log_level}")
        logger.info(f"Data Directory: {self.data_dir}")
        logger.info(f"Output Directory: {self.out_dir}")
        logger.info(f"Logs Directory: {self.log_dir}")
        logger.info(f"Use Backpropagation: {self.use_bp}")
        logger.info(f"Number of Ants: {self.num_ants}")
        logger.info(f"Maximum Pheromone Value: {self.max_pheromone}")
        logger.info(f"Minimum Pheromone Value: {self.min_pheromone}")
        logger.info(f"Ant's Population: {self.ant_population_size}")
        logger.info(f"Colony's Population: {self.colony_population_size}")
        logger.info(f"Time Lags: {self.time_lags}")
        logger.info(f"Evaporation Rate: {self.evaporation_rate}")
        logger.info(f"DBSCAN Distance: {self.dbscan_dist}")
        logger.info(f"DBSCAN Samples: {self.dbscan_min_sample}")
        logger.info(f"Number of Colonies: {self.num_colonies}")
        logger.info(f"Communication Intervals: {self.communication_intervals}")
        logger.info(f"Living Time: {self.living_time}")

        if not os.path.exists(self.data_dir):
            logger.error(f"Data folder ({self.data_dir})does not exit")
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
