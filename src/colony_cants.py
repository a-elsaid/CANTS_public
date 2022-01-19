"""
A colony controls:
 - the foraging of ants
 - the creation of RNN from the paths of the ants
 - the training/testing of the RNN
 - updating the pheromone value
 - updating the weights in the search space from the tested RNN
 - evolving the ants charactaristics
 - updating the poplution of the best performing RNN
also use as a PSO particle to evolve coexisting colonies
"""
import sys
from typing import List
import warnings
import numpy as np
from sklearn.cluster import DBSCAN
from mpi4py import MPI
from loguru import logger
from tqdm import tqdm
import torch

from search_space_cants import Point, RNNSearchSpace, MAX_PHEROMONE
from ant_cants import Ant
from rnn import RNN
from timeseries import Timeseries
from helper import Args_Parser

warnings.filterwarnings("error")
np.warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)


class Colony:
    """
    Ants Colony
    """

    counter = 1

    def __init__(
        self,
        data: Timeseries,
        logger,
        use_bp: bool = False,
        num_epochs: int = 100,
        num_ants: int = 100,
        ants_mortality: int = 0.5,
        population_size: int = 20,
        search_space: RNNSearchSpace = None,
        max_pheromone: float = 10.0,
        min_pheromone: float = 0.5,
        default_pheromone: float = 1.0,
        space_lags: int = 10,
        dbscan_dist: float = 0.5,
        evaporation_rate: float = 0.9,
        log_dir: str = ".",
        col_log_level: str = "INFO",
    ) -> None:
        self.logger = logger
        self.col_log_level = col_log_level
        self.id = self.counter
        self.logger = logger.bind(col_id=self.id)
        self.logger.add(
            f"{log_dir}/colony_{self.id}.log",
            filter=lambda record: record["extra"].get("col_id") == self.id,
            level=self.col_log_level,
        )
        self.num_epochs = num_epochs
        self.space_lags = space_lags
        self.use_bp = use_bp
        self.log_dir = log_dir
        self.mpi_size = 0
        if use_bp:
            self.comm = MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
        self.num_ants = num_ants
        self.best_population_size = population_size
        self.mortality_rate = ants_mortality
        self.evaporation_rate = evaporation_rate
        self.max_pheromone = max_pheromone
        self.min_pheromone = min_pheromone
        self.default_pheromone = default_pheromone
        self.dbscan_dist = dbscan_dist
        self.data = data
        self.space = search_space
        if not self.space:
            self.space = RNNSearchSpace(
                inputs_names=data.input_names,
                outs_names=data.output_names,
                lags=self.space_lags,
                logger=self.logger,
            )
        self.foragers = [
            Ant(i + 1, self.logger, colony_id=self.id, log_dir=self.log_dir)
            for i in range(self.num_ants)
        ]
        self.best_rnns = []
        self.pso_position = [self.num_ants, self.mortality_rate, self.evaporation_rate]
        self.pso_velocity = np.random.uniform(
            low=-1, high=1, size=len(self.pso_position)
        )
        self.pso_best_position = self.pso_position
        self.pso_bounds = [[10, 300], [0.01, 0.1], [0.15, 0.95]]
        Colony.counter += 1

    def update_velocity(self, pos_best_g):
        """update new particle velocity"""

        self.logger.info(f"COLONY({self.id}):: Updating Colony PSO velocity")

        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognative constant
        c2 = 2  # social constant

        for i, pos in enumerate(self.pso_position):
            r1 = np.random.random()
            r2 = np.random.random()

            vel_cognitive = c1 * r1 * (self.pso_best_position[i] - pos)
            vel_social = c2 * r2 * (pos_best_g[i] - pos)
            self.pso_velocity[i] = w * self.pso_velocity[i] + vel_cognitive + vel_social

    def update_position(self):
        """update the particle position based off new velocity updates"""
        self.logger.info(f"COLONY({self.id}):: Updating Colony PSO position")
        for i, pos in enumerate(self.pso_position):
            self.pso_position[i] = pos + self.pso_velocity[i]

            # adjust position if necessary
            if pos < self.pso_bounds[i][0] or pos > self.pso_bounds[i][1]:
                if i < 1:
                    self.pso_position[i] = np.random.randint(
                        low=self.pso_bounds[i][0], high=self.pso_bounds[i][1]
                    )
                else:
                    self.pso_position[i] = np.random.uniform(
                        low=self.pso_bounds[i][0], high=self.pso_bounds[i][1]
                    )
            """
            # adjust maximum position if necessary
            if pos > self.pso_bounds[i][1]:
                self.pso_position[i] = self.pso_bounds[i][1]

            # adjust minimum position if neseccary
            if pos < self.pso_bounds[i][0]:
                self.pso_position[i] = self.pso_bounds[i][0]
            """

    def forage(
        self,
    ) -> None:
        """
        Letting ants forage to create paths to create RNN
        """

        def thread_work(space, ant, logger):
            """
            to be used by threads doing ant work
            """
            ant.forage(space)

        """
        threads = []
        for _, ant in enumerate(self.foragers):
            ant.reset()
            thread = th.Thread(
                target=thread_work,
                args=(
                    self.space,
                    ant,
                    logger,
                ),
            )
            threads.append(thread)
            logger.info(f"Ant {ant.id} starting a forage")
            thread.start()
            logger.info(f"Ant {ant.id} Finished forage")
        for i, _ in enumerate(self.foragers):
            threads[i].join()
        logger.info("All Ants Finished Foraging")
        """
        for _, ant in enumerate(self.foragers):
            ant.reset()
            ant.forage(self.space)

    def calcualte_distance_ceteroid_cluster(
        self,
        point: Point,
        cluster: List[Point],
    ) -> np.ndarray:
        """
        calculates distance between the centeriod of cluster
        and the points in the cluster
        """
        distances = []
        for cluster_point in cluster:
            distance = (
                np.sqrt((point.pos_x - cluster_point.pos_x) ** 2)
                + np.sqrt((point.pos_y - cluster_point.pos_y) ** 2)
                + np.sqrt((point.pos_l - cluster_point.pos_l) ** 2)
                + np.sqrt((point.pos_w - cluster_point.pos_w) ** 2)
            )
            distances.append(
                [
                    distance,
                    cluster_point,
                ]
            )
        if len(distances) < 2:
            return [distances[0][1]]
        try:
            temp = np.array(sorted(distances, key=lambda d: d[0]))[:, 1]
        except np.VisibleDeprecationWarning:
            self.logger.error("Distances list should not be less than 2 elements")
        return temp

    def update_pheromone_const(
        self, rnn: RNN, pheromone_increment: float = 1.0
    ) -> None:
        """
        update the pheromone values using a constant
        """
        for node in rnn.nodes.values():
            point = node.point
            lag_pheromone_boost = (
                1 + float(self.space_lags - point.pos_l) / self.space_lags
            )
            pheromone_increment *= lag_pheromone_boost
            if point.type == 0:
                self.space.input_space.increase_pheromone(point, pheromone_increment)
                continue
            if point.type == 2:
                self.space.output_space.increase_pheromone(point, pheromone_increment)
                continue
            point.pheromone = min(point.pheromone + pheromone_increment, MAX_PHEROMONE)

            # calculate distance between the centeroid and cluster points
            cluster_distances = self.calcualte_distance_ceteroid_cluster(
                point, rnn.centeroids_clusters[point.id]
            )
            # distribute pheromone over the cluster points based on distance
            # from ceteroid
            pheromone_step = pheromone_increment / len(cluster_distances)
            for i, pnt in enumerate(rnn.centeroids_clusters[point.id]):
                pnt.pheromone += pheromone_step * (i + 1) * (1 + pnt.pos_l * 0.1)
                pnt.pheromone = min(pnt.pheromone, MAX_PHEROMONE)

    def update_pheromone_noreg(
        self,
    ) -> None:
        """
        updating the pheromone values using fitness
        """
        ...

    def update_pheromone_l1reg(
        self,
    ) -> None:
        """
        updating the pheromone values using fitness and L1 regularization
        """
        ...

    def update_pheromone_l2_reg(
        self,
    ) -> None:
        """
        updating the pheromone values using fitness and L2 regularization
        """
        ...

    def update_search_space_weights(self, rnn: RNN) -> None:
        """
        update the weights of the search space from the trained/tested RNN
        """
        with torch.no_grad():
            for node in rnn.nodes.values():
                if len(node.fan_out) == 0:
                    continue
                node_edges_avr_wghts = np.average(
                    [e.weight for e in node.fan_out.values()]
                )
                for cluster_point in rnn.centeroids_clusters.get(node.point.id, []):
                    cluster_point.pos_w = (
                        cluster_point.pos_w + node_edges_avr_wghts
                    ) / 2
                node.point.pos_w = (node.point.pos_w + node_edges_avr_wghts) / 2

    def find_centeroid(self, cluster: np.ndarray) -> Point:
        """
        calculate the cenriod of a cluster of points
        used to condensate the points to a centriod
        """
        n = len(cluster)
        sum_x = np.sum(cluster[:, 0])
        sum_y = np.sum(cluster[:, 1])
        sum_l = np.sum(cluster[:, 2])
        sum_w = np.sum(cluster[:, 3])
        new_point = Point(sum_x / n, sum_y / n, round(sum_l / n), sum_w / n)
        self.space.all_points[new_point.id] = new_point
        return new_point

    def create_nn(
        self,
    ) -> RNN:
        """
        create RNN from the paths of the ants
        """
        self.logger.info(f"COLONY({self.id}):: Starting to build RNN from Ants' paths")
        points = []
        for ant in self.foragers:
            for p in ant.path:
                if not p.name:
                    points.append(p)
        points_vertecies = np.array(
            [[p.pos_x, p.pos_y, p.pos_l, p.pos_w] for p in points]
        )
        db = DBSCAN(
            eps=self.dbscan_dist, min_samples=2, metric="euclidean", n_jobs=-1
        ).fit(points_vertecies)
        labels = db.labels_  # these are the labels for each point in space
        # number of clusters used to iterate over the labels of the clusters
        num_clusters = np.max(db.labels_)
        centeroids = []
        for i in range(num_clusters + 1):
            centeroids.append(self.find_centeroid(points_vertecies[labels == i]))
        # condensed_path: Ant: List[condensed_points]
        condensed_paths = [[] for _ in range(len(self.foragers))]
        counter = 0
        for i, ant in enumerate(self.foragers):
            prev_pnt = ant.path[0]
            for p in ant.path:
                if p.name:  # checks if the point is for an input or output
                    condensed_paths[i].append(p)
                    continue
                label = labels[counter]
                counter += 1
                if label != -1:  # not single point cluster
                    p = centeroids[label]

                if (p.pos_l <= prev_pnt.pos_l and p.pos_y < prev_pnt.pos_y) or (
                    p in condensed_paths[i]
                ):
                    continue  # skip point if same-level and before prev_point or centroid is same as anther point in path

                condensed_paths[i].append(p)
                prev_pnt = condensed_paths[i][-1]
        # clusters will be used to distribute the pheromone
        # on the points in the vicinity of the ceteriods
        clusters = {}
        points = np.array(points)
        labels = np.array(labels)
        for i, point in enumerate(centeroids):
            clusters[point.id] = points[labels == i]
        for i, lag in enumerate(labels):
            if lag == -1:
                clusters[points[i].id] = [points[i]]
        for path in condensed_paths:
            clusters[path[0].id] = [path[0]]
            clusters[path[-1].id] = [path[-1]]

        for i, path in enumerate(condensed_paths):
            self.logger.debug(f"Path {i}")
            for pnt in path:
                self.logger.debug(
                    f"\t Point id: {pnt.id} Point Type: {pnt.type}  Point Name: {pnt.name} x:{pnt.pos_x:.4f} y:{pnt.pos_y:.4f} l:{pnt.pos_l} w:{pnt.pos_w:.4f}"
                )
        self.logger.info(f"COLONY({self.id}):: Finished building RNN from Ants' paths")
        return RNN(condensed_paths, clusters, lags=self.space_lags)

    def insert_rnn(self, rnn: RNN) -> None:
        """
        inserts rnn to the population and sorts based on rnns fitnesses
        """
        if len(self.best_rnns) < self.best_population_size:
            self.best_rnns.append([rnn.fitness, rnn])
        else:
            if self.best_rnns[-1][0] > rnn.fitness:
                self.best_rnns[-1] = [rnn.fitness, rnn]
                self.update_search_space_weights(rnn)
                self.update_pheromone_const(
                    rnn
                )  # TODO Put the other pheromone update options
        self.best_rnns = sorted(self.best_rnns, key=lambda r: r[0])
        self.space.evaporate_pheromone()
        self.logger.info(
            f"COLONY({self.id})::\t RNN Fitness: {rnn.fitness:.5f} (Best RNN Fitness: {self.best_rnns[0][0]:.5f})"
        )
        if rnn.fitness > 1.0:
            sys.exit()

    def evaluate_rnn(self, rnn: RNN) -> None:
        """
        training/testing RNN
        """
        self.logger.info(f"COLONY({self.id}):: Staring RNN Colony Evaluating")
        if self.use_bp:
            for i in tqdm(range(self.num_epochs)):
                rnn.do_epoch(self.data.train_input, self.data.train_output)
                rnn.feedbackward()
                if i % 100 == 0:
                    self.logger.info(f"COLONY({self.id}):: BP Training MSE: {rnn.err}")
            rnn.test_rnn(self.data.test_input, self.data.test_output)
            self.logger.info(f"Testing MSE: {rnn.fitness}")
        else:
            self.logger.info(f"COLONY({self.id}):: \t starting epoch")
            rnn.do_epoch(self.data.train_input, self.data.train_output)
            self.logger.info(f"COLONY({self.id}):: \t finished epoch")
            self.logger.info(f"COLONY({self.id}):: \t starting RNN evaluation")
            rnn.test_rnn(self.data.test_input, self.data.test_output)
            self.logger.info(f"COLONY({self.id}):: \t finished RNN evaluation")
        self.logger.info(f"COLONY({self.id}):: Finished RNN Colony Evaluation")
        return rnn

    def main_process(
        self,
        num_marchs,
    ) -> RNN:
        """
        for BP training RNNs asychronously: sends jobs to training workers
        """
        for march_num in range(self.mpi_size):
            self.forage()
            rnn = self.create_nn()
            self.comm.send(rnn)
        for march_num in range(num_marchs - self.mpi_size):
            self.logger.info(f"COLONY({self.id}): Interation {march_num}/{num_marchs}")
            rnn = self.comm.recv()
            for ant in self.foragers:
                ant.update_best_behaviors(rnn.fitness)
                ant.evolve_behavior()
            self.insert_rnn(rnn)
            self.forage()
            rnn = self.create_nn()
            self.comm.send(rnn)
        for march_num_ in range(self.mpi_size):
            self.logger.info(
                f"COLONY({self.id}): Interation {march_num+march_num_}/{num_marchs}"
            )
            rnn = self.comm.recv()
            self.insert_rnn(rnn)
            self.comm.send(None)

    def worker(
        self,
    ) -> None:
        """
        for BP training RNNs asychronously: trains RNN
        """
        while True:
            rnn = self.comm.recv(source=0)
            if rnn:
                rnn = self.evaluate_rnn(rnn)
                self.comm.send(dest=0)
            else:
                break

    def live(self, total_marchs) -> None:
        """
        Do one colony foraging step
        """
        if self.mpi_size != 0:
            rank = self.comm.Get_rank()
            if rank == 0:
                self.main_process(total_marchs)
            else:
                self.worker()
        else:
            for march_num in range(total_marchs):
                self.logger.info(
                    f"Colony({self.id}): Interation {march_num}/{total_marchs}"
                )
                self.forage()
                rnn = self.create_nn()
                rnn = self.evaluate_rnn(rnn)
                for ant in self.foragers:
                    ant.update_best_behaviors(rnn.fitness)
                    ant.evolve_behavior()
                self.insert_rnn(rnn)


if __name__ == "__main__":

    args = Args_Parser(sys.argv)

    logger.remove()
    logger.add(sys.stderr, level=args.term_log_level)

    data_dir = "2018_coal"
    input_params = "Conditioner_Inlet_Temp,Conditioner_Outlet_Temp".replace(",", " ")
    output_params = "Main_Flm_Int"
    data_files = args.data_files
    input_params = args.input_names
    output_params = args.output_names
    data_dir = args.data_dir
    data = Timeseries(
        data_files=data_files,
        input_params=input_params,
        output_params=output_params,
        data_dir=data_dir,
    )
    colony = Colony(
        data=data,
        num_ants=args.num_ants,
        use_bp=args.use_bp,
        max_pheromone=args.max_pheromone,
        min_pheromone=args.min_pheromone,
        default_pheromone=args.default_pheromone,
        num_epochs=args.bp_epochs,
        population_size=args.colony_population_size,
        space_lags=args.time_lags,
        dbscan_dist=args.dbscan_dist,
        evaporation_rate=args.evaporation_rate,
        log_dir=args.log_dir,
        logger=logger,
        col_log_level=args.col_log_level,
    )

    data.train_input = data.train_input[:20]
    data.test_input = data.test_input[:20]
    data.train_output = data.train_output[:20]
    data.test_output = data.test_output[:20]
    colony.live(args.living_time)

    colony.use_bp = True
    colony.evaluate_rnn(colony.best_rnns[0][1])
