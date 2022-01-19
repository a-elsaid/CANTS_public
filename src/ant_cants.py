"""
Ant class for the optimization agents
"""

import sys
from typing import List
from threading import Lock
import numpy as np

from search_space_cants import Point, RNNSearchSpace


space_lock = Lock()


class Ant:
    """
    This class represents the optimization agents
    """

    def __init__(
        self,
        ant_id: int,
        logger,
        sense_range: float = 0.1,
        explore_rate: float = 0.5,
        mortality: float = 0.0,
        colony_id: int = None,
        log_dir: str = ".",
    ) -> None:
        self.id = ant_id
        self.colony_id = colony_id
        self.sense_range = sense_range
        self.death_probablity = mortality
        self.current_x: float = 0.5
        self.current_y: float = 0.0
        self.current_w: float = 0.0
        self.current_l: int = 0
        self.exploration_rate = explore_rate
        self.path: List[Point] = []
        self.mutation_sigma = 0.15
        self.region_move1 = 0.0
        self.region_move2 = 0.0
        self.best_behaviors: List[
            List[float]
        ] = []  # [[RNN_performance, exploration_rate, sense_range, region_move]]
        self.logger = logger

    def update_best_behaviors(self, rnn_fitness) -> None:
        """
        update ant charactaristics based on the rnn performance
        """
        if len(self.best_behaviors) < 10:
            self.best_behaviors.append(
                [
                    rnn_fitness,
                    self.exploration_rate,
                    self.sense_range,
                    self.region_move1,
                    self.region_move2,
                ]
            )
        else:
            if rnn_fitness < self.best_behaviors[-1][0]:
                self.best_behaviors[-1] = [
                    rnn_fitness,
                    self.exploration_rate,
                    self.sense_range,
                    self.region_move1,
                    self.region_move2,
                ]
        self.best_behaviors.sort()

    def evolve_behavior(
        self,
    ) -> None:
        """
        using GA to evolve ant characteristics
        using cross over and mutations
        """

        def mutate():
            """
            perform mutations
            """
            (
                self.exploration_rate,
                self.sense_range,
                self.region_move1,
                self.region_move2,
            ) = (
                np.random.random(),
                np.random.random(),
                np.random.uniform(low=-1, high=1),
                np.random.uniform(low=-1, high=1),
            )

        def cross_over(behavior1: np.ndarray, behavior2: np.ndarray):
            """
            perform cross over
            """
            # new_behavior = (behavior1 + behavior2) / 2
            new_behavior = list(
                ((np.subtract(behavior2[1:], behavior1[1:])) * np.random.random())
                + behavior1[1:]
            )
            (
                self.exploration_rate,
                self.sense_range,
                self.region_move1,
                self.region_move2,
            ) = new_behavior

        if len(self.best_behaviors) < 10 or np.random.random() < self.mutation_sigma:
            mutate()
        else:
            indecies = np.arange(len(self.best_behaviors))
            indecies = np.random.choice(indecies, 2, replace=False)
            cross_over(
                self.best_behaviors[indecies[0]], self.best_behaviors[indecies[1]]
            )

    def center_of_mass(self, proximity_points: List[Point]) -> Point:
        """
        find the center of mass of points based on pheromone value
        """
        pheromone_mass = np.sum([pnt.pheromone for pnt in proximity_points])
        pheromone_mass_center_x = (
            np.sum([pnt.pos_x * pnt.pheromone for pnt in proximity_points])
            / pheromone_mass
        )
        pheromone_mass_center_y = (
            np.sum([pnt.pos_y * pnt.pheromone for pnt in proximity_points])
            / pheromone_mass
        )
        pheromone_mass_center_l = int(
            np.round(
                np.sum([pnt.pos_l * pnt.pheromone for pnt in proximity_points])
                / pheromone_mass
            )
        )
        pheromone_mass_center_w = (
            np.sum([pnt.pos_w * pnt.pheromone for pnt in proximity_points])
            / pheromone_mass
        )
        point = Point(
            pheromone_mass_center_x,
            pheromone_mass_center_y,
            pheromone_mass_center_l,
            pheromone_mass_center_w,
        )
        return point

    def get_proximity_points(
        self, all_points: List[Point], same_level: bool
    ) -> List[Point]:
        def decide_same_level(pnt_level):
            if same_level:
                return pnt_level <= self.current_l
            return False

        proximity_points = [
            pnt
            for pnt in all_points.values()
            if np.sqrt(
                (self.current_x - pnt.pos_x) ** 2
                + (self.current_y - pnt.pos_y) ** 2
                + (self.current_l - pnt.pos_l) ** 2
                + (self.current_w - pnt.pos_w) ** 2
            )
            <= max(
                self.sense_range
                - (
                    self.current_y ** 2 * self.region_move1
                    + self.current_y * self.region_move2
                ),  # 2nd degree polynomial to control ant speed based on location
                0.1,
            )
            and decide_same_level(pnt.pos_l)
        ]
        return proximity_points

    def pick_point(self, space: RNNSearchSpace) -> Point:
        """
        Pick the next point in the path
        param: List current_level_points: all points in the current level of ant
        return: Point: return the picked point or the created point
        """
        # TODO: CAN USE CYTHON HERE
        # with space_lock:
        proximity_points = self.get_proximity_points(space.all_points, same_level=False)
        if len(proximity_points) == 0:
            return self.create_point(space)
        point = self.center_of_mass(proximity_points)
        if point.pos_l <= self.current_l:
            proximity_points = self.get_proximity_points(
                space.all_points, same_level=True
            )

        for pnt in proximity_points:
            if (
                point.pos_x == pnt.pos_x
                and point.pos_y == pnt.pos_y
                and point.pos_l == pnt.pos_l
                and point.pos_w == pnt.pos_w
            ):
                point = pnt
                break
        return point

    def create_point(self, space: RNNSearchSpace) -> Point:
        """
        Create a new point to be added to the search space
        """

        new_l = np.random.randint(self.current_l, space.time_lags - 1, dtype=int)
        new_w = np.random.random()
        new_x = 99
        while new_x < 0.0 or new_x > 1.0:
            new_x = (
                np.random.uniform(-self.sense_range, self.sense_range) + self.current_x
            )
        different_level = new_l > self.current_l
        new_y = max(
            0.0,
            min(
                np.random.uniform(
                    -self.sense_range if different_level else 0, self.sense_range
                )
                + self.current_y,
                1.0,
            ),
        )
        new_point = Point(new_x, new_y, new_l, new_w)
        # with space_lock:
        space.all_points[new_point.id] = new_point
        return new_point

    def move(self, space: RNNSearchSpace) -> None:
        """
        Move from one position to another
        :param Dict space: all points in the search space
        """

        with space_lock:
            if np.random.random() > self.exploration_rate:
                point = self.pick_point(space)
                if point.pos_l <= self.current_l and point.pos_y < self.current_y:
                    self.logger.error(
                        f"Picked Point Error: P.type:{point.type} P.l:{point.pos_l} P.y:{point.pos_y} Ant.l: {self.current_l} Ant.y: {self.current_y}"
                    )
                    sys.exit()

            else:
                point = self.create_point(space)
                if point.pos_l <= self.current_l and point.pos_y < self.current_y:
                    self.logger.error(
                        f"Create Point P.l:{point.pos_l} P.y:{point.pos_y} Ant.l: {self.current_l} Ant.y: {self.current_y}"
                    )
                    sys.exit()

        self.path.append(point)
        self.current_x = point.pos_x
        self.current_y = point.pos_y
        self.current_w = point.pos_w
        self.current_l = point.pos_l

    def forage(self, space: RNNSearchSpace) -> None:
        """
        keep moving in the search space till reach output nodes
        """
        self.path.append(space.input_space.get_input(self.exploration_rate))
        while self.path[-1].pos_y < 1.0:
            self.move(space)
        self.path.pop(-1)
        self.path.append(space.output_space.get_point())
        # self.path.append(space.output_space.get_output(self.exploration_rate))

    def reset(
        self,
    ) -> None:
        """
        reset ant charactaristics
        """
        self.path = []
        self.current_x = 0.5
        self.current_y = 0.0
        self.current_w = 0.0
        self.current_l = 0
