import sys
from typing import List
import numpy as np

DEFAULT_PHEROMONE_VALUE = 1.0
EVAPORATION_RATE = 0.9
MIN_PHEROMONE = 0.5
MAX_PHEROMONE = 10.0
NODE_TYPE = {"INPUT": 0, "HIDDEN": 1, "OUTPUT": 2}
DEFAULT_LAG = 10


class RNNSearchSpaceCANTS:
    def __init__(
        self,
        logger,
        inputs_names: List[str],
        outs_names: List[str],
        lags: int = DEFAULT_LAG,
        max_pheromone: float = MAX_PHEROMONE,
        min_pheromone: float = MIN_PHEROMONE,
        evaporation_rate: float = EVAPORATION_RATE,
    ) -> None:
        self.logger = logger
        self.width = 1.0
        self.length = 1.0
        self.time_lags = lags
        self.all_points = {}
        self.evaporation_rate = evaporation_rate
        self.max_pheromone = max_pheromone
        self.min_pheromone = min_pheromone
        self.evaporated_points = 0

        self.inputs_space = self.Inputs_Space(
            self.logger,
            inputs_names,
            lags,
            self.max_pheromone,
            self.min_pheromone,
            self.evaporation_rate,
        )
        self.output_space = self.Outputs_Space(outs_names)

    def evaporate_pheromone(
        self,
    ) -> None:
        self.inputs_space.evaporate_pheromone()
        self.output_space.evaporate_pheromone()
        points_to_be_removed = []
        for point in self.all_points.values():
            point.pheromone *= self.evaporation_rate
            if point.pheromone < 0.05:
                points_to_be_removed.append(point)
        for pnt in points_to_be_removed:
            self.all_points.pop(pnt.id)
            self.evaporated_points = +1
            '''
            if self.evaporated_points > 250:
                new_all_points = {k: v for k, v in self.all_points.items()}
                self.all_points = new_all_points
                self.evaporated_points = 0
            '''
    def add_new_points(self, new_points):
        for p in new_points:
            self.all_points[p.id] = p
            p.new = False

    class Point:
        counter = 0

        def __init__(
            self,
            x: float,
            y: float,
            lag: int,
            w: float,
            point_type: int = None,
            name: str = None,
            inout_num: int = None,
        ) -> None:
            self.id = self.counter
            self.pos_x = x
            self.pos_y = y
            self.pos_l = lag  # time lag
            self.pos_w = w  # weight
            self.pheromone = DEFAULT_PHEROMONE_VALUE
            self.type = NODE_TYPE["HIDDEN"]
            self.name = name
            self.inout_num = inout_num
            self.new = True
            if self.name is not None:
                self.type = point_type
            RNNSearchSpaceCANTS.Point.counter += 1

        def print_point(
            self,
        ) -> None:
            print(
                f"Pos X: {self.pos_x}, Pos Y: {self.pos_y}, Pos L: {self.pos_l}, " +
                f"Weight: {self.pos_w}, Type: {self.type}, Name: {self.name}, " +
                f"In_No: {self.inout_num}"
            )

    class Single_Input_Space:
        def __init__(self, in_num: int, lags: int, name: str) -> None:
            self.time_lags = lags
            self.name: str = name
            self.in_num: int = in_num
            self.points: List[RNNSearchSpaceCANTS.Point] = []
            self.input_pheromone = DEFAULT_PHEROMONE_VALUE

        def get_point(
            self,
        ):
            pheromones = [pnt.pheromone for pnt in self.points]
            norm_pheromones = pheromones / np.sum(pheromones)
            return np.random.choice(self.points, 1, p=norm_pheromones)[0]

    class Inputs_Space:
        def __init__(
            self,
            log,
            inputs_names,
            lags: int,
            max_pheromone,
            min_pheromone,
            evaporation_rate,
        ) -> None:
            self.inputs_spaces = {}
            self.inputs_names = inputs_names
            self.lags = lags
            self.max_pheromone = max_pheromone
            self.min_pheromone = min_pheromone
            self.evaporation_rate = evaporation_rate
            for i, name in enumerate(inputs_names):
                self.inputs_spaces[name] = RNNSearchSpaceCANTS.Single_Input_Space(
                    in_num=i, lags=lags, name=self.inputs_names[i]
                )
            if len(self.inputs_spaces) != len(inputs_names):
                log.error(f"Input Names Problem: {inputs_names}")
                sys.exit()

        def get_input(self, ant_exploration_rate: float):
            if np.random.random() < ant_exploration_rate:
                input_space = self.inputs_spaces[
                    self.inputs_names[
                        np.random.randint(low=0, high=len(self.inputs_names))
                    ]
                ]
            else:
                pheromones = [
                    input_space.input_pheromone
                    for input_space in self.inputs_spaces.values()
                ]
                norm_pheromones = pheromones / np.sum(pheromones)
                input_name = np.random.choice(self.inputs_names, 1, p=norm_pheromones)[
                    0
                ]
                input_space = self.inputs_spaces[input_name]
            if (
                np.random.random() < ant_exploration_rate
                or len(input_space.points) == 0
            ):
                new_point = RNNSearchSpaceCANTS.Point(
                    0.0,
                    0.0,
                    np.random.randint(0, self.lags),
                    np.random.random(),
                    NODE_TYPE["INPUT"],
                    input_space.name,
                    inout_num=input_space.in_num,
                )
                input_space.points.append(new_point)
                return new_point
            return input_space.get_point()

        def increase_pheromone(self, point, pheromone_step: float) -> None:
            new_phermone_value = (
                self.inputs_spaces[point.name].input_pheromone + pheromone_step
            )
            self.inputs_spaces[point.name].input_pheromone = min(
                new_phermone_value, self.max_pheromone
            )
            point.pheromone = min(point.pheromone + pheromone_step, self.max_pheromone)

        def evaporate_pheromone(
            self,
        ) -> None:
            for input_space in self.inputs_spaces.values():
                max(
                    input_space.input_pheromone * self.evaporation_rate,
                    self.min_pheromone,
                )
                points_to_be_removed = []
                for point in input_space.points:
                    max(point.pheromone * self.evaporation_rate, self.min_pheromone)
                    if point.pheromone < 0.05:
                        points_to_be_removed.append(point)
                for pnt in points_to_be_removed:
                    input_space.pop(pnt)

    class Outputs_Space:
        def __init__(self, outputs_names):
            self.points = [
                RNNSearchSpaceCANTS.Point(
                    i / 10.0, 1.0, 0, np.random.random(), NODE_TYPE["OUTPUT"], name, i
                )
                for i, name in enumerate(outputs_names)
            ]

        def get_point(
            self,
        ) -> None:
            pheromones = [pnt.pheromone for pnt in self.points]
            norm_pheromones = pheromones / np.sum(pheromones)
            return np.random.choice(self.points, 1, p=norm_pheromones)[0]

        def evaporate_pheromone(
            self,
        ) -> None:
            for point in self.points:
                point.pheromone = max(point.pheromone * EVAPORATION_RATE, MIN_PHEROMONE)

        def increase_pheromone(self, pnt, pheromone_step) -> None:
            pnt.pheromone = min(pnt.pheromone + pheromone_step, MAX_PHEROMONE)
