""" RNN class """
import sys
from typing import List, Dict
import numpy as np
import torch
from loguru import logger
from helper import ACTIVATIONS, LOSS
from search_space_cants import Point


class Node:
    """
    RNN node
    :param int id: node id
    :param Point point: search space point
    :param bias: node bias
    :param str activation_type: type of the activation function
    """

    counter = 0

    def __init__(
        self,
        point: Point,
        lag: int,
        activation_type: str = "sigmoid",
    ) -> None:
        self.id = self.counter
        self.activation = ACTIVATIONS[activation_type]
        self.lag = lag
        self.point = point
        self.bias = 0.0
        if self.point.name:
            self.bias = torch.tensor(
                np.random.random(), dtype=torch.float32, requires_grad=True
            )
        self.fan_in = []  # Coming in nodes
        self.fan_out = {}  # Going out edges and their nodes
        self.type = self.point.type
        self.value = torch.tensor(
            0.0, dtype=torch.float32, requires_grad=False
        )  # value in node
        self.signals_to_receive: int = 0  # number of fan_in
        self.waiting_signals: int = 0
        Node.counter += 1
        self.fired = False

    def add_fan_out_node(self, in_node, out_node, wght: float) -> None:
        if out_node not in self.fan_out:
            self.fan_out[out_node] = Edge(in_node, out_node, wght)

    def add_fan_in_node(self, in_node) -> None:
        if in_node not in self.fan_in:
            self.fan_in.append(in_node)
            self.signals_to_receive += 1

    def fire(
        self,
    ) -> float:
        """
        firing the node when activated
        """
        self.waiting_signals = self.signals_to_receive
        logger.debug(
            f"Node({self.id}) [Point({self.point.id})] Fired: Sig({self.value} \
            + {self.bias}) = {self.activation(self.value + self.bias)}"
        )
        self.value = self.activation(self.value + self.bias)
        self.fired = True
        for edge in self.fan_out.values():
            edge.out_node.synaptic_signal(self.value * edge.weight)
        with torch.no_grad():
            if self.type != 2:
                self.value = 0.0

    def synaptic_signal(self, signal: float) -> None:
        """
        receiving a synaptic signal and firing when all signals are received
        """
        self.fired = False
        logger.debug(
            f"Node({self.id}) [Point:{self.point.id}, Type: {self.type}] revieved signal: {signal}"
        )
        logger.debug(
            f"Node({self.id}) [Point({self.point.id})] value before receiving signal: {self.value}"
        )
        self.value += signal
        logger.debug(
            f"Node({self.id}) [Point({self.point.id})] value after received signal: {self.value} -- Waiting {self.waiting_signals} Signals"
        )
        self.waiting_signals -= 1

        if self.waiting_signals <= 0:
            logger.debug(f"Node({self.id}) is going to fire: {self.value}")
            self.fire()


class Edge:
    """
    RNN Edge
    :param Node from: source node
    :param Node to: sink node
    :param float weight: weight fro source node to sink node
    """

    counter = 0

    def __init__(self, in_node: Node, out_node: Node, weight: float = None):
        self.id = self.counter
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        if not self.weight:
            self.weight = np.random.random()
        self.weight = torch.tensor(self.weight, dtype=torch.float32, requires_grad=True)


class RNN:
    """
    RNN class
    """

    def __init__(
        self,
        paths: List[List[Point]],
        centeroids_clusters: Dict[int, np.ndarray],
        lags: int,
        loss_fun: str = "mse",
    ):
        self.fitness: float = None
        self.err: torch.float32 = torch.tensor(0.0, dtype=torch.float32)
        self.nodes: Dict[int, Node] = {}
        self.input_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        logger.debug(f"LOSS type: {loss_fun}  ")
        self.loss_fun = LOSS[loss_fun]
        self.centeroids_clusters = centeroids_clusters
        self.lags = lags

        for point in [pnt for path in paths for pnt in path]:
            if point.id in self.nodes:
                continue
            node = Node(point, point.pos_l)
            self.nodes[point.id] = node
            if point.type == 0:
                self.input_nodes.append(node)
            elif point.type == 2:
                self.output_nodes.append(node)

        for path in paths:
            for i, curr_p in enumerate(path[:-1]):
                curr_node = self.nodes[curr_p.id]
                next_p = path[i + 1]
                next_node = self.nodes[next_p.id]
                if curr_node.point.id in next_node.fan_out:
                    continue
                curr_node.add_fan_out_node(curr_node, next_node, curr_p.pos_w)
                next_node.add_fan_in_node(curr_node)

        for node in self.nodes.values():
            node.waiting_signals = node.signals_to_receive
            logger.debug(
                f"Node({node.id}) Point({node.point.id}) Type: {node.type} WaitingSignals: {node.waiting_signals}"
            )
            for e in node.fan_out.values():
                logger.debug(f"\t Out Node({e.out_node.id})")
            for n in node.fan_in:
                logger.debug(f"\t In Node({n.id})")

    def feedforward(
        self,
        inputs: np.ndarray,
    ) -> List[float]:
        """
        feeding forward a data point to get an output
        """
        """
        for node in self.nodes.values():
            node.waiting_signals = node.signals_to_receive
            node.fired = False
        """
        for node in self.input_nodes:
            node.synaptic_signal(inputs[self.lags - node.lag - 1][node.point.inout_num])

        raise_err = False
        for n in self.nodes.values():
            if not n.fired:
                logger.error(f"Node {n.id} Didn't Fire  Point: {n.point.id}")
                raise_err = True
        if raise_err:
            sys.exit()

        res = [node.value for node in self.output_nodes]
        for node in self.output_nodes:
            node.value = 0.0
        return res

    def do_epoch(self, inputs: np.ndarray, outputs: np.ndarray, loss_fun=None) -> None:
        """
        perform one epoch using the whole dataset
        """
        if not loss_fun:
            loss_fun = self.loss_fun
        self.err = [0.0 for i in range(len(outputs[0]))]
        for i in range(len(inputs) - self.lags):
            res = self.feedforward(inputs[i : i + self.lags])
            logger.debug(f"feedforward return (output nodes values): {res}")
            self.err = [
                e + loss for e, loss in zip(self.err, loss_fun(res, outputs[i]))
            ]
            logger.debug(f"ERR in do_epoch loop:: {self.err}")
        self.err = [e / (i + 1) for e in self.err]
        self.err = sum(self.err) / len(self.err)

    def test_rnn(self, inputs: np.ndarray, outputs: np.ndarray, loss_fun=None) -> None:
        """
        feeding forward a all testing data points and getting
        the output to calculate the error between the output
        and the groundtruth
        """
        with torch.no_grad():
            if not loss_fun:
                loss_fun = self.loss_fun

            err = [torch.tensor(0.0) for i in range(len(outputs[0]))]
            for i in range(len(inputs) - self.lags):
                res = self.feedforward(inputs[i : i + self.lags])
                err = [e + loss for e, loss in zip(err, loss_fun(res, outputs[i]))]
            err = [e / (i + 1) for e in err]
            err = sum(err) / len(err)

            self.fitness = err.item()

    def feedbackward(
        self,
    ) -> None:
        """
        calculating gradients
        """
        logger.debug(f"ERR: {self.err}")
        self.err.backward()
        for node in self.nodes.values():
            for edge in node.fan_out.values():
                with torch.no_grad():
                    logger.debug(
                        f"From Point {node.point.id} To Point \
                        {edge.out_node.point.id}: dweight = {edge.weight.grad}"
                    )
                    logger.debug(f"\t Weight before update: {edge.weight}")
                    edge.weight -= edge.weight.grad
                    logger.debug(f"\t Weight after update: {edge.weight}")
                edge.weight.grad.zero_()
